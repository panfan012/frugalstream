# 管理员 GPU 复位操作指南

## 背景（给管理员看）
宿主机上一张 `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
（PCI `0000:02:00.0`，驱动 `580.126.09`，共 97887 MiB）被容器内一个
vLLM 进程异常退出后“卡住”：

- 进程已经全部退出，容器内看不到任何存活 GPU 进程；
- 但 GPU 仍显示 `97141 MiB used / 74 MiB free`、`utilization 100%`、功耗 600W；
- 这是 CUDA 驱动层的显存泄漏 + compute 引擎挂起，不是还有进程在跑；
- 容器内无法自行复位：`/sys` 只读、无 `CAP_SYS_ADMIN/SYS_RAWIO`，
  且 `nvidia-smi --gpu-reset` 被拒（GPU 被识别为 primary GPU，绑有 fb0）。

请在宿主机上执行以下步骤，目标是把这张卡复位到空闲状态。

---

## 第 1 步：确认没有其他人的进程在用它（重要）
```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
sudo fuser -v /dev/nvidia0   # 若没有 fuser 可跳过
```

判断：
- 如果进程表为空、但显存还是满的 → 就是泄漏状态，可以放心复位。
- 如果进程表里还有别的 PID（例如 MERA / 其他租户）→ 复位会打断它们，
  请先与相关用户确认，不要直接复位。

## 第 2 步：先尝试标准软件复位
```bash
sudo nvidia-smi --gpu-reset -i 0
```

## 第 3 步：如果报 “GPU is the primary GPU”
这是因为它被当成主显示设备（NVIDIA 卡上绑着 fb0 帧缓冲）。这是台无显示
的服务器，解绑后再复位即可。

### 3a. 解绑控制台（如果存在绑定 NVIDIA 的 vtconsole）
```bash
ls /sys/class/vtconsole/
for c in /sys/class/vtconsole/vtcon*/name; do echo "$c: $(cat $c)"; done
# 找到绑定在 NVIDIA 卡上的 console（不是 dummy 的那个），记下 vtconX
echo 0 | sudo tee /sys/class/vtconsole/vtconX/bind
```

### 3b. 解绑 NVIDIA 上的 framebuffer（如果 fb0 属于 NVIDIA）
```bash
ls -l /sys/class/graphics/fb0/device
readlink -f /sys/class/graphics/fb0/device
# 确认它指向 0000:02:00.0（NVIDIA）
readlink /sys/class/graphics/fb0/device/driver
# 若驱动是 efi-framebuffer：
#   echo "$(basename /sys/class/graphics/fb0/device)" | sudo tee /sys/bus/platform/drivers/efi-framebuffer/unbind
# 若驱动是 simple-framebuffer：
#   echo "$(basename /sys/class/graphics/fb0/device)" | sudo tee /sys/bus/platform/drivers/simple-framebuffer/unbind
```

### 3c. 再次复位
```bash
sudo nvidia-smi --gpu-reset -i 0
```

### 3d. 复位成功后恢复控制台绑定（可跳过，SSH 不受影响）
```bash
echo 1 | sudo tee /sys/class/vtconsole/vtconX/bind
```

## 第 4 步：仍失败就直接重启宿主机（最稳妥）
```bash
sudo reboot
```

## 第 5 步：验证复位成功
```bash
nvidia-smi
```
期望看到：
- `Memory-Usage` 降至几百 MiB 以内（约 `0-500 MiB / 97887 MiB`）；
- `GPU-Util` 为 `0 %`；
- 进程表为空。

## 第 6 步：完成后
管理员不需要再启动任何模型。GPU 空闲后，容器内会自动重新拉起
Qwen3-Omni 服务并继续实验。

---

### 联系信息 / 日志参考
- 容器内状态文件：`/workspace/frugalstream_qwen3_four_point/STATUS.md`
- 旧 Qwen 服务日志：`/workspace/qwen3_omni_deploy/logs/qwen3_omni_server.log`
- GPU 监控历史：`/workspace/focalstream_qwen3_crossmodel/logs/gpu_monitor.log`
- 涉及的进程（已退出）：`516282 / 516443 / 516444 / 516764`
