#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch
import joblib
from decord import VideoReader
from PIL import Image
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_DIR = "/root/OVO-Bench-full"
MODEL_PATH = "/root/models/MiniCPM-o-4_5"
ANNO_PATH = os.path.join(BASE_DIR, "annotations", "ovo_bench_new.json")
VIDEO_ROOT = os.path.join(BASE_DIR, "data", "src_videos")
RESULTS_CSV = os.path.join(BASE_DIR, "results", "minicpmo45_full_v2_results.csv")
SUMMARY_JSON = os.path.join(BASE_DIR, "results", "minicpmo45_full_v2_summary.json")

STREAM_FPS = 1
MAX_SLICE_NUMS = 1
SELECTOR_MODE = "decision_tree"
RESIZE_MODE = None
CHANGE_THRESHOLD = 0.03
MAX_GAP_SECONDS = 2
TREE_PATH = "/root/OVO-Bench-mini/results/minicpmo45_token_v1_decision_tree.joblib"

BACKWARD_TASKS = {"EPM", "ASI", "HLD"}
REALTIME_TASKS = {"OCR", "ACR", "ATR", "STU", "FPD", "OJR"}
FORWARD_TASKS = {"REC", "SSR", "CRR"}

BR_PROMPT_TEMPLATE = """
Question: {}
Options:
{}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
Do not include any additional text or explanation in your response.
"""

REC_PROMPT_TEMPLATE = """
You're watching a video in which people may perform a certain type of action repetively. 
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one. 
Now, answer the following question: {}
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
"""

SSR_PROMPT_TEMPLATE = """
You're watching a tutorial video which contain a sequential of steps. 
The following is one step from the whole procedures: 
{}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.
"""

CRR_PROMPT_TEMPLATE = """
You're responsible of answering questions based on the video content. 
The following question are relevant to the latest frames, i.e. the end of the video.
{}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.
"""

FIELD_NAMES = [
    "video_id",
    "query_id",
    "task",
    "category",
    "query_timestamp",
    "question",
    "choices",
    "ground_truth",
    "prediction",
    "raw_output",
    "correct",
    "invalid",
    "processing_error",
    "status",
    "frames_seen",
    "candidate_seconds_seen",
    "frames_dropped",
    "actual_sampled_fps",
    "prefill_time_total",
    "prefill_time_avg",
    "query_prefill_time",
    "generation_time",
    "peak_vram_gb",
    "avg_actual_slices_per_frame",
    "avg_visual_tokens_per_frame",
    "visual_tokens_seen",
    "input_tokens_seen",
    "live_kv_tokens_at_query",
]

CATEGORY_BY_TASK = {}
for task in BACKWARD_TASKS:
    CATEGORY_BY_TASK[task] = "Backward Tracing"
for task in REALTIME_TASKS:
    CATEGORY_BY_TASK[task] = "Real-Time Visual Perception"
for task in FORWARD_TASKS:
    CATEGORY_BY_TASK[task] = "Forward Active Responding"


def safe_video_id(video: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", video).strip("_")


def clamp_seconds(value, video_duration, total_frames):
    value = max(0, int(math.ceil(float(value))))
    max_available = max(0, int(video_duration))
    if total_frames == 0:
        return 0
    if max_available <= 0:
        return min(value, 1)
    return min(value, max_available)


def get_actual_slice_info(model, image, max_slice_nums):
    """Use the real processor output to count actual slices and visual tokens."""
    processed = model.processor.process_image(
        [image],
        max_slice_nums=max_slice_nums,
        return_tensors="pt",
    )
    tgt = processed["tgt_sizes"][0]
    if hasattr(tgt, "tolist"):
        tgt_list = tgt.tolist()
    else:
        tgt_list = [list(row) for row in tgt]
    slice_count = len(tgt_list)
    patches_per_slice = [int(int(row[0]) * int(row[1])) for row in tgt_list]
    visual_tokens = slice_count * model.processor.image_processor.image_feature_size
    return {
        "slice_count": slice_count,
        "visual_tokens": visual_tokens,
        "patches": sum(patches_per_slice),
        "tgt_sizes": tgt_list,
    }


def resize_image_for_mode(image, resize_mode):
    if not resize_mode or resize_mode == "orig":
        return image
    if resize_mode == "960x540":
        return image.resize((960, 540), Image.LANCZOS)
    if resize_mode == "448x252":
        return image.resize((448, 252), Image.LANCZOS)
    return image


def to_change_tensor(frame_array):
    small = Image.fromarray(frame_array).convert("L").resize((64, 64))
    return np.asarray(small, dtype=np.float32) / 255.0


def frame_change_score(last_gray, current_gray):
    if last_gray is None:
        return float("inf")
    return float(np.mean(np.abs(current_gray - last_gray)))


def make_tree_feature(last_sent_gray, prev_gray, current_gray, ema_change, gap, norm_time, duration):
    change_last = frame_change_score(last_sent_gray, current_gray)
    if np.isinf(change_last):
        change_last = 0.0
    change_prev = frame_change_score(prev_gray, current_gray)
    if np.isinf(change_prev):
        change_prev = 0.0
    return np.asarray(
        [change_last, change_prev, float(ema_change), float(gap), float(norm_time), float(duration)],
        dtype=np.float32,
    ).reshape(1, -1)


def extract_letter(text: str):
    if not text:
        return ""
    patterns = [
        r"(?:final answer|answer|option|correct|letter)\s*[:：]?\s*([A-Ha-h])\b",
        r"\b([A-Ha-h])\s*[\.。\)]",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search(r"\b([A-Ha-h])\b", text, flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""


def extract_yes_no(text: str):
    if not text:
        return ""
    t = text.strip()
    if re.fullmatch(r"[Yy]", t):
        return "Yes"
    if re.fullmatch(r"[Nn]", t):
        return "No"
    m = re.search(r"\b(yes|no)\b", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    m = re.search(r"\b([yn])\b", t, flags=re.IGNORECASE)
    if m:
        return "Yes" if m.group(1).lower() == "y" else "No"
    return ""


def extract_digits(text: str):
    if not text:
        return ""
    return "".join(re.findall(r"\d+", text))


def build_br_prompt(question, options):
    formatted_options = "; ".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options)) + ";"
    return BR_PROMPT_TEMPLATE.format(question, formatted_options)


def expand_events(annotations):
    events_by_video = defaultdict(list)
    video_meta = {}

    for item in annotations:
        video = item["video"]
        if video not in video_meta:
            video_meta[video] = {"task_counts": defaultdict(int), "query_count": 0, "max_realtime": 0.0}
        task = item["task"]
        video_meta[video]["task_counts"][task] += 1

        if task in BACKWARD_TASKS or task in REALTIME_TASKS:
            realtime = float(item["realtime"])
            options = item.get("options") or []
            choices = "; ".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options)) + ";"
            prompt = build_br_prompt(item["question"], options)
            event = {
                "query_id": str(item["id"]),
                "task": task,
                "category": CATEGORY_BY_TASK[task],
                "realtime": realtime,
                "needed_seconds": clamp_seconds(realtime, 0, 1),
                "kind": "mc",
                "question": item["question"],
                "choices": choices,
                "ground_truth": chr(65 + int(item["gt"])),
                "prompt": prompt,
                "test_type": None,
                "count": None,
                "step": None,
            }
            # Correct clamp after video metadata is known. We store raw realtime and clamp per-video later.
            event["needed_seconds"] = int(math.ceil(realtime))
            events_by_video[video].append(event)
            video_meta[video]["query_count"] += 1
            video_meta[video]["max_realtime"] = max(video_meta[video]["max_realtime"], realtime)

        elif task in FORWARD_TASKS:
            test_info = item.get("test_info") or []
            for j, ti in enumerate(test_info):
                realtime = float(ti["realtime"])
                query_id = f"{item['id']}_{j}"
                if task == "REC":
                    question = "How many times did they " + item["activity"] + "?"
                    prompt = REC_PROMPT_TEMPLATE.format(question)
                    choices = ""
                    ground_truth = str(ti["count"])
                    kind = "rec"
                    step = None
                    test_type = None
                elif task == "SSR":
                    step = ti["step"]
                    question = f"Is the current step: {step}"
                    prompt = SSR_PROMPT_TEMPLATE.format(step)
                    choices = ""
                    ground_truth = "Yes" if ti["type"] == 1 else "No"
                    kind = "ssr"
                    test_type = int(ti["type"])
                elif task == "CRR":
                    question = item["question"]
                    prompt = CRR_PROMPT_TEMPLATE.format(question)
                    choices = ""
                    ground_truth = "Yes" if ti["type"] == 1 else "No"
                    kind = "crr"
                    step = None
                    test_type = int(ti["type"])
                else:
                    continue

                event = {
                    "query_id": query_id,
                    "task": task,
                    "category": CATEGORY_BY_TASK[task],
                    "realtime": realtime,
                    "needed_seconds": int(math.ceil(realtime)),
                    "kind": kind,
                    "question": question,
                    "choices": choices,
                    "ground_truth": ground_truth,
                    "prompt": prompt,
                    "test_type": test_type,
                    "count": ti.get("count") if task == "REC" else None,
                    "step": step,
                }
                events_by_video[video].append(event)
                video_meta[video]["query_count"] += 1
                video_meta[video]["max_realtime"] = max(video_meta[video]["max_realtime"], realtime)

    for video, events in events_by_video.items():
        events.sort(key=lambda e: (e["needed_seconds"], e["query_id"]))
        video_meta[video]["query_count"] = len(events)
    return events_by_video, video_meta


def score_event(event, raw_output):
    if raw_output is None:
        return False, "", True, False, "invalid"

    raw_output = str(raw_output)
    kind = event["kind"]
    if kind == "mc":
        gt_letter = event["ground_truth"]
        pred = extract_letter(raw_output)
        invalid = pred == ""
        correct = (not invalid) and (gt_letter in raw_output)
        return correct, pred, invalid, False, "invalid" if invalid else "ok"

    if kind == "rec":
        pred = extract_digits(raw_output)
        invalid = pred == ""
        correct = (not invalid) and (pred == str(event["ground_truth"]))
        return correct, pred, invalid, False, "invalid" if invalid else "ok"

    if kind in ("ssr", "crr"):
        pred = extract_yes_no(raw_output)
        invalid = pred == ""
        gt_type = int(event["test_type"])
        raw_stripped = raw_output.strip()
        if (raw_stripped == "N" and gt_type == 0) or (raw_stripped == "Y" and gt_type == 1):
            correct = True
        else:
            gt = "No" if gt_type == 0 else "Yes"
            correct = gt in raw_output
        return correct, pred, invalid, False, "invalid" if invalid else "ok"

    return False, "", True, False, "invalid"


def load_existing_rows(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["correct", "invalid", "processing_error"]:
                if key in row and isinstance(row[key], str):
                    row[key] = row[key].strip().lower() in ("true", "1", "yes")
            rows.append(row)
    return rows


def append_row(path, row):
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_NAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--video-filter", type=str, default=None)
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--output-tag", type=str, default=None)
    parser.add_argument("--selector", type=str, choices=["all", "change_anchor", "decision_tree"], default="decision_tree")
    parser.add_argument("--resize-mode", type=str, choices=["orig", "960x540", "448x252"], default="orig")
    parser.add_argument("--change-threshold", type=float, default=0.03)
    parser.add_argument("--max-gap-seconds", type=int, default=2)
    parser.add_argument("--tree-path", type=str, default=None)
    args = parser.parse_args()

    global MAX_SLICE_NUMS, RESULTS_CSV, SUMMARY_JSON, SELECTOR_MODE, RESIZE_MODE, CHANGE_THRESHOLD, MAX_GAP_SECONDS, TREE_PATH
    MAX_SLICE_NUMS = int(args.max_slice_nums)
    SELECTOR_MODE = args.selector
    RESIZE_MODE = args.resize_mode
    CHANGE_THRESHOLD = float(args.change_threshold)
    MAX_GAP_SECONDS = int(args.max_gap_seconds)
    TREE_PATH = args.tree_path or TREE_PATH
    tag = args.output_tag or "full_v2"
    RESULTS_CSV = os.path.join(BASE_DIR, "results", f"minicpmo45_{tag}_results.csv")
    SUMMARY_JSON = os.path.join(BASE_DIR, "results", f"minicpmo45_{tag}_summary.json")
    print(f"max_slice_nums={MAX_SLICE_NUMS}")
    print(f"selector={SELECTOR_MODE} change_threshold={CHANGE_THRESHOLD} max_gap_seconds={MAX_GAP_SECONDS}")
    print(f"resize_mode={RESIZE_MODE}")
    print(f"tree_path={TREE_PATH}")
    print(f"results_csv={RESULTS_CSV}")
    print(f"summary_json={SUMMARY_JSON}")

    with open(ANNO_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if args.video_filter:
        filter_set = {x.strip() for x in args.video_filter.split(",") if x.strip()}
        annotations = [a for a in annotations if a["video"] in filter_set]
    if args.limit_videos:
        seen = []
        for a in annotations:
            if a["video"] not in seen:
                seen.append(a["video"])
        selected_videos = set(seen[: args.limit_videos])
        annotations = [a for a in annotations if a["video"] in selected_videos]

    events_by_video, video_meta = expand_events(annotations)
    selected_videos = list(events_by_video.keys())
    selected_videos.sort()
    print("Selected videos:", len(selected_videos))
    print("Selected queries:", sum(len(events_by_video[v]) for v in selected_videos))

    existing_rows = load_existing_rows(RESULTS_CSV)
    all_rows = existing_rows[:]
    completed = {(r["video_id"], r["query_id"]) for r in existing_rows}
    print("Resume completed queries:", len(completed))

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().cuda()
    torch.cuda.synchronize()
    model_load_time = time.perf_counter() - load_start
    print(f"Model loaded in {model_load_time:.2f}s")
    frame_tree = joblib.load(TREE_PATH)
    print(f"Frame decision tree loaded from {TREE_PATH}")

    torch.cuda.reset_peak_memory_stats()

    global_frame_latencies = []
    global_slice_counts = []
    global_frame_visual_tokens = []
    global_frame_input_ids = []
    video_stats = {}

    for video in selected_videos:
        events = events_by_video[video]
        pending = [e for e in events if (video, e["query_id"]) not in completed]
        if not pending:
            print(f"[{video}] already complete, skip")
            continue

        video_path = os.path.normpath(os.path.join(VIDEO_ROOT, video))
        print(f"[{video}] processing {len(pending)} pending queries; video_path={video_path}")

        if not os.path.exists(video_path):
            print(f"[{video}] missing video, marking processing_error")
            for e in pending:
                row = {
                    "video_id": video,
                    "query_id": e["query_id"],
                    "task": e["task"],
                    "category": e["category"],
                    "query_timestamp": e["realtime"],
                    "question": e["question"],
                    "choices": e["choices"],
                    "ground_truth": e["ground_truth"],
                    "prediction": "",
                    "raw_output": "MISSING_VIDEO",
                    "correct": False,
                    "invalid": False,
                    "processing_error": True,
                    "status": "processing_error",
                    "frames_seen": 0,
                    "candidate_seconds_seen": 0,
                    "frames_dropped": 0,
                    "actual_sampled_fps": 0.0,
                    "prefill_time_total": 0.0,
                    "prefill_time_avg": 0.0,
                    "query_prefill_time": 0.0,
                    "generation_time": 0.0,
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 6),
                }
                append_row(RESULTS_CSV, row)
                all_rows.append(row)
                completed.add((video, row["query_id"]))
            continue

        try:
            reader = VideoReader(video_path)
            total_frames = len(reader)
            original_fps = float(reader.get_avg_fps()) if total_frames else 0.0
            video_duration = (total_frames / original_fps) if original_fps else 0.0
        except Exception as exc:
            print(f"[{video}] failed to open video: {exc}")
            for e in pending:
                row = {
                    "video_id": video,
                    "query_id": e["query_id"],
                    "task": e["task"],
                    "category": e["category"],
                    "query_timestamp": e["realtime"],
                    "question": e["question"],
                    "choices": e["choices"],
                    "ground_truth": e["ground_truth"],
                    "prediction": "",
                    "raw_output": f"VIDEO_OPEN_ERROR: {exc}",
                    "correct": False,
                    "invalid": False,
                    "processing_error": True,
                    "status": "processing_error",
                    "frames_seen": 0,
                    "candidate_seconds_seen": 0,
                    "frames_dropped": 0,
                    "actual_sampled_fps": 0.0,
                    "prefill_time_total": 0.0,
                    "prefill_time_avg": 0.0,
                    "query_prefill_time": 0.0,
                    "generation_time": 0.0,
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 6),
                }
                append_row(RESULTS_CSV, row)
                all_rows.append(row)
                completed.add((video, row["query_id"]))
            continue

        max_needed = max(e["needed_seconds"] for e in events)
        max_needed = clamp_seconds(max_needed, video_duration, total_frames)
        session_id = f"ovobench_{safe_video_id(video)}"

        model.reset_session()
        sys_msg = model.get_sys_prompt(mode="omni", language="en")
        model.streaming_prefill(
            session_id=session_id,
            msgs=[sys_msg],
            omni_mode=True,
            is_last_chunk=False,
            tokenizer=tokenizer,
        )

        seen_seconds = 0
        candidate_seconds = 0
        video_frame_prefill_total = 0.0
        video_selector_time_total = 0.0
        video_frame_latencies = []
        video_total_slices = 0
        video_total_visual_tokens = 0
        video_total_frame_input_ids = 0
        current_snapshot = None
        current_snapshot_seconds = None
        last_sent_gray = None
        prev_gray = None
        ema_change = 0.0
        steps_since_sent = MAX_GAP_SECONDS

        decode_failed = False
        for e in pending:
            target_seconds = clamp_seconds(e["needed_seconds"], video_duration, total_frames)
            if target_seconds > max_needed:
                target_seconds = max_needed

            while candidate_seconds < target_seconds:
                second = candidate_seconds
                frame_idx = min(int(second * original_fps), total_frames - 1)
                try:
                    frame_array = reader[frame_idx].asnumpy()
                except Exception as exc:
                    print(f"[{video}] frame read error at second {second}: {exc}")
                    for pending_event in pending:
                        if (video, pending_event["query_id"]) in completed:
                            continue
                        error_row = {
                            "video_id": video,
                            "query_id": pending_event["query_id"],
                            "task": pending_event["task"],
                            "category": pending_event["category"],
                            "query_timestamp": pending_event["realtime"],
                            "question": pending_event["question"],
                            "choices": pending_event["choices"],
                            "ground_truth": pending_event["ground_truth"],
                            "prediction": "",
                            "raw_output": f"VIDEO_DECODE_ERROR: {exc}",
                            "correct": False,
                            "invalid": False,
                            "processing_error": True,
                            "status": "video_decode_error",
                            "frames_seen": seen_seconds,
                            "candidate_seconds_seen": candidate_seconds,
                            "frames_dropped": candidate_seconds - seen_seconds,
                            "actual_sampled_fps": 0.0,
                            "prefill_time_total": round(video_frame_prefill_total, 6),
                            "prefill_time_avg": round(video_frame_prefill_total / max(seen_seconds, 1), 6),
                            "query_prefill_time": 0.0,
                            "generation_time": 0.0,
                            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 6),
                            "avg_actual_slices_per_frame": round(video_total_slices / max(seen_seconds, 1), 6),
                            "avg_visual_tokens_per_frame": round(video_total_visual_tokens / max(seen_seconds, 1), 6),
                            "visual_tokens_seen": video_total_visual_tokens,
                            "input_tokens_seen": video_total_frame_input_ids,
                            "live_kv_tokens_at_query": model._get_kv_cache_length(),
                        }
                        append_row(RESULTS_CSV, error_row)
                        all_rows.append(error_row)
                        completed.add((video, pending_event["query_id"]))
                    decode_failed = True
                    break

                selector_start = time.perf_counter()
                current_gray = to_change_tensor(frame_array)
                change_prev = frame_change_score(prev_gray, current_gray)
                if np.isinf(change_prev):
                    change_prev = 0.0
                if candidate_seconds == 0:
                    ema_change = change_prev
                else:
                    ema_change = 0.8 * ema_change + 0.2 * change_prev
                norm_time = candidate_seconds / max(video_duration, 1.0)
                if SELECTOR_MODE == "all" or candidate_seconds == 0 or steps_since_sent >= MAX_GAP_SECONDS:
                    send_frame = True
                else:
                    feat = make_tree_feature(
                        last_sent_gray,
                        prev_gray,
                        current_gray,
                        ema_change,
                        steps_since_sent,
                        norm_time,
                        video_duration,
                    )
                    send_frame = bool(int(frame_tree.predict(feat)[0]) == 1)
                video_selector_time_total += time.perf_counter() - selector_start

                if not send_frame:
                    steps_since_sent += 1
                    prev_gray = current_gray
                    candidate_seconds += 1
                    if candidate_seconds % 60 == 0 or candidate_seconds == target_seconds:
                        print(f"[{video}] candidate {candidate_seconds}/{target_seconds}s sent={seen_seconds}")
                    continue

                image = Image.fromarray(frame_array).convert("RGB")
                image = resize_image_for_mode(image, RESIZE_MODE)
                slice_info = get_actual_slice_info(model, image, MAX_SLICE_NUMS)
                user_msg = {"role": "user", "content": [image]}
                model._next_round_id += 1
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.streaming_prefill(
                    session_id=session_id,
                    msgs=[user_msg],
                    omni_mode=True,
                    max_slice_nums=MAX_SLICE_NUMS,
                    is_last_chunk=False,
                    tokenizer=tokenizer,
                )
                torch.cuda.synchronize()
                if model._omni_chunk_history:
                    last_ids = model._omni_chunk_history[-1].get("input_ids")
                    chunk_len = int(last_ids.shape[1]) if last_ids is not None else 0
                else:
                    chunk_len = 0
                latency = time.perf_counter() - t0
                video_frame_prefill_total += latency
                video_frame_latencies.append(latency)
                global_frame_latencies.append(latency)
                global_slice_counts.append(slice_info["slice_count"])
                global_frame_visual_tokens.append(slice_info["visual_tokens"])
                global_frame_input_ids.append(chunk_len)
                video_total_slices += slice_info["slice_count"]
                video_total_visual_tokens += slice_info["visual_tokens"]
                video_total_frame_input_ids += chunk_len
                seen_seconds += 1
                last_sent_gray = current_gray
                prev_gray = current_gray
                steps_since_sent = 0
                candidate_seconds += 1

                if candidate_seconds % 60 == 0 or candidate_seconds == target_seconds:
                    print(f"[{video}] candidate {candidate_seconds}/{target_seconds}s sent={seen_seconds}")

            if decode_failed:
                break

            # snapshot for this exact causal prefix; reuse for multiple queries at the same timestamp
            if current_snapshot is None or current_snapshot_seconds != target_seconds:
                if current_snapshot is not None:
                    del current_snapshot
                    current_snapshot = None
                current_snapshot = model.save_speculative_snapshot()
                current_snapshot_seconds = target_seconds

            snapshot_kv_len = model._get_kv_cache_length()

            query_id = e["query_id"]
            row = {
                "video_id": video,
                "query_id": query_id,
                "task": e["task"],
                "category": e["category"],
                "query_timestamp": e["realtime"],
                "question": e["question"],
                "choices": e["choices"],
                "ground_truth": e["ground_truth"],
                "prediction": "",
                "raw_output": "",
                "correct": False,
                "invalid": False,
                "processing_error": False,
                "status": "processing_error",
                "frames_seen": seen_seconds,
                "candidate_seconds_seen": candidate_seconds,
                "frames_dropped": candidate_seconds - seen_seconds,
                "actual_sampled_fps": round(seen_seconds / max(float(e["needed_seconds"]), 1.0), 6),
                "prefill_time_total": round(video_frame_prefill_total, 6),
                "prefill_time_avg": round(video_frame_prefill_total / max(seen_seconds, 1), 6),
                "query_prefill_time": 0.0,
                "generation_time": 0.0,
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 6),
                "avg_actual_slices_per_frame": round(video_total_slices / max(seen_seconds, 1), 6),
                "avg_visual_tokens_per_frame": round(video_total_visual_tokens / max(seen_seconds, 1), 6),
                "visual_tokens_seen": video_total_visual_tokens,
                "input_tokens_seen": 0,
                "live_kv_tokens_at_query": snapshot_kv_len,
            }

            try:
                restored = model.restore_speculative_snapshot(current_snapshot)
                if restored:
                    model.new_user_msg = True
                if not restored:
                    raise RuntimeError("snapshot restore failed before query")

                question_msg = {"role": "user", "content": [e["prompt"]]}
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.streaming_prefill(
                    session_id=session_id,
                    msgs=[question_msg],
                    omni_mode=True,
                    is_last_chunk=True,
                    tokenizer=tokenizer,
                )
                torch.cuda.synchronize()
                row["query_prefill_time"] = round(time.perf_counter() - t0, 6)
                if model._omni_chunk_history:
                    q_ids = model._omni_chunk_history[-1].get("input_ids")
                    q_chunk_len = int(q_ids.shape[1]) if q_ids is not None else 0
                else:
                    q_chunk_len = 0
                row["input_tokens_seen"] = video_total_frame_input_ids + q_chunk_len

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                generator = model.streaming_generate(
                    session_id=session_id,
                    generate_audio=False,
                    use_tts_template=True,
                    enable_thinking=False,
                    do_sample=False,
                    max_new_tokens=128,
                    tokenizer=tokenizer,
                )
                response_text = ""
                for text_chunk, is_finished in generator:
                    response_text += text_chunk
                torch.cuda.synchronize()
                row["generation_time"] = round(time.perf_counter() - t0, 6)
                row["raw_output"] = response_text

                correct, pred, invalid, proc_error, status = score_event(e, response_text)
                row["correct"] = correct
                row["prediction"] = pred
                row["invalid"] = invalid
                row["processing_error"] = proc_error
                row["status"] = status

                # restore pure video prefix again after generation so QA text/answer does not contaminate the stream
                restored_after = model.restore_speculative_snapshot(current_snapshot)
                if restored_after:
                    model.new_user_msg = True
                if not restored_after:
                    print(f"[{video}/{query_id}] warning: snapshot restore after query failed")

                row["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 6)
                append_row(RESULTS_CSV, row)
                all_rows.append(row)
                completed.add((video, query_id))
                print(
                    f"[{video}/{query_id}] task={e['task']} t={e['realtime']} "
                    f"frames={seen_seconds} pred={pred or '-'} gt={e['ground_truth']} "
                    f"correct={correct} status={status}"
                )
            except Exception as exc:
                row["raw_output"] = f"ERROR: {exc}"
                row["correct"] = False
                row["invalid"] = False
                row["processing_error"] = True
                row["status"] = "processing_error"
                row["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 6)
                append_row(RESULTS_CSV, row)
                all_rows.append(row)
                completed.add((video, query_id))
                print(f"[{video}/{query_id}] processing error: {exc}")
                try:
                    model.restore_speculative_snapshot(current_snapshot)
                    model.new_user_msg = True
                except Exception:
                    pass

        if current_snapshot is not None:
            del current_snapshot
            current_snapshot = None

        if decode_failed:
            video_stats[video] = {
                "queries": len(pending),
                "video_duration": round(video_duration, 3),
                "total_frames": int(total_frames),
                "original_fps": round(original_fps, 3),
                "frames_prefilled": seen_seconds,
                "candidate_seconds": candidate_seconds,
                "frames_dropped": candidate_seconds - seen_seconds,
                "selector_time_total": round(video_selector_time_total, 6),
                "max_query_timestamp": round(max(e["realtime"] for e in events), 3),
                "frame_prefill_time_total": round(video_frame_prefill_total, 6),
                "average_frame_prefill_time": round(video_frame_prefill_total / max(len(video_frame_latencies), 1), 6),
                "rtf_frame_prefill": round(video_frame_prefill_total / max(candidate_seconds, 1), 6),
                "decode_failed": True,
            }
            continue

        video_stats[video] = {
            "queries": len(pending),
            "video_duration": round(video_duration, 3),
            "total_frames": int(total_frames),
            "original_fps": round(original_fps, 3),
            "frames_prefilled": seen_seconds,
            "candidate_seconds": candidate_seconds,
            "frames_dropped": candidate_seconds - seen_seconds,
            "selector_time_total": round(video_selector_time_total, 6),
            "max_query_timestamp": round(max(e["realtime"] for e in events), 3),
            "frame_prefill_time_total": round(video_frame_prefill_total, 6),
            "average_frame_prefill_time": round(video_frame_prefill_total / max(len(video_frame_latencies), 1), 6),
            "rtf_frame_prefill": round(video_frame_prefill_total / max(candidate_seconds, 1), 6),
        }

    # Compute summary from all rows (including resumed).
    if all_rows:
        summary_rows = all_rows
    else:
        summary_rows = []

    total = len(summary_rows)
    correct = sum(1 for r in summary_rows if bool(r.get("correct")))
    invalid = sum(1 for r in summary_rows if bool(r.get("invalid")))
    proc_error = sum(1 for r in summary_rows if bool(r.get("processing_error")))
    incorrect = total - correct - invalid - proc_error
    valid = total - invalid - proc_error

    per_task = defaultdict(lambda: {"total": 0, "correct": 0, "invalid": 0, "processing_error": 0})
    per_category = defaultdict(lambda: {"total": 0, "correct": 0, "invalid": 0, "processing_error": 0})
    for r in summary_rows:
        task = r.get("task", "UNKNOWN")
        category = r.get("category", "UNKNOWN")
        per_task[task]["total"] += 1
        per_category[category]["total"] += 1
        if bool(r.get("correct")):
            per_task[task]["correct"] += 1
            per_category[category]["correct"] += 1
        if bool(r.get("invalid")):
            per_task[task]["invalid"] += 1
            per_category[category]["invalid"] += 1
        if bool(r.get("processing_error")):
            per_task[task]["processing_error"] += 1
            per_category[category]["processing_error"] += 1

    task_acc = {}
    for task, s in per_task.items():
        task_acc[task] = round(s["correct"] / max(s["total"], 1), 6)

    category_raw_acc = {}
    for cat, s in per_category.items():
        category_raw_acc[cat] = round(s["correct"] / max(s["total"], 1), 6)

    # Official OVO metric is macro-averaged over tasks within each category, then macro-averaged over categories.
    official_category_acc = {}
    category_to_tasks = {
        "Backward Tracing": sorted(BACKWARD_TASKS),
        "Real-Time Visual Perception": sorted(REALTIME_TASKS),
        "Forward Active Responding": sorted(FORWARD_TASKS),
    }
    official_total = []
    for cat, tasks in category_to_tasks.items():
        accs = [task_acc[t] for t in tasks if t in task_acc]
        if accs:
            official_category_acc[cat] = round(sum(accs) / len(accs), 6)
            official_total.append(official_category_acc[cat])
    official_ovo = round(sum(official_total) / len(official_total), 6) if official_total else 0.0

    total_candidate_seconds = sum(max(int(v.get("candidate_seconds", 0)), 1) for v in video_stats.values())
    total_frame_prefill_time = sum(v.get("frame_prefill_time_total", 0.0) for v in video_stats.values())
    total_selector_time = sum(v.get("selector_time_total", 0.0) for v in video_stats.values())
    overall_rtf = total_frame_prefill_time / max(total_candidate_seconds, 1)
    candidate_sum_rows = sum(float(r.get("candidate_seconds_seen", 0) or 0) for r in summary_rows)
    dropped_sum_rows = sum(float(r.get("frames_dropped", 0) or 0) for r in summary_rows)

    summary = {
        "model_path": MODEL_PATH,
        "annotation_path": ANNO_PATH,
        "results_csv": RESULTS_CSV,
        "mode": "MiniCPM streaming_prefill + streaming_generate",
        "stream_fps": STREAM_FPS,
        "max_slice_nums": MAX_SLICE_NUMS,
        "resize_mode": RESIZE_MODE,
        "selected_videos": len(selected_videos),
        "total_queries": total,
        "valid_queries": valid,
        "correct": correct,
        "incorrect": incorrect,
        "invalid": invalid,
        "processing_error": proc_error,
        "raw_accuracy": round(correct / max(total, 1), 6),
        "valid_accuracy": round(correct / max(valid, 1), 6),
        "task_accuracy_raw": task_acc,
        "category_accuracy_raw": category_raw_acc,
        "official_category_accuracy_macro": official_category_acc,
        "official_ovo_accuracy_macro": official_ovo,
        "selector_mode": SELECTOR_MODE,
        "change_threshold": CHANGE_THRESHOLD,
        "max_gap_seconds": MAX_GAP_SECONDS,
        "average_frames_seen_per_query": round(
            sum(float(r.get("frames_seen", 0) or 0) for r in summary_rows) / max(total, 1), 6
        ),
        "average_candidate_seconds_seen_per_query": round(candidate_sum_rows / max(total, 1), 6),
        "average_frames_dropped_per_query": round(dropped_sum_rows / max(total, 1), 6),
        "frame_drop_rate": round(dropped_sum_rows / max(candidate_sum_rows, 1), 6),
        "effective_fps": round(
            sum(float(r.get("frames_seen", 0) or 0) for r in summary_rows) / max(candidate_sum_rows, 1), 6
        ),
        "average_prefill_latency_per_frame_s": round(
            sum(global_frame_latencies) / max(len(global_frame_latencies), 1), 6
        ),
        "p50_prefill_latency_s": round(percentile(global_frame_latencies, 50), 6),
        "p95_prefill_latency_s": round(percentile(global_frame_latencies, 95), 6),
        "max_prefill_latency_s": round(max(global_frame_latencies) if global_frame_latencies else 0.0, 6),
        "average_generation_time_s": round(
            sum(float(r.get("generation_time", 0) or 0) for r in summary_rows) / max(total, 1), 6
        ),
        "average_actual_slices_per_frame": round(
            sum(global_slice_counts) / max(len(global_slice_counts), 1), 6
        ),
        "p50_actual_slices_per_frame": round(percentile(global_slice_counts, 50), 6),
        "p95_actual_slices_per_frame": round(percentile(global_slice_counts, 95), 6),
        "max_actual_slices_per_frame": max(global_slice_counts) if global_slice_counts else 0,
        "average_visual_tokens_per_frame": round(
            sum(global_frame_visual_tokens) / max(len(global_frame_visual_tokens), 1), 6
        ),
        "average_visual_tokens_per_query": round(
            sum(float(r.get("visual_tokens_seen", 0) or 0) for r in summary_rows) / max(total, 1), 6
        ),
        "average_input_tokens_per_query": round(
            sum(float(r.get("input_tokens_seen", 0) or 0) for r in summary_rows) / max(total, 1), 6
        ),
        "average_live_kv_tokens_per_query": round(
            sum(float(r.get("live_kv_tokens_at_query", 0) or 0) for r in summary_rows) / max(total, 1), 6
        ),
        "total_candidate_video_seconds_processed": total_candidate_seconds,
        "total_frame_prefill_time_s": round(total_frame_prefill_time, 6),
        "total_selector_time_s": round(total_selector_time, 6),
        "overall_rtf_frame_prefill": round(overall_rtf, 6),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 6),
        "peak_vram_reserved_gb": round(torch.cuda.memory_reserved() / (1024 ** 3), 6),
        "model_load_time_s": round(model_load_time, 3),
        "video_stats": video_stats,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== MiniCPM-o 4.5 OVO-Bench Streaming Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
