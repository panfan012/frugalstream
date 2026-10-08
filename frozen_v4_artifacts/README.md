# V4 Frozen — Quick Start

Official V4 policy:

risk <= 0.3554:
    C070
else:
    Full HR

C070:
center crop
70% image width
70% image height
(49% source area)

Target backend:
black-box MiniCPM-o 4.5

Temporal policy:
frozen V3 Decision Tree

Router signal:
frozen query-risk (tfidf_score)

One final target-VLM generation per query.

See V4_MANIFEST.md, V4_VERSION_INFO.json, and V4_STRICT_OOF_VALIDATION_REPORT.md for details.
