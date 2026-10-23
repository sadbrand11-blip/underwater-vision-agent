# 0.6.0 — General natural-image experiment

- Daily interface now targets ordinary8-bitJPEG/PNG photos and COCO80 common objects, with the official complete detection head and preprocessing. Underwater versions remain archived.
- Quality and bounded correction work without a detector. Missing weights produce an explicit unavailable result, never a fake empty detection.
- General quality heuristics, local/ROI evidence, Chinese class aliases, strict unsupported targets, independent RAG and memory namespaces. Native/LangGraph, MCP and existing limits remain.
- COCO project split3,000/1,000/1,000; the one held-out visual test reports AP24.12%, AP50 39.46%. Thresholds trade recall for fewer false positives. Only22/80 classes met calibration-set rules;10 of those remain below80% per-class precision on test. No guarantee of individual-box correctness.
- Neither correction method passed the predeclared development gate, so the evaluated installation retains originals by default. Explicit candidate comparison remains available. Clean clones do not bundle local calibration artifacts.
- CPU exposure Quick Start needs no weights or key. Optional detection downloads the SHA256-verified official PyTorch checkpoint, with upstream notices retained.
- SICE author downloads were unavailable (Drive quota / Baidu client requirements); real multi-exposure and PSNR/SSIM evaluation remain not run. Synthetic COCO exposure variants are labeled.
- No training or paid API calls. Runtime demos use deterministic offline rules with actual visual tools; these are not a new formal LLM Agent Eval.

See [evaluation](GENERAL_EVALUATION.md), [data and reproducibility](GENERAL_DATA.md), [Chinese Quick Start](QUICK_START_ZH.md), [current MCP](GENERAL_MCP.md), and [underwater archive](ARCHIVE_UNDERWATER.md).
