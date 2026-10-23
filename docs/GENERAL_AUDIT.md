> 路径已脱敏；完整原记录留在本机D盘。该审查来自fresh GPT-6-Astra ultra，same-family / provisional，并非外部人工认证。PASS只评价本次留存实验的完整性，不代表视觉性能已达部署要求。

# Experiment Audit Report — General v0.6.0

**Date:** 2026-10-23
**Auditor:** GPT-6-Astra, ultra, fresh read-only reviewer
**Review independence:** same-family
**Acceptance status:** provisional
**Overall verdict:** PASS
**Integrity status:** pass

The retained experiment supports the scoped COCO visual baseline claims. No active concrete mismatch was found in ground truth, denominators, cached predictions, reported scores, calibration isolation or frozen source. This is a same-family provisional audit, not independent-family acceptance or field validation.

The auditor made no source edits, downloads, paid API calls, detector forward passes or LLM inference calls. Only these two audit files were written. Verification used direct file reads, hashes, official COCOeval reaggregation and deterministic calculations on existing caches.

## A. Ground-truth provenance: PASS

All 5,000 manifest image IDs exactly equal official COCO val2017 annotation IDs; every manifest ground-truth box/class/crowd/annotation ID matches the annotation source, and every cached GT matches the manifest. Original-image AP uses pycocotools.COCOeval. No model output becomes ground truth. SICE is explicitly not_run with no reference PSNR/SSIM.

Evidence: [experiments/prepare_general_data.py:270](PROJECT_ROOT/optical_agent/experiments/prepare_general_data.py:270); [experiments/prepare_general_data.py:276](PROJECT_ROOT/optical_agent/experiments/prepare_general_data.py:276); [experiments/evaluate_general_vision.py:112](PROJECT_ROOT/optical_agent/experiments/evaluate_general_vision.py:112); [experiments/evaluate_general_vision.py:131](PROJECT_ROOT/optical_agent/experiments/evaluate_general_vision.py:131); [general_v060/data_summary.json:33](LOCAL_DATA_ROOT/general_v060/data_summary.json:33); [sice/acquisition.json:37](LOCAL_DATA_ROOT/general_v060/sice/acquisition.json:37).

## B. Score normalization: PASS

IoU uses intersection over union; recall uses noncrowd GT counts; precision uses TP+FP; FP/image includes all evaluated images. No reported accuracy metric is divided by prediction max/min/mean. Raw COCO AP and operating-point metrics are separate. Image quality uses fixed 255 scaling and descriptive pixel statistics, expressly not exposure truth.

Evidence: [optical_agent/general_eval.py:150](PROJECT_ROOT/optical_agent/general_eval.py:150); [optical_agent/general_eval.py:159](PROJECT_ROOT/optical_agent/general_eval.py:159); [optical_agent/general_eval.py:188](PROJECT_ROOT/optical_agent/general_eval.py:188); [optical_agent/quality.py:71](PROJECT_ROOT/quality.py:71); [docs/GENERAL_EVALUATION.md:19](PROJECT_ROOT/docs/GENERAL_EVALUATION.md:19); [docs/GENERAL_EVALUATION.md:28](PROJECT_ROOT/docs/GENERAL_EVALUATION.md:28); [docs/GENERAL_EVALUATION.md:77](PROJECT_ROOT/docs/GENERAL_EVALUATION.md:77).

## C. Result existence, numbers and completion: PASS

All 5,990 expected unique cache artifacts exist, with no extra/missing keys or identity mismatches; they contain 5,790 non-reused forwards and 200 exact reused prediction arrays with zero inference_ms. All original and 36 diagnostic operating summaries, all three official AP summaries, the complete calibration object, development policy, fixed-0.35 test comparison and public quality aggregates reproduce. The public engineering log counts match. The one initially stale UI hash is superseded by a separately preserved final node verification whose seven hashes and log hash match current files.

Evidence: [general_v060/test_summary.json:5](LOCAL_DATA_ROOT/general_v060/test_summary.json:5); [general_v060/test_summary.json:663](LOCAL_DATA_ROOT/general_v060/test_summary.json:663); [general_v060/public_summary.json:19](LOCAL_DATA_ROOT/general_v060/public_summary.json:19); [general_v060/public_summary.json:84](LOCAL_DATA_ROOT/general_v060/public_summary.json:84); [general_v060/public_summary.json:126](LOCAL_DATA_ROOT/general_v060/public_summary.json:126); [public_cpu/final_frozen_verification.json:55](LOCAL_DATA_ROOT/general_v060/validation/public_cpu/final_frozen_verification.json:55); [validation/node_final_verification.json:5](LOCAL_DATA_ROOT/general_v060/validation/node_final_verification.json:5); [docs/GENERAL_EVALUATION.md:103](PROJECT_ROOT/docs/GENERAL_EVALUATION.md:103).

## D. Metric reachability and execution: PASS

All in-scope detection, calibration, quality-failure and latency metric functions are invoked by the evaluator and have outputs in real result files. Official COCOeval is called directly; hand-checked duplicate/crowd/empty-image/sparse-calibration/single-box AP tests cover meaningful metric boundaries. No claimed PSNR/SSIM or other dead metric was found.

Evidence: [experiments/evaluate_general_vision.py:131](PROJECT_ROOT/optical_agent/experiments/evaluate_general_vision.py:131); [experiments/evaluate_general_vision.py:141](PROJECT_ROOT/optical_agent/experiments/evaluate_general_vision.py:141); [experiments/evaluate_general_vision.py:319](PROJECT_ROOT/optical_agent/experiments/evaluate_general_vision.py:319); [experiments/evaluate_general_vision.py:328](PROJECT_ROOT/optical_agent/experiments/evaluate_general_vision.py:328); [tests/test_general_eval.py:17](PROJECT_ROOT/tests/test_general_eval.py:17); [tests/test_general_eval.py:115](PROJECT_ROOT/tests/test_general_eval.py:115); [public_cpu/official_eval.log:2](LOCAL_DATA_ROOT/general_v060/validation/public_cpu/official_eval.log:2).

## E. Scope, configurations and isolation: PASS

Evidence is one public COCO validation corpus, one split seed (20261023), 3,000/1,000/1,000 images, and 30 diagnostic base images per split under four variants and three methods. The text correctly distinguishes 360 correlated test rows from 30 independent base images, acknowledges public-data and scene-independence limits, 58 insufficient classes, 10/22 calibrated classes below 0.8 test precision, and SICE/real-LLM/RAG accuracy not evaluated. Freeze precedes retained test cache mtimes and all test signatures are frozen-current; old archives contain no test rows.

Evidence: [docs/GENERAL_EVALUATION.md:13](PROJECT_ROOT/docs/GENERAL_EVALUATION.md:13); [docs/GENERAL_EVALUATION.md:15](PROJECT_ROOT/docs/GENERAL_EVALUATION.md:15); [docs/GENERAL_EVALUATION.md:36](PROJECT_ROOT/docs/GENERAL_EVALUATION.md:36); [docs/GENERAL_EVALUATION.md:37](PROJECT_ROOT/docs/GENERAL_EVALUATION.md:37); [docs/GENERAL_EVALUATION.md:99](PROJECT_ROOT/docs/GENERAL_EVALUATION.md:99); [docs/GENERAL_EVALUATION.md:126](PROJECT_ROOT/docs/GENERAL_EVALUATION.md:126); [general_v060/freeze.json:53](LOCAL_DATA_ROOT/general_v060/freeze.json:53).

## F. Evaluation-type classification: PASS

COCO original detection and its calibration/test operating metrics are real_gt. Digital brightness/blackout experiments are simulation_only input experiments, with dataset GT retained for COCO detector scoring; they are not model-generated-reference synthetic_proxy. Image-only exposure/quality indicators and scripted workflow equivalence are self_supervised_proxy/engineering checks, not human_eval or semantic accuracy. SICE is not_run and real LLM acceptance is not_run.

Evidence: [experiments/evaluate_general_vision.py:18](PROJECT_ROOT/optical_agent/experiments/evaluate_general_vision.py:18); [experiments/evaluate_general_vision.py:22](PROJECT_ROOT/optical_agent/experiments/evaluate_general_vision.py:22); [optical_agent/quality.py:173](PROJECT_ROOT/quality.py:173); [runtime_visual/summary.json:2](LOCAL_DATA_ROOT/general_v060/validation/runtime_visual/summary.json:2); [recording/gif-manifest.json:15](LOCAL_DATA_ROOT/general_v060/recording/gif-manifest.json:15); [docs/GENERAL_EVALUATION.md:122](PROJECT_ROOT/docs/GENERAL_EVALUATION.md:122).

## Deterministic verification performed

| Item | Independently checked result |
|---|---|
| COCO image identity | All 5,000 IDs exactly match annotation image IDs; all 5,000 source-image file hashes match |
| Ground truth | Every manifest box, class, crowd flag and annotation ID matches the dataset; every cached GT matches its manifest row |
| Decoded sample | 14 distinct source images decoded; dimensions, pixel SHA, file SHA and pHash all match |
| Split/grouping | Seed 20261023 reproduces all assignments; 5,000 groups, no exact or pHash-radius-4 links; brute-force comparison of all stored pHashes also finds no close pair |
| Prediction artifacts | 5,990 expected and actual; no missing, extra or duplicate configuration/image keys |
| Unique cached forward evidence | Development 3,266; calibration 1,260; test 1,264; total 5,790 |
| Reuse evidence | 200 rows have the exact corresponding original prediction array and zero inference time |
| Operating metrics | All three original summaries and all 36 diagnostic configuration summaries reproduce, including class counts and latency aggregation |
| Official AP | Every numeric COCOeval field reproduces on development, calibration and test |
| Calibration/policy | Complete 80-class calibration JSON regenerates from 1,000 calibration rows only; development policy regenerates as original |
| Quality diagnostics | All 12 test configuration aggregates across seven reported image-quality measures reproduce within 1e-12 |
| Frozen source | All 86 recursive source hashes match both current project and source_frozen_v3; all six runtime-v2 signature files match source_runtime_v2 |
| Public claims | Fixed-0.35 comparison, calibrated subset statistics and checked-in machine summary match actual cache-derived values |
| Engineering evidence | Log totals and current final UI hashes match; native/LangGraph core results agree on three fixtures; MCP source/code hashes match |
| Recording | Exported public assets/demo.gif matches the recorded SHA, size, five frames, 860×833 dimensions and 20,000 ms duration; five source frame hashes match |

Counts refer to retained current evidence and exclude the explicitly archived first pass.

### Verified detection totals

| Split | Images | All annotations | Noncrowd GT | Crowd |
|---|---:|---:|---:|---:|
| Development | 3,000 | 21,844 | 21,578 | 266 |
| Calibration | 1,000 | 7,393 | 7,309 | 84 |
| Test | 1,000 | 7,544 | 7,448 | 96 |

Test official AP@[.50:.95] = **0.24120508670617372**; AP50 = **0.39455022761428565**. At frozen class thresholds, TP/FP/FN = **2,499 / 1,086 / 4,949**, micro recall = **0.3355263157894737**, macro recall = **0.368512215841278**, FP/image = **1.086**. Fixed 0.35 instead gives **2,708 / 1,525 / 4,740**, as publicly reported. The tradeoff is correctly described as reduced false positives and reduced recall.

Evidence: [general_v060/data_summary.json:33](LOCAL_DATA_ROOT/general_v060/data_summary.json:33); [general_v060/test_summary.json:5](LOCAL_DATA_ROOT/general_v060/test_summary.json:5); [general_v060/test_summary.json:663](LOCAL_DATA_ROOT/general_v060/test_summary.json:663); [general_v060/public_summary.json:19](LOCAL_DATA_ROOT/general_v060/public_summary.json:19).

## Freeze, cache bridge and held-out isolation

The raw development/calibration identity is retained, rather than overwritten. New test caches use the current frozen signature. Four visual implementation files are byte-identical across the bridge. Direct source diff shows the calibration change is the explicit insufficient-bin fallback to 0.35, plus new bridge/orchestration code; prediction, perturbation and scoring functions are unchanged.

The bridge proof was recomputed, and archived/current calibration and policy objects match exactly. All 58 insufficient classes retain 0.35. Full calibration recomputation produces the current calibration JSON exactly, so this is not merely reliance on a stored equality assertion. Recursive hashes, pre-test summary hashes, calibration/policy hashes, raw identity hash and bridge hash all match their freeze entries.

Freeze time is **2026-10-23 09:03:54.822873 UTC**. Earliest retained original test cache modification is **09:04:01.510501 UTC**; all test cache signatures match the freeze. The superseded first pass contains no test rows. Retained evidence is consistent with one frozen test and no held-out retuning; the audit does not turn file timestamps into a claim about unlogged activity.

Evidence: [optical_agent/general_eval.py:240](PROJECT_ROOT/optical_agent/general_eval.py:240); [experiments/evaluate_general_vision.py:168](PROJECT_ROOT/optical_agent/experiments/evaluate_general_vision.py:168); [experiments/evaluate_general_vision.py:254](PROJECT_ROOT/optical_agent/experiments/evaluate_general_vision.py:254); [experiments/evaluate_general_vision.py:311](PROJECT_ROOT/optical_agent/experiments/evaluate_general_vision.py:311); [general_v060/aggregation_identity.json:145](LOCAL_DATA_ROOT/general_v060/aggregation_identity.json:145); [general_v060/freeze.json:53](LOCAL_DATA_ROOT/general_v060/freeze.json:53); [general_v060/freeze.json:148](LOCAL_DATA_ROOT/general_v060/freeze.json:148).

Representative inspected cache records include calibration image **139**, test image **285**, development image **23937**, and their synthetic/correction variants. The audit program validated identity and GT for every cache, not only these samples. Evidence: [original/139.json:2](LOCAL_DATA_ROOT/general_v060/predictions/original/original/139.json:2); [original/285.json:2](LOCAL_DATA_ROOT/general_v060/predictions/original/original/285.json:2); [original/285.json:60](LOCAL_DATA_ROOT/general_v060/predictions/original/original/285.json:60).

## Resolved provenance discrepancy

The historical public CPU verification binds static/conversation.js to SHA **bc6422cb04768ce2c8be4ea52f8965619809b13ac0bfd7141f711d356011350c**. The current file is **2c9c0668578f3369c9b2f860d3d2a5a5b62195d101373a1e95b416976c45ecbe**. This was reported during the audit.

A separately retained final verification explicitly supersedes that historical UI state. All seven hashes in that final record and its log SHA match the actual current files; the log contains 12 + 6 + 4 + 5 = 27 passing cases. The old record was not rewritten. This is **resolved**, not an active integrity finding, and has no effect on the frozen COCO evaluation.

Evidence: [public_cpu/final_frozen_verification.json:55](LOCAL_DATA_ROOT/general_v060/validation/public_cpu/final_frozen_verification.json:55); [validation/node_final_verification.json:5](LOCAL_DATA_ROOT/general_v060/validation/node_final_verification.json:5); [validation/node_final_verification.json:7](LOCAL_DATA_ROOT/general_v060/validation/node_final_verification.json:7); [validation/node_final_verification.json:15](LOCAL_DATA_ROOT/general_v060/validation/node_final_verification.json:15).

## Evaluation classification and claim impact

| Evidence | Classification | Supported claim ceiling |
|---|---|---|
| Original COCO detector AP and operating scores | real_gt | This public-validation project split and frozen configuration |
| COCO synthetic brightness diagnostics | simulation_only input experiment, retaining real_gt boxes | Behavior under the declared digital transformations; not real recapture |
| Luminance/quality, no-GT workflow and MCP comparisons | self_supervised_proxy / engineering checks | Measured image statistics, protocol behavior and fixture equivalence |
| SICE multi-exposure/reference evaluation | not_run | Acquisition failure/status only |
| Actual LLM scheduling and new RAG retrieval accuracy | not_run | No new acceptance or semantic-accuracy claim |
| Human evaluation | not_run | None |

The paper-style fraud category synthetic_proxy does not describe the COCO GT: references were not generated by a model. A synthetic input alone does not make the retained dataset boxes model-generated ground truth.

SICE acquisition records five author-source attempts, preserves both Drive quota-exceeded HTML responses, and has no imported SICE input corpus. PSNR/SSIM is explicitly not computed. Runtime and recording artifacts identify deterministic offline scheduling and no object annotations. The runtime request counters are fixture bookkeeping; they are not paid model-call evidence.

The 22 calibrated classes have aggregate test precision **1612/(1612+377) = 0.8104575163398693**, yet **10/22** are below 0.8 individually. This limitation is publicly disclosed and the thresholds remain frozen. Quality-pass rate 100% on test originals describes the fixed near-endpoint heuristic, not 100% physically correct exposure.

## Action items

No corrective action is required for the current scoped experiment claims. Preserve the existing distinctions between detector accuracy, synthetic exposure diagnostics, image-only quality indicators and offline engineering checks. Keep the final UI validation alongside its historical predecessor.

No claim of real SICE exposure performance, improved enhancement accuracy, universal per-class precision, independent unseen scenes, true LLM acceptance, or new RAG accuracy is supported by this experiment; the current documentation explicitly avoids those claims.

Machine-readable evidence, exact audited input hashes, aggregate prediction-corpus digest and deterministic verification outputs are in **EXPERIMENT_AUDIT.json**. All semantic conclusions remain **same-family / provisional**.
