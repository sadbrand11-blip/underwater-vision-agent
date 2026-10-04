# 0.5.4：统一实验入口

常用启动、图片分析和 MCP 服务入口保持兼容。实验命令迁入包内，原参数保持；总帮助不加载 Torch、权重或访问网络。

```text
python -m optical_agent.experiments --help
python -m optical_agent.experiments --list
```

旧实验命令需要改为下表入口。历史报告描述原版本行为，复现冻结实验应使用其对应代码、数据和语料版本；不要用新版本覆盖旧运行。

| 旧命令 | 新命令 |
|---|---|
| `python build_rag_questions.py …` | `python -m optical_agent.experiments build_rag_questions …` |
| `python calibrate_detector.py …` | `python -m optical_agent.experiments calibrate_detector …` |
| `python calibrate_quality.py …` | `python -m optical_agent.experiments calibrate_quality …` |
| `python demo_agent.py …` | `python -m optical_agent.experiments demo_agent …` |
| `python demo_independent_eval.py …` | `python -m optical_agent.experiments demo_independent_eval …` |
| `python demo_langgraph.py …` | `python -m optical_agent.experiments demo_langgraph …` |
| `python demo_mcp.py …` | `python -m optical_agent.experiments demo_mcp …` |
| `python demo_memory.py …` | `python -m optical_agent.experiments demo_memory …` |
| `python demo_task_validation.py …` | `python -m optical_agent.experiments demo_task_validation …` |
| `python diagnose_robot_training.py …` | `python -m optical_agent.experiments diagnose_robot_training …` |
| `python download_underwater_data.py …` | `python -m optical_agent.experiments download_underwater_data …` |
| `python download_vision_data.py …` | `python -m optical_agent.experiments download_vision_data …` |
| `python evaluate_adaptive.py …` | `python -m optical_agent.experiments evaluate_adaptive …` |
| `python evaluate_agent.py …` | `python -m optical_agent.experiments evaluate_agent …` |
| `python evaluate_detector.py …` | `python -m optical_agent.experiments evaluate_detector …` |
| `python evaluate_orchestration.py …` | `python -m optical_agent.experiments evaluate_orchestration …` |
| `python evaluate_quality_pairs.py …` | `python -m optical_agent.experiments evaluate_quality_pairs …` |
| `python evaluate_rag.py …` | `python -m optical_agent.experiments evaluate_rag …` |
| `python evaluate_underwater_illumination.py …` | `python -m optical_agent.experiments evaluate_underwater_illumination …` |
| `python evaluate_vision_detectors.py …` | `python -m optical_agent.experiments evaluate_vision_detectors …` |
| `python evaluate_vision_exposure.py …` | `python -m optical_agent.experiments evaluate_vision_exposure …` |
| `python prepare_rag.py …` | `python -m optical_agent.experiments prepare_rag …` |
| `python prepare_rag_corpus.py …` | `python -m optical_agent.experiments prepare_rag_corpus …` |
| `python prepare_vision_data.py …` | `python -m optical_agent.experiments prepare_vision_data …` |
| `python publish_rag.py …` | `python -m optical_agent.experiments publish_rag …` |
| `python reconcile_vision_calibration.py …` | `python -m optical_agent.experiments reconcile_vision_calibration …` |
| `python report_adaptive.py …` | `python -m optical_agent.experiments report_adaptive …` |
| `python report_vision.py …` | `python -m optical_agent.experiments report_vision …` |
| `python summarize_formal_eval.py …` | `python -m optical_agent.experiments summarize_formal_eval …` |
| `python train_detector.py …` | `python -m optical_agent.experiments train_detector …` |
| `python train_vision_detectors.py …` | `python -m optical_agent.experiments train_vision_detectors …` |
| `python verify_vision_web.py …` | `python -m optical_agent.experiments verify_vision_web …` |
