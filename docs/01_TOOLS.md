# 第一步：看懂一次工具调用

运行 `python -m optical_agent.experiments demo_agent`，打开 `runs/tool_layer_demo.json`。默认例子只做真实质量计算和校正，不需要密钥或检测权重。

你会看到三次操作：评估原图 → 创建校正图 → 评估校正图。要加入真实目标检测，可以运行：

```powershell
python -m optical_agent.experiments demo_agent --image D:\my_underwater_image.jpg --output runs\my_tool_demo.json
```

| 概念 | 在项目中的含义 |
|---|---|
| Tool | 一项真实能力，例如质量评估 |
| Action | 工具名称加参数，例如 image_id=original |
| Registry | 从名称找到函数的注册表 |
| Context / State | 保存原图、校正图、检测器和结果的容器 |
| Observation | 工具执行后返回的结果，错误也属于结果 |

入口 `agent_tools.py` 导出四个学习用对象，真实实现位于 `optical_agent/tools.py`。先读 ToolContext、execute_tool，再读三个视觉函数。模型不会执行 Python；模型给出工具名称，程序才查表执行。

练习：观察第二次质量工具的 image_id 为什么是 corrected。重复检测同一张图时，trace 中 cached 为什么变成 true？
