# 中文操作指南

## 先运行无需密钥、权重的演示

使用 Python 3.11，在 PowerShell 依次执行：

```powershell
git clone https://github.com/sadbrand11-blip/underwater-vision-agent.git
cd underwater-vision-agent
python -m venv D:\CodexData\optical_agent\public\venv
$agentPython = "D:\CodexData\optical_agent\public\venv\Scripts\python.exe"
& $agentPython -m pip install -r requirements-demo.txt
& $agentPython run_demo.py
```

打开 `http://127.0.0.1:7860/showcase`。选择 **Assess exposure only** 或 **Correct exposure only**，点击 **Run local tools**。示例来自许可明确的SODD图片，也可以上传自己的图片。

这里执行真实曝光评估与校正，调度方式是离线脚本。没有检测权重时明确显示“检测不可用”，不显示假的空检测结果。7860被占用时，用 `run_demo.py --port 7861`。

## 启用真实目标检测

先停止本项目演示服务，再在项目目录执行：

```powershell
$agentPython = "D:\CodexData\optical_agent\public\venv\Scripts\python.exe"
& $agentPython -m pip install -r requirements-detector.txt --index-url https://download.pytorch.org/whl/cpu
& $agentPython run_demo.py --with-detector
```

启动器下载既有Release中的SODD六类权重，校验大小与SHA256，使用CPU加载。权重约76MB，不进入Git历史。六类为 `propeller`、`pipe_type2`、`red_fin`、`net`、`qr_codes`、`pipe`；“管道”对应两类pipe。

选择 **Assess, correct if needed, detect pipes & verify** 可以查看实际工具、候选与程序可靠性原因。框和分数仍是候选证据；“不可靠”是保留的真实结果。偏暗×0.6和全黑示例是模拟扰动，不代表真实相机重拍。

## 对话页面与高级能力

打开 `/agent`：选择图片、输入任务、查看结果。默认native动态调度，离线脚本支持示例任务。高级设置可展开视觉配置、RAG、LangGraph和旧调度；未安装的功能会说明原因。

自由中文任务需将 `.env.example` 复制为 `.env`，在本地填写API配置，重新启动并明确选择云端模式。不要将实际密钥提交到Git。云端调用可能产生费用；图像处理留在本地。

结果首先显示回答、任务状态、视觉可靠性、原图与采用图。计划、候选、热图、引用和执行记录可展开；错误详情默认显示。显示偏好在 `/memory` 设置。

[可选依赖与接口](OPTIONAL_MODES.md) · [实验命令迁移](CLI_MIGRATION.md) · [架构与功能清单](ARCHITECTURE.md)

## 数据目录与其他平台

Windows默认数据根目录为 `D:\CodexData\optical_agent`，公开版权重、日志、记忆和RAG缓存位于其 `public` 子目录。`OPTICAL_AGENT_DATA_ROOT` 表示数据根目录，`public` 子目录由程序创建。

其他平台启动前设置 `OPTICAL_AGENT_DATA_ROOT`，在该目录下创建虚拟环境，然后安装 `requirements-demo.txt` 并运行 `run_demo.py`。例如：

```sh
export OPTICAL_AGENT_DATA_ROOT="$PWD/.local-data"
python3.11 -m venv "$OPTICAL_AGENT_DATA_ROOT/venv"
"$OPTICAL_AGENT_DATA_ROOT/venv/bin/python" -m pip install -r requirements-demo.txt
"$OPTICAL_AGENT_DATA_ROOT/venv/bin/python" run_demo.py
```

不要把本地数据目录提交到Git。公开语料是 `public-sanitized-v2`，默认TF-IDF，历史C1检索分数不适用于它。公开版不提供机器人实验权重。

## 常见状态

- 任务完成、质量失败：分析已正确执行，但图像不能支持可靠判断。
- 未完成：缺少必要证据；展开任务校验与错误详情。
- 检测不可用：准备权重和检测依赖；不是“没有目标”。
- 云端错误：查看HTTP状态、失败阶段及建议，密钥不会显示在错误内容中。
