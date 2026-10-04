"""Cheap resource probes for presentation; no imports of models or cloud requests."""
from importlib.util import find_spec
from pathlib import Path

def probe_capabilities(model_path, embedding_root, *, experimental=True):
    dependencies = all(find_spec(name) is not None for name in ('torch', 'torchvision'))
    model = Path(model_path)
    detector_ready = dependencies and model.is_file() and model.stat().st_size > 1_000_000
    embedding = Path(embedding_root)
    dense_ready = (find_spec('torch') is not None and find_spec('transformers') is not None
        and (embedding/'model.safetensors').is_file() and (embedding/'config.json').is_file())
    modes = {}
    for mode, filename in [('experiment_control','facilities_control.pt'), ('experiment_candidate','facilities_improved.pt')]:
        folder = Path(r'D:\CodexData\optical_agent\vision_v040\models')
        facilities, robot = (folder/filename).is_file(), (folder/'robot_improved.pt').is_file()
        ready = experimental and dependencies and (facilities or robot)
        reason = ('实验资源已准备；推理时仍检查模型和校准。' if ready and facilities and robot else
            '部分实验分支缺少权重；仅可使用已准备的分支。' if ready else
            '实验权重未准备，不能将不可用解释为没有目标。' if experimental else
            '公开版本不提供机器人实验权重；使用本地开发版进行实验。')
        modes[mode] = {'available':ready, 'reason':reason}
    return {'detector':{'available':detector_ready,'reason':'检测资源已准备；实际加载仍验证模型。' if detector_ready else
        '检测资源未准备；可先评估与校正曝光，目标检测会明确报告不可用。'},
        'embedding':{'available':dense_ready,'reason':'本地向量模型已准备。' if dense_ready else
        'Embedding 依赖或模型未准备；TF-IDF 仍可使用。'}, 'vision_modes':modes}
