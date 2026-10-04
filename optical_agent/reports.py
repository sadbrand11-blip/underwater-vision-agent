"""Render facts from observations; never trust a model-supplied score or status."""

import json

STATUS = {'reliable': '通过当前内部可靠性规则', 'unreliable': '识别结论不可靠',
          'quality_failure': '图像质量失败', 'quality_unassessable': '无法评估原始曝光'}
EXPOSURE = {'normal': '正常', 'underexposed': '偏暗', 'overexposed': '偏亮', 'mixed': '明暗混合',
            'preprocessed_binary': '接近二值化'}
IMAGE_NAMES = {'original': '原图', 'corrected': '校正图'}


def quality_text(image_id, quality):
    metrics = quality['global']
    return (f'{IMAGE_NAMES.get(image_id, image_id)}曝光提示：{EXPOSURE.get(quality["exposure_state"], quality["exposure_state"])}；'
            f'暗像素 {metrics["dark_fraction"]:.1%}，高亮像素 {metrics["bright_fraction"]:.1%}，'
            f'对比度 {metrics["contrast"]:.4f}，边缘强度 {metrics["edge_strength"]:.4f}，'
            f'清晰度指标 {metrics["sharpness"]:.3g}，噪声估计 {metrics["noise_estimate"]:.4f}。'
            f'质量规则：{"通过" if quality["quality_pass"] else "未通过"}。')


def evidence_map(context):
    facts = {}
    for image_id, quality in context.qualities.items():
        facts[f'quality:{image_id}'] = {'type': 'quality', 'image_id': image_id, 'data': quality,
            'text': quality_text(image_id, quality)}
    for image_id, detections in context.detections.items():
        items = '；'.join(f'{d["class_name"]}（模型分数 {d["score"]:.4f}）' for d in detections)
        facts[f'detections:{image_id}'] = {'type': 'detections', 'image_id': image_id, 'data': detections,
            'text': f'{IMAGE_NAMES.get(image_id, image_id)}检测到 {len(detections)} 个候选框。' + (items or '未检出候选，不能据此证明图中没有目标。')
                    + ' 这些是检测候选；未经完整规则检查不能自动接受。'}
    for image_id, parameters in context.corrections.items():
        applied = parameters.get('applied', False)
        facts[f'correction:{image_id}'] = {'type': 'correction', 'image_id': image_id,
            'data': {'source_image_id': image_id, 'output_image_id': 'corrected', 'parameters': parameters},
            'text': ('已生成一轮校正候选。' if applied else '本次未调整，候选图与原图一致；无需校正或输入不适合曝光估计。')
                    + ' 处理参数：' + json.dumps(parameters, ensure_ascii=False)
                    + ' 校正完成不代表目标识别可靠。'}
    for image_id, report in context.reports.items():
        facts[f'comparison:{image_id}'] = {'type': 'comparison', 'image_id': image_id, 'data': report,
            'text': f'原图候选 {len(report["detections_original"])} 个，校正图候选 {len(report["detections_corrected"])} 个。'
                    f'采用 {IMAGE_NAMES.get(report["selected_image"], report["selected_image"])}；{STATUS[report["status"]]}。'
                    + (' 原因：' + '；'.join(report['reasons']) if report['reasons'] else '')
                    + ' 候选数量增加或分数提高本身不等于识别准确率提高。\n'
                    + quality_text('original', report['quality_before']) + '\n'
                    + quality_text('corrected', report['quality_after'])}
    return facts


def build_report(context, selection, execution_status, trace, usage, execution_mode, model_name, error=None):
    facts = evidence_map(context)
    if not isinstance(selection, dict) or not {'evidence_ids', 'citation_ids'} <= selection.keys():
        raise ValueError('报告必须明确提供证据与引用列表')
    selected_ids = selection.get('evidence_ids', [])
    citation_ids = selection.get('citation_ids', [])
    if not isinstance(selected_ids, list) or not isinstance(citation_ids, list):
        raise ValueError('报告中的证据编号必须是列表')
    if any(not isinstance(x, str) or x not in facts for x in selected_ids):
        raise ValueError('模型引用了未执行的分析证据')
    if any(not isinstance(x, str) or x not in context.citations for x in citation_ids):
        raise ValueError('模型引用了未检索的知识来源')
    if execution_status == 'completed' and not selected_ids and not citation_ids:
        raise ValueError('模型没有返回任何实际依据，不能标记为有效结束')
    evidence = [dict(evidence_id=x, **facts[x]) for x in dict.fromkeys(selected_ids)]
    citations = [context.citations[x] for x in dict.fromkeys(citation_ids)]
    compared = next((x['data'] for x in evidence if x['type'] == 'comparison'), None)
    texts = [x['text'] for x in evidence]
    if citations:
        texts.append('知识依据（检索片段，非本图测量）：')
        texts.extend(f'[{c["citation_id"]}] {c["title"]}：{c["text"]}' for c in citations)
    elif any(e.get('tool') == 'retrieve_knowledge' and e.get('ok') for e in trace):
        texts.append('本次没有选用可支持结论的知识引用。')
    if error:
        texts.insert(0, error)
    if not texts:
        texts.append('暂无可展示的分析证据，请先调用相应工具。')
    image_ids = ['original']
    for item in evidence:
        image_id = item['image_id']
        image_ids.append(image_id)
        if item['type'] == 'quality':
            image_ids.append(f'{image_id}_quality_heatmap')
        if item['type'] == 'detections':
            image_ids.append(f'{image_id}_detections')
        if item['type'] == 'comparison':
            image_ids.extend(['corrected', 'report_detections_selected'])
        if item['type'] == 'correction':
            image_ids.append('corrected')
    return {'answer': '\n\n'.join(texts), 'execution_status': execution_status,
            'execution_mode': execution_mode, 'model': model_name,
            'vision_status': compared['status'] if compared else None,
            'vision_result': compared, 'evidence': evidence, 'citations': citations,
            'image_refs': [x for x in dict.fromkeys(image_ids) if x in context.images],
            'trace': trace, 'usage': usage, 'error': error}
