"""Publish only actual experiment aggregates. Images, weights and raw results stay on D:."""
from optical_agent.paths import PROJECT_ROOT
import csv
import json
from pathlib import Path
from optical_agent.vision_data import ROOT, digest, save_json
from metrics import match_counts

BASE=PROJECT_ROOT

def pct(value):return '未定义' if value is None else f'{value*100:.2f}%'
def number(value):return '未定义' if value is None else f'{value:.4f}'

def write_case_table(folder, detectors):
    """One row per actual heldout image; abstention is distinct from detection error."""
    fields=['model','image_id','group_id','gt_instances','original_tp','original_fp','original_fn',
            'selected_tp','selected_fp','selected_fn','quality_failed','unreliable','new_correction_fp','reasons']
    rows=[]
    for model, result in detectors['models'].items():
        for path in sorted((folder/'detection_test'/model).glob('*.json')):
            item=json.loads(path.read_text(encoding='utf8')); report=item['report']
            threshold=result['original']['threshold']
            original=match_counts([item['original']],[item['truth']],threshold)
            selected=match_counts([report['detections_selected']],[item['truth']],threshold)
            row={'model':model,'image_id':item['id'],'group_id':item['group_id'],
                 'gt_instances':len(item['truth']),'quality_failed':not report['quality_before']['quality_pass'],
                 'unreliable':report['status']!='reliable','new_correction_fp':item['new_correction_false_positives'],
                 'reasons':'；'.join(report['reasons'])}
            for prefix, counts in [('original',original),('selected',selected)]:
                row.update({prefix+'_'+key:counts[key] for key in ['tp','fp','fn']})
            rows.append(row)
    with (BASE/'docs/reports/VISION_CASES.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader();writer.writerows(rows)
    return {'images_including_model_repeats':len(rows),
            'detection_error_rows':sum(bool(r['selected_fp'] or r['selected_fn']) for r in rows),
            'unreliable_rows':sum(r['unreliable'] for r in rows)}

def main():
    folder=ROOT/'vision_v040'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
    exposure=json.loads((folder/'exposure/test_summary.json').read_text(encoding='utf8'))
    selection=json.loads((folder/'exposure/selection.json').read_text(encoding='utf8'))
    detectors=json.loads((folder/'detection_test/summary.json').read_text(encoding='utf8'))
    if detectors['status']!='complete':raise ValueError('Actual complete results are required')
    name=selection['selected']['name']; old=exposure['paired']['legacy_v030']; new=exposure['paired'][name]
    exposure_test={'ssim':new['ssim']>old['ssim'],'psnr':new['psnr']>=old['psnr']-.2}
    promoted=selection['promote_exposure'] and all(exposure_test.values())
    summary={'version':'0.4.0','evaluation_status':'complete','scope':detectors['run_scope'],
        'manifest_sha256':digest(folder/'manifest.json'),'freeze_sha256':digest(folder/'freeze.json'),
        'data':manifest['summary'],'grouping':manifest['grouping'],'quarantined':manifest['quarantined'],
        'exposure':{'profile':selection['selected'],'validation':selection['validation'],
            'validation_gates':selection['gates'],'test':exposure['paired'],'test_gates':exposure_test,'adoption_gate_passed':promoted},
        'facilities':{'validation_gates':detectors['facility_validation_gates'],'test_gates':detectors['facility_test_gates'],
            'adoption_gate_passed':detectors['promote_facilities']},
        'models':detectors['models'],'training':detectors['training'],
        'stress':{'challenge':exposure['challenge'],'moud':exposure['moud']},
        'default':'historical six-class default retained; new models/profile explicitly selectable as experiments',
        'ap_scope':'Original AP50 uses internal score floor0.05 candidates; corrected/selected/accepted AP50 are operating-threshold-truncated diagnostics, not equal-floor AP comparisons.',
        'paid_llm_requests':0,'cost_verification_scope':'Visual scripts do not call a paid provider; offline web records have zero HTTP model attempts; no independent billing ledger.',
        'formal_agent_eval':'unchanged, not repeated in this visual experiment',
        'runtime_revision':'v040_delivery_r1; frozen single-frame implementation unchanged',
        'protocol_limits':{'calibration':'One pooled isotonic curve per branch, fit on original validation images; no per-class or corrected-image calibration validation.',
            'freeze':'MOUD 24-image diagnostics, some imported dependencies and provenance records were not included in the pre-test freeze; subsequent supplementary hashes are post-evaluation.',
            'delivery':'Adaptive offline demo chooses gamma-only by rule; fixed-pipeline reported correction uses full local profile. They are distinct execution paths.',
            'detector_latency':'Excludes model load, input decode, overlays, web rendering and LLM; timer stops before overlays.',
            'exposure_latency':'Correction plus paired metric calculation where applicable; excludes before/after quality assessments and input decoding.'},
        'training_recovery':{'robot':'two runs aborted in epoch4; final resume of validation-selected weights, at most4epochs; AdamW reset; RPN threshold0 only in training, .05 validation/inference; same seed, at most12attemptedepochs total',
                            'skipped_batches':detectors['training']['robot_improved'].get('numerical_skips',[])},
        'independence':'pixel/perceptual groups; true scene independence not verified; one seed only'}
    summary['case_table']=write_case_table(folder,detectors)
    save_json(BASE/'docs/reports/VISION_RESULTS.json',summary)
    lines=['# 0.4.0 视觉优化实验报告','', '日期：2026-10-01。真实本机 RTX 3060 Laptop GPU 训练；无付费 LLM 请求。',
        '', '## 结论与采用状态','',
        f'- 曝光验证门槛通过：{selection["promote_exposure"]}；保留测试门槛通过：{all(exposure_test.values())}。候选配置：`{name}`。',
        f'- 六类检测验证门槛：{detectors["facility_validation_gates"]}；测试门槛：{detectors["facility_test_gates"]}。',
        '- 网页提供旧配置、实验重训对照、实验改进候选。历史默认暂保留；单种子、小验证集结果不能构成可靠部署保证。',
        '- 这是视觉实验，不是重新完成正式 Agent Eval；0.2.4、0.3.0 报告未改动。交付入口修订为 `v040_delivery_r1`，冻结的单帧算法和结果保持原样。',
        '', '## 数据、划分与检查','',
        '| 数据 | 训练图 | 选模验证图 | 校准图 | 测试图 |', '|---|---:|---:|---:|---:|']
    for ds,splits in manifest['summary'].items():
        lines.append('|'+ds+'|'+'|'.join(str(splits[s]['images']) for s in ['train','selection','calibration','test'])+'|')
    lines += ['',f'全局分组 {manifest["grouping"]["groups"]} 组，跨数据集重叠 {manifest["grouping"]["cross_dataset_groups"]} 组，最大组 {manifest["grouping"]["largest_group"]} 图。采用像素哈希和 pHash 距离≤6；无可靠序列编号，不能保证真正场景独立。',
        '', 'UIEB 890 对参考图配对/尺寸/解码通过；60 张挑战图只作测试。UIIS10K 实际 10,048 图、502 个机器人实例；隔离一张含两个非机器人空框的校准负图后使用 10,047 图，机器人实例未变。SODD 240 图。原始文件保留。',
        '', '训练与选模数据在隔离前后完全相同，校准在测试前重算；初始增强策略不匹配的试跑已中止并保留，不参与成绩。旧划分没有替换；当前为项目实验划分。',
        '', '机器人两次训练均在第4轮因非有限损失中止，均保留日志和当时最佳权重。纯负批次的RoI损失非有限、RPN损失有限；控制实验复现空候选导致的损失异常，但没有原失败即时权重，不称逐位重放。最终恢复从验证选中的权重继续最多4轮，AdamW重置；仅训练时RPN候选阈值为0，验证/推理保持0.05。合计最多12轮尝试；同种子恢复不等于独立重复实验。异常最多允许两批记录后跳过。',
        '',f'最终清单 SHA256：`{summary["manifest_sha256"]}`；冻结文件 SHA256：`{summary["freeze_sha256"]}`。',
        '', '## UIEB 校正结果','',
        'RGB PSNR 和 SSIM（skimage、7×7、data_range=255）；原图/参考图相同缩放，长边最多768px。参考图是作者的增强参考，不是相机曝光真值；数值不是原分辨率官方榜单成绩。',
        '', '| 集合／方案 | 图像数 | PSNR dB | SSIM |', '|---|---:|---:|---:|']
    for split,scores in [('选模验证',{'original':selection['original'],**selection['validation']}),('保留测试',exposure['paired'])]:
        for method,score in scores.items():
            lines.append(f'|{split}／{method}|{score["images"]}|{number(score.get("psnr"))}|{number(score.get("ssim"))}|')
    lines += ['',f'候选参数：目标中位数 {selection["selected"]["target_median"]}，Gamma 范围 {selection["selected"]["gamma_min"]}～{selection["selected"]["gamma_max"]}，CLAHE混合 {selection["selected"]["clahe_blend"]}。训练638图搜索24组，按平均SSIM选前三组，验证62图选最终配置，校准62图未用于选参。',
        '', '## 检测结果（原图）','',
        '新划分上两个设施模型从同一 COCO 初始化训练。固定种子20261001、AdamW、lr0.0002、batch2、最多12轮、连续3轮主指标未升则停止。对照采用原有标量曝光扰动；改进候选加入有界Gamma、水平光照梯度和轻噪声。机器人单独训练，保留全部训练正图、固定哈希选负图，上限3倍/1000。',
        '', '阈值只在选模验证集、每图误报≤1约束下按宏召回选取；校准使用独立验证组。保留测试没有阈值扫描。IoU0.5、一对一框匹配，101点AP50；不是官方实例分割评测。',
        '', '| 模型 | 最佳轮 | 测试图／实例 | 宏召回 | 总召回 | mAP50 | 误报/图 |', '|---|---:|---:|---:|---:|---:|---:|']
    for model,result in detectors['models'].items():
        history=json.loads((folder/'models'/(model+'.history.json')).read_text(encoding='utf8')); best=max(history['history'],key=lambda h:h['primary_metric'])
        score=result['original']
        best_label=str(best['epoch']) if best['epoch'] else '恢复起点（重试第'+str(best['source_epoch'])+'轮）'
        lines.append(f'|{model}|{best_label}|{score["images"]}／{result["instances"]}|{pct(score["macro_recall"])}|{pct(score["recall"])}|{pct(score["map50"])}|{number(score["fp_per_image"])}|')
    lines += ['', '| 类别 | 对照召回 | 改进召回 | 对照AP50 | 改进AP50 | 测试实例 |', '|---|---:|---:|---:|---:|---:|']
    a=detectors['models']['facilities_control']['original']; b=detectors['models']['facilities_improved']['original']
    for c,r in a['by_class'].items():
        other=b['by_class'][c]
        lines.append(f'|{c}|{pct(r["recall"])}|{pct(other["recall"])}|{pct(r["ap50"])}|{pct(other["ap50"])}|{r["instances"]}|')
    robot=detectors['models']['robot_improved']
    lines += ['',f'机器人测试 {robot["instances"]} 实例、{robot["positive_groups"]} 个阳性组、{robot["positive_images"]} 张阳性图；样本量不足标记：{robot["exploratory_only"]}。该标记仅检查30实例/5组门槛，能力仍为实验状态。原图候选精确率仅{pct(robot["original"]["precision"])}，自动接受结果召回{pct(robot["accepted"]["recall"])}、精确率{pct(robot["accepted"]["precision"])}。只评估合并robot类别，不证明AUV/ROV型号分类或可靠航行器识别。',
        '', '## 端到端及模拟曝光','', '| 模型 | 原图召回 | 校正图召回 | 最终图召回 | 自动接受召回 | 不可靠比例 | 新增误报数 | 流程P95 ms |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,r in detectors['models'].items():
        lines.append(f'|{n}|{pct(r["original"]["recall"])}|{pct(r["corrected"]["recall"])}|{pct(r["selected"]["recall"])}|{pct(r["accepted"]["recall"])}|{pct(r["unreliable_ratio"])}|{r["new_correction_false_positives"]}|{r["latency_p95_ms"]:.1f}|')
    lines += ['', '自动接受召回将不可靠图计为未自动接受；不可靠不等于没有目标。新增误报先与GT一对一匹配，再与原图误报一对一匹配；候选数量上升不解释为准确率提升。流程耗时不含模型加载、解码、框图绘制、网页渲染或LLM，训练/评测并行时会受资源竞争影响。',
        '', '本表是固定完整视觉流程的校正结果，使用局部校正配置。网页动态离线演示默认尝试Gamma候选，不含CLAHE，与本表不是同一路径；网页实际方法以工具记录为准，不能把本表成绩直接移到其结果上。',
        '', '原图AP50使用内部0.05分数下限后的候选。机器摘要中校正/最终/自动接受的AP字段使用运行阈值截断后的结果，仅作诊断，不与原图AP作相同候选下限的比较；端到端比较使用同一运行阈值下的召回、误报。',
        '', '模拟亮度×0.6结果：']
    for n,r in detectors['models'].items():
        if 'simulated_under' in r:
            lines.append(f'- {n}：召回{pct(r["simulated_under"]["recall"])}，宏召回{pct(r["simulated_under"]["macro_recall"])}，误报/图{number(r["simulated_under"]["fp_per_image"])}。')
    lines += ['', '上述是同一保留图的模拟扰动，不是实际相机重拍收益。',
        '', '## 无参考压力诊断','', '| 数据／方案 | 图数 | 质量失败 | 平均暗部占比 | 平均高亮饱和 | 噪声代理 | 校正及指标P95 ms |', '|---|---:|---:|---:|---:|---:|---:|']
    for ds,scores in [('挑战集',exposure['challenge']),('MOUD子集',exposure['moud'])]:
        for method,score in scores.items():
            lines.append(f'|{ds}／{method}|{score["images"]}|{score["quality_failures"]}|{number(score["dark_fraction"])}|{number(score["bright_fraction"])}|{number(score["noise_estimate"])}|{score.get("elapsed_ms_p95",0):.1f}|')
    lines += ['', 'MOUD照明强度变化不是已知相机曝光时间序列；无配准参考、不计算PSNR。挑战集/MOUD没有本项目完整目标框、不报告检测召回。MOUD的24张诊断图没有进入测试前freeze.json，后补哈希不能称为预冻结。耗时不含输入解码及前后质量评估；原图行的0表示未执行校正。',
        '', '## 校准、限制与后续','']
    for n,r in detectors['models'].items():
        c=r['calibration']
        lines.append(f'- {n}：{c["status"]}，{c["groups"]}个校准组，类别状态{c["class_status"]}。')
    lines += ['- 样本不足的类别拒绝将原始分数作为可靠概率；每分支使用跨类合并的小样本isotonic，只在原图校准，未建立逐类或校正图校准保证。',
        '- 未验证真实场景独立性，只做像素/感知近重复隔离；单种子不能证明稳定泛化。旧历史模型不在新划分上充当公平训练对照。',
        '- 两个独立分支避免缺失标注当背景；动态调度只检测管道不调用机器人模型。实验模型禁止与旧调度组合；旧调度对照使用旧默认模型。模型缺失直接显示不可用，框、计数和报告使用同一类别过滤。',
        '- UIEB学术、非商业、禁止重分发。公开代码/摘要不包含图片、缓存和新权重。',
        '', '## 本地证据与复现','',
        '完整数据清单、逐图变化、训练历史、原始/校正/最终采用检测框在 `D:\\CodexData\\optical_agent\\vision_v040`；机器可读公開摘要为 [VISION_RESULTS.json](VISION_RESULTS.json)。逐图失败定位表 [VISION_CASES.csv](VISION_CASES.csv) 包含所有测试图的误报、漏报、质量失败与不可靠状态；不可靠的无机器人图不自动算作检测误报。操作见 [10_VISION_OPTIMIZATION.md](docs/10_VISION_OPTIMIZATION.md)。审计见 `VISION_EXPERIMENT_AUDIT.md`，同模型家族审核为暂定意见。',
        '', '[UIEB作者说明](https://li-chongyi.github.io/proj_benchmark.html)；[UIIS10K作者说明](https://github.com/LiamLian0727/UIIS10K)。']
    (BASE/'docs/reports/VISION_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'exposure_gate_passed':promoted,'facilities_gate_passed':detectors['promote_facilities'],
        'robot_recall':robot['original']['recall'],'report':'docs/reports/VISION_RESULTS.md'}),flush=True)

if __name__=='__main__':main()
