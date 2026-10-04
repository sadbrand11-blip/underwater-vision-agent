"""Source-anchored question construction. Never imports or calls a retriever."""
import json
from pathlib import Path
import random

from optical_agent.rag_engine import DATA_ROOT, fingerprint

# Each tuple is a fact family, source, exact supporting quote, and two phrasings.
FACTS = [
('lost_texture','exposure','gamma和CLAHE可以调整已有像素的分布，但不能确定丢失的真实纹理。','全黑图被调亮后，为什么还不能说纹理恢复了？','Can gamma or CLAHE determine texture that was already lost?'),
('candidate_noise','exposure','候选框增加也可能是增强放大的噪声或误报。','增强后多了几个框，为什么还可能是坏结果？','候选目标变多可能来自什么干扰？'),
('no_gt_accuracy','exposure','没有真实标注时，只能比较检测证据，不能声称准确率提高。','没有真实框标注，能否说校正提高了识别准确率？','Why are before/after detections not an accuracy evaluation without ground truth?'),
('dark_background','exposure','当前水下质量配置允许部分暗背景，但阈值来自小样本启发式','水下背景较暗是否一定意味着整张图不能用？','水下质量规则对暗背景的容忍有什么依据限制？'),
('internal_reliability','exposure','reliable表示通过当前规则，不表示经过分布外可靠性验证。','reliable标记能保证新水域也可靠吗？','当前的可靠标记和 OOD 验证是一回事吗？'),
('multiframe_registration','exposure','现有多帧功能选择有效帧并检查框的一致性，没有实现图像配准或纹理重建。','现有多帧流程会自动配准和重建纹理吗？','多帧功能实际做了什么，还有什么没实现？'),
('human_queue','exposure','系统不设人工复核队列。','结果失败后会进入人工复核队列吗？','系统有没有转人工排队功能？'),
('legacy_architecture','model','模型：Faster R-CNN MobileNetV3 320 FPN；COCO初始化后，以160张SODD原图微调4轮。','旧默认检测器的架构与微调数据是什么？','Describe the legacy SODD detector architecture and training initialization.'),
('legacy_mapping','model','类别编号为背景0、propeller 1、pipe_type2 2、red_fin 3、net 4、qr_codes 5、pipe 6。','默认模型的类别编号怎样对应？','背景、螺旋桨和两种管道各是什么 class id？'),
('isotonic_probability','model','estimated_box_precision来自验证分布的保序回归（IsotonicRegression），不是任意水域下的正确概率。','estimated_box_precision是不是任意水域都准确的概率？','IsotonicRegression 输出为什么不能当作跨水域正确概率？'),
('legacy_threshold','model','阈值0.33在40张验证图上确定。','旧默认检测阈值是多少，在哪些数据上定的？','Where was the legacy 0.33 detection threshold selected?'),
('legacy_biological_classes','model','没有鱼、珊瑚或MOUD九类生物的训练头。','默认模型能直接识别鱼和珊瑚吗？','默认 SODD 模型是否具备 MOUD 九类生物检测头？'),
('empty_detection','model','未检出候选不能证明没有目标。','没有检测框是否就能证明图里没目标？','没有任何候选框时，结论的边界是什么？'),
('moud_illumination','moud','真实灯光变化不等于相机快门曝光变化。','MOUD 的明暗分档是不是快门曝光变化？','MOUD 的 light levels 能当成不同 shutter exposure 吗？'),
('moud_unaligned','moud','抽样帧没有逐帧配准参考','可以直接用不同照明的 MOUD 抽样帧算 PSNR 吗？','MOUD 本地抽样帧是否有逐帧对齐参考？'),
('moud_subset','moud','本项目已下载Scene 1的24张图，三档光照各8张。','电脑上用了多少 MOUD 图片，各档有多少？','本地 MOUD 使用的是哪一场景和多大子集？'),
('moud_scope','moud','这是当前实验观察，不是MOUD全数据集结论。','本地最低照明失败能代表整个 MOUD 吗？','为什么不能把24张图的结果推广到整个照明数据集？'),
('uieb_no_boxes','vision_data','本项目没有 UIEB 检测框。','UIEB 在本项目里能用于计算目标框召回吗？','我们是否有 UIEB 对应的 detection bounding boxes？'),
('robot_instance_count','vision_data','10,048 张图片，官方类别名 robots、类别编号 10，502 个机器人框。','下载包中机器人官方类别编号和实例数是多少？','UIIS10K robots 的类别编号以及实际框数是什么？'),
('robot_merge','vision_data','该分支只训练合并的 underwater_robot，包括水下机器人 AUV/ROV 等，不区分二者','实验机器人模型能分开判断 AUV 与 ROV 吗？','underwater_robot 标签包含什么，是否细分机器人类型？'),
('missing_robot_model','vision_data','机器人模型文件不存在应显示不可用，不能解释为没有目标。','机器人权重丢失时应该返回什么状态？','机器人模型 unavailable 能被当作没检测到目标吗？'),
('scene_independence','vision_data','缺少可核实序列编号，因此这种分组不能保证真正场景独立。','感知哈希分组能保证拍摄场景完全独立吗？','pHash 划分防泄漏后，还存在什么场景独立性限制？'),
('paired_resize','vision_data','PSNR/SSIM 使用一致缩放长边最多 768 像素，仅限 UIEB 对齐参考图。','本项目 PSNR 和 SSIM 的图像尺寸及参考条件是什么？','UIEB paired metrics 是原尺寸计算还是有缩放？'),
('robot_background','vision_data','其他官方类别作为本分支背景，不输出鱼、珊瑚等类别。','机器人分支会输出鱼、珊瑚等 UIIS 其他类别吗？','训练机器人分支时怎样处理其他已标注类别？'),
('insufficient_calibration','vision_data','校准缺失或样本不足的类别保留候选，即使框分数很高也不能自动变为可信结论。','没有足够校准数据，高分框能自动变成可信结果吗？','样本不足或 calibration missing 时，候选如何处理？'),
('v040_no_llm','vision_results_v040','无付费 LLM 请求。','0.4.0视觉训练实验有没有使用付费大模型调用？','0.4.0 是付费 LLM 调度评测还是本地视觉实验？'),
('d_drive_data','readme','全部数据集下载物位于 `D:\\CodexData\\optical_agent`，不提交到 Git。','数据集都保存在哪里，是否上传到了 Git？','Where are dataset downloads stored and are they committed?'),
('weights_autoload','readme','模型文件会自动加载，无需手动打开 `.pt` 文件。','我要先手动打开 pt 模型文件才能分析吗？','启动网页之后如何加载模型文件？'),
('no_camera_control','readme','没有相机控制或人工复核队列。','这个项目能控制相机重拍吗？','当前原型有没有接入相机控制功能？'),
('measured_by_program','readme','视觉测量和可靠性状态由程序产生。','曝光测量和可靠性状态是谁产生的？','是否由语言模型自行填写视觉测量和可靠性状态？'),
('sea_coefficients','paper_sea_thru','直接信号衰减与后向散射由不同系数控制，不能简单共用一个大气去雾系数。','Sea-Thru 为什么不直接套用大气去雾系数？','Do direct attenuation and backscatter share one coefficient in Sea-Thru?'),
('sea_rgbd','paper_sea_thru','Sea-Thru 使用 RGBD 图像及已知距离信息','Sea-Thru 的输入需要深度或距离吗？','只有 RGB 图片满足原始 Sea-Thru 的输入条件吗？'),
('machine_preference','paper_machine_iqa','人眼偏好与机器视觉任务偏好存在差异，观感更好不保证检测、分割或问答结果更好。','MPD 研究是否认为观感更好就保证机器任务结果更好？','Does better visual appeal guarantee better downstream detection according to machine IQA?'),
('machine_protocol','paper_machine_iqa','机器质量需要指定下游任务、测试模型及评价指标。','评价机器视觉图像质量需要先确定哪些条件？','机器质量评估协议应指定哪些项目？'),
('reed_alignment','paper_reed','采用裁剪和改进的 SIFT 对齐','REED 真实序列如何处理运动造成的错位？','极端曝光连拍图通过什么办法对齐？'),
('reed_brightness','paper_reed','论文指出某些情况下仅看亮度不足以判断曝光状态','REED 论文是否认为仅看亮度在所有情况下都足以判断曝光？','Does the REED paper regard luminance alone as sufficient in every case?'),
('rhc_modules','paper_rhcnet','提出残差引导特征增强 RGFE 和分层特征校准金字塔 HFCP。','RHCNet 提出的两个核心模块是什么？','What are the names and abbreviations of the two core RHCNet modules?'),
('rhc_calibration','paper_rhcnet','这里的 calibration 是特征对齐和校准，不是本项目把检测分数映射为经验精确率的 isotonic 概率校准','RHCNet 特征校准可以代替现有置信度校准吗？','RHCNet calibration 和 isotonic calibration 是同一个东西吗？'),
('mixed_domain','paper_enhancement_augmentation','混合域训练与推理前对每张图固定增强是不同的操作。','增强作为增广与推理时统一增强有什么区别？','How does mixed-domain training differ from always enhancing images at inference?'),
('uieb_challenge','paper_uieb','其余 60 张作为挑战集，没有令人满意的参考结果。','UIEB 的60张挑战图为什么没有参考增强图？','为什么 UIEB challenging set 不能照搬有参考图的评测？'),
]

UNKNOWN = [
('camera_iso','这张上传图的原始 ISO 数值是多少？'),
('camera_shutter','本地机器人样本每张照片的准确快门时间是多少毫秒？'),
('camera_aperture','MOUD 当前样本采集时每张图片的镜头光圈值是多少？'),
('robot_mass','UIIS10K test_0352 机器人整机质量是多少千克？'),
('robot_manufacturer','UIIS10K 所有机器人实例分别由哪家厂商制造？'),
('robot_battery','这张图中的 ROV 电池还剩多少分钟续航？'),
('field_temperature','当前上传图片拍摄时水温是多少摄氏度？'),
('field_salinity','Scene 1 每帧采集时水体盐度的精确数值是多少？'),
('field_depth','上传图目标距相机的实测距离是多少米？'),
('hardware_response','本机镜头与传感器联合光谱响应的标定曲线数值是什么？'),
('cake','黑森林蛋糕需要哪些食材和烘焙步骤？'),
('taxi','明天上海机场到火车站出租车多少钱？'),
('chess','国际象棋王车易位有哪些规则？'),
('gardening','如何给阳台的月季施肥？'),
('music','小提琴 D 大调音阶的指法是什么？'),
('history','唐代科举殿试的演变过程是什么？'),
('accounting','增值税一般纳税人的账务分录怎样做？'),
('football','今年足球世界杯决赛的比分是多少？'),
('quantum','超导量子芯片的约瑟夫森结临界电流是多少？'),
('astronomy','木星大红斑最近的风速测量结果是多少？'),
]

# Curated equivalent statements, before seeing rankings; not string-similarity labels.
ALTERNATIVES = {
 'human_queue': [('readme','没有相机控制或人工复核队列。')],
 'internal_reliability': [('readme','`reliable` 只表示通过本原型的内部规则，不代表对任意水域或目标都有可靠识别能力。')],
 'moud_illumination': [('results','照明来自灯光档位，不是相机快门曝光'),('readme','分档来自水下灯光强度，不是相机曝光时间')],
 'moud_unaligned': [('results','MOUD 没有对齐的高质量参考图'),('readme','MOUD 没有对齐参考图，因此没有在它上面计算 PSNR/SSIM。'),('vision_results_v040','MOUD照明强度变化不是已知相机曝光时间序列；无配准参考、不计算PSNR。')],
 'lost_texture': [('vision_data','全黑、全白、关键区域信息丢失不能靠增强恢复可靠事实。')],
 'candidate_noise': [('results','校正候选虽增加了 3.4 个百分点的召回，但也比原图多产生 12 个误检')],
 'legacy_biological_classes': [('readme','当前模型不负责识别鱼、珊瑚等其他生物。')],
 'legacy_threshold': [('results','最终模型在 40 张验证图上选出的工作阈值为 **0.33**')],
 'moud_subset': [('results','Scene 1 高、中、低照明各 8 张，覆盖四个区域')],
 'moud_scope': [('readme','现有 240 张 SODD 图和 24 张 MOUD 图只够验证流程与暴露失败模式。')],
 'v040_no_llm': [('readme','已增加 UIEB 官方下载与配对评测、跨数据集分组划分、六类设施对照训练和独立水下机器人分支。数据、图像缓存和新训练权重统一保存在 `D:\\CodexData\\optical_agent`；本轮不调用付费 LLM。')],
 'no_gt_accuracy': [('paper_machine_iqa','质量分数、检测框数量及分数上升均不能代替真实框标注下的召回与误报评测。')],
 'legacy_architecture': [('readme','**目标识别：**TorchVision Faster R-CNN MobileNetV3 320，从 COCO 权重微调，用 SODD 原始图和训练集内的像素域曝光扰动训练。')],
 'd_drive_data': [('results','数据集文件及下载清单保存在 `D:\\CodexData\\optical_agent`。Git 仓库只包含代码、模型与报告，不再分发数据集图片。')],
 'scene_independence': [('vision_results_v040','无可靠序列编号，不能保证真正场景独立。')],
 'paired_resize': [('vision_results_v040','RGB PSNR 和 SSIM（skimage、7×7、data_range=255）；原图/参考图相同缩放，长边最多768px。参考图是作者的增强参考，不是相机曝光真值')],
}


def build(repo=Path(__file__).parent):
    sources = json.loads((repo / 'knowledge/sources_v2.json').read_text(encoding='utf8'))
    by_id = {s['id']: s for s in sources}
    rng = random.Random(20261003)
    bundles = [[2, 9], [16, 21]] + [[i] for i in range(30) if i not in {2, 9, 16, 21}]
    rng.shuffle(bundles)
    old_dev = []
    for bundle in bundles:
        if len(old_dev) + len(bundle) <= 15:
            old_dev.extend(bundle)
    new = list(range(30, 40)); rng.shuffle(new)
    dev = set(old_dev + new[:5])
    bundle_names = {2:'no_gt_and_calibration',9:'no_gt_and_calibration',
                    16:'small_subset_scene_independence',21:'small_subset_scene_independence'}
    evidence, cases = {}, []
    for index, (family, sid, quote, first, second) in enumerate(FACTS):
        source_root = DATA_ROOT / ('c0_source' if index < 30 else 'c1_source')
        text = (source_root / by_id[sid]['path']).read_text(encoding='utf8')
        assert quote in text, (family, quote)
        start = text.index(quote)
        evidence[family] = {'alternatives': [{'source_id': sid, 'path': by_id[sid]['path'],
            'quote': quote, 'start': start, 'end': start + len(quote)}]}
        for variant, query in enumerate([first, second], 1):
            cases.append({'id': f'q{index+1:02d}_{variant}', 'fact_group': bundle_names.get(index,family),
                'split': 'dev' if index in dev else 'test', 'kind': 'existing' if index < 30 else 'literature',
                'query': query, 'required_groups': [family], 'vision_mode':
                    'experiment_candidate' if sid=='vision_results_v040' or family.startswith('robot_') or family=='missing_robot_model' else 'legacy'})
    quote='论文分别测试四种增强模型，每次将一种增强器的变体与原图配对用于混合域训练，保持检测器架构与推理流程不变'
    sid='paper_enhancement_augmentation'
    text=(DATA_ROOT/'c1_source'/by_id[sid]['path']).read_text(encoding='utf8')
    start=text.index(quote)
    evidence['mixed_training_design']={'alternatives':[{'source_id':sid,'path':by_id[sid]['path'],
        'quote':quote,'start':start,'end':start+len(quote)}]}
    for family, alternatives in ALTERNATIVES.items():
        for sid, quote in alternatives:
            text=(DATA_ROOT/'c1_source'/by_id[sid]['path']).read_text(encoding='utf8')
            start=text.index(quote)
            evidence[family]['alternatives'].append({'source_id':sid,'path':by_id[sid]['path'],
                'quote':quote,'start':start,'end':start+len(quote)})
    # Cross-source questions bind all involved families to the same split.
    for case in cases:
        if case['id']=='q03_2':
            case['query']='没有真实标注时，为什么不能据 estimated_box_precision 上升宣称准确率改善，它的概率解释又有什么限制？'
            case['required_groups']=['no_gt_accuracy','isotonic_probability']
        if case['id']=='q22_2':
            case['query']='为什么不能仅用 pHash 和24张 MOUD 抽样图宣称场景泛化已经验证？'
            case['required_groups']=['scene_independence','moud_scope']
        if case['fact_group']=='mixed_domain':
            case['required_groups']=['mixed_domain','mixed_training_design']
    domain = list(range(10)); outside = list(range(10,20)); rng.shuffle(domain); rng.shuffle(outside)
    negative_dev = set(domain[:5] + outside[:5])
    for index,(family,query) in enumerate(UNKNOWN):
        cases.append({'id':f'n{index+1:02d}','fact_group':family,'split':'dev' if index in negative_dev else 'test',
            'kind':'unanswerable','negative_type':'in_domain_unknown' if index<10 else 'out_of_domain',
            'query':query,'required_groups':[], 'vision_mode':'experiment_candidate' if 'robot' in family else 'legacy'})
    result={'version':'1','seed':20261003,'labels':'source_anchored_automatic_requires_review',
        'metric_unit':'canonical evidence group, alternatives deduplicated',
        'evidence':evidence,'questions':cases}
    result['construction_sha256']=fingerprint({'facts':FACTS,'unknown':UNKNOWN})
    return result


if __name__=='__main__':
    root=Path(__file__).parent
    output=root/'eval/rag_questions.json'
    output.write_text(json.dumps(build(root),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Constructed 100 source-anchored questions, no retrieval called')
