"""Stable COCO ontology, independent of Torch and model loading.

Category IDs below deliberately retain the original COCO 91-slot numbering.
The compact ``classes`` tuple is for capability discovery, never label lookup.
"""
from __future__ import annotations

import hashlib
import json

PROFILE_ID = 'general_rgb_v1'
ONTOLOGY_VERSION = 'coco2017_80_official_ids_zh_v1'

_ROWS = (
    (1, 'person', '人'), (2, 'bicycle', '自行车'), (3, 'car', '汽车'),
    (4, 'motorcycle', '摩托车'), (5, 'airplane', '飞机'), (6, 'bus', '公交车'),
    (7, 'train', '火车'), (8, 'truck', '卡车'), (9, 'boat', '船'),
    (10, 'traffic light', '交通信号灯'), (11, 'fire hydrant', '消防栓'),
    (13, 'stop sign', '停车标志'), (14, 'parking meter', '停车计时器'),
    (15, 'bench', '长椅'), (16, 'bird', '鸟'), (17, 'cat', '猫'),
    (18, 'dog', '狗'), (19, 'horse', '马'), (20, 'sheep', '羊'),
    (21, 'cow', '牛'), (22, 'elephant', '大象'), (23, 'bear', '熊'),
    (24, 'zebra', '斑马'), (25, 'giraffe', '长颈鹿'), (27, 'backpack', '背包'),
    (28, 'umbrella', '雨伞'), (31, 'handbag', '手提包'), (32, 'tie', '领带'),
    (33, 'suitcase', '行李箱'), (34, 'frisbee', '飞盘'), (35, 'skis', '滑雪板'),
    (36, 'snowboard', '单板滑雪板'), (37, 'sports ball', '运动球'),
    (38, 'kite', '风筝'), (39, 'baseball bat', '棒球棒'),
    (40, 'baseball glove', '棒球手套'), (41, 'skateboard', '滑板'),
    (42, 'surfboard', '冲浪板'), (43, 'tennis racket', '网球拍'),
    (44, 'bottle', '瓶子'), (46, 'wine glass', '高脚杯'), (47, 'cup', '杯子'),
    (48, 'fork', '叉子'), (49, 'knife', '刀'), (50, 'spoon', '勺子'),
    (51, 'bowl', '碗'), (52, 'banana', '香蕉'), (53, 'apple', '苹果'),
    (54, 'sandwich', '三明治'), (55, 'orange', '橙子'), (56, 'broccoli', '西兰花'),
    (57, 'carrot', '胡萝卜'), (58, 'hot dog', '热狗'), (59, 'pizza', '披萨'),
    (60, 'donut', '甜甜圈'), (61, 'cake', '蛋糕'), (62, 'chair', '椅子'),
    (63, 'couch', '沙发'), (64, 'potted plant', '盆栽'), (65, 'bed', '床'),
    (67, 'dining table', '餐桌'), (70, 'toilet', '马桶'), (72, 'tv', '电视'),
    (73, 'laptop', '笔记本电脑'), (74, 'mouse', '鼠标'), (75, 'remote', '遥控器'),
    (76, 'keyboard', '键盘'), (77, 'cell phone', '手机'), (78, 'microwave', '微波炉'),
    (79, 'oven', '烤箱'), (80, 'toaster', '烤面包机'), (81, 'sink', '水槽'),
    (82, 'refrigerator', '冰箱'), (84, 'book', '书'), (85, 'clock', '时钟'),
    (86, 'vase', '花瓶'), (87, 'scissors', '剪刀'), (88, 'teddy bear', '泰迪熊'),
    (89, 'hair drier', '吹风机'), (90, 'toothbrush', '牙刷'),
)
COCO_ID_TO_NAME = {category_id: name for category_id, name, _ in _ROWS}
COCO_NAME_TO_ID = {name: category_id for category_id, name, _ in _ROWS}
COCO_CLASSES = tuple(name for _, name, _ in _ROWS)
CLASS_NAMES = {name: zh for _, name, zh in _ROWS}
GENERAL_ALIASES = {zh: (name,) for _, name, zh in _ROWS}
GENERAL_ALIASES.update({
    '人物': ('person',), '行人': ('person',), '人员': ('person',), '人类': ('person',),
    '轿车': ('car',), '小汽车': ('car',), '单车': ('bicycle',), '巴士': ('bus',),
    '客车': ('bus',), '货车': ('truck',), '交通灯': ('traffic light',),
    '船只': ('boat',), '小船': ('boat',), '狗狗': ('dog',), '猫咪': ('cat',),
    '水杯': ('cup',), '茶杯': ('cup',), '酒杯': ('wine glass',),
    '可乐瓶': ('bottle',), '水瓶': ('bottle',), '植物': ('potted plant',),
    '电视机': ('tv',), '计算机': ('laptop',), '笔记本': ('laptop',),
    '电话': ('cell phone',), '智能手机': ('cell phone',),
    'sports_ball': ('sports ball',), 'cell_phone': ('cell phone',),
    'potted_plant': ('potted plant',), 'traffic_light': ('traffic light',),
    'sofa': ('couch',), 'aeroplane': ('airplane',), 'motorbike': ('motorcycle',),
})
ONTOLOGY_SHA256 = hashlib.sha256(json.dumps(_ROWS, ensure_ascii=False).encode()).hexdigest()


def supported_classes(detector):
    """Capability list without background or official vacant category slots."""
    return tuple(name for name in getattr(detector, 'classes', ())[1:]
                 if name and name not in {'N/A', '__background__', 'background'})
