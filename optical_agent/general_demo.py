"""Small local demos: real NASA photo when prepared, explicit procedural fallback."""
import os
from pathlib import Path

import numpy as np

from optical_agent.image_input import read_image_bytes


def demo_image(variant='normal'):
    if variant not in {'normal', 'dark', 'lost'}:
        raise ValueError('演示图类型必须为normal、dark或lost')
    root = Path(os.environ.get('OPTICAL_AGENT_DATA_ROOT', r'D:\CodexData\optical_agent'))
    sample = root / 'general_v060/demo/astronaut.png'
    if sample.is_file():
        rgb = read_image_bytes(sample.read_bytes())
        provenance = {'kind': 'real_natural_photo', 'source': 'NASA / scikit-image astronaut',
                      'source_url': 'https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut',
                      'license': 'public_domain', 'ground_truth': 'no_object_annotations'}
    else:
        gray = np.tile(np.linspace(45, 190, 384, dtype=np.uint8), (256, 1))
        rgb = np.stack([gray, np.roll(gray, 45, axis=1), np.roll(gray, 90, axis=1)], axis=2)
        provenance = {'kind': 'procedural_exposure_fixture', 'source': 'generated_gradient',
                      'warning': '程序生成的曝光测试图，不是真实照片，也不含目标标注'}
    provenance['variant'] = variant
    if variant == 'dark':
        rgb = np.uint8(rgb.astype(float) * .45)
        provenance.update(kind='simulated_exposure_perturbation', brightness_factor=.45)
    elif variant == 'lost':
        rgb = np.zeros_like(rgb)
        provenance.update(kind='simulated_information_loss', brightness_factor=0.)
    return rgb, provenance
