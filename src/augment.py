"""Safe, GEOMETRY-PRESERVING augmentation (pixel-level ONLY).

Never crop / shear / resize / rotate — those move pixels and would break the [0,1000]
coordinate labels. We only perturb appearance, to model real low-quality scans/PDFs:
brightness, contrast, blur (low-res softness), gaussian pixel noise, faint scan lines.
Applied on-the-fly in the SFT collator when config.AUGMENT is on; text stays readable.
"""
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def augment(img, rng=None):
    rng = rng or random
    if rng.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.82, 1.18))
    if rng.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.82, 1.25))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.1)))   # scan/low-res softness
    if rng.random() < 0.4:                                                   # gaussian pixel noise
        a = np.asarray(img).astype(np.float32)
        a += np.random.normal(0.0, rng.uniform(4.0, 16.0), a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.15:                                                  # faint scan lines
        a = np.asarray(img).astype(np.float32)
        a[:: rng.choice([2, 3, 4]), :, :] *= rng.uniform(0.86, 0.97)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return img
