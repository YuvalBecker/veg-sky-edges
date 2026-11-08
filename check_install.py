#!/usr/bin/env python3
"""Smoke test: verifies every bundled weight file loads and the full pipeline runs end-to-end
on a synthetic 1008x1776 frame (sky gradient, a dark lower half, some shapes)."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from vsedge.pipeline import Pipeline  # noqa: E402


def synthetic(h=1008, w=1776):
    y = np.linspace(0, 1, h)[:, None, None]
    img = np.zeros((h, w, 3), np.float32)
    img[:] = np.array([0.45, 0.65, 0.95]) * (1 - 0.4 * y)               # sky gradient
    img[h // 2:] = np.array([0.02, 0.025, 0.02])                          # dark ground
    img[h // 2 + 100: h // 2 + 300, 300:700] = (0.05, 0.06, 0.05)          # faint dark box
    rng = np.random.default_rng(0)
    img += rng.normal(0, 0.004, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


if __name__ == "__main__":
    pipe = Pipeline(device=sys.argv[1] if len(sys.argv) > 1 else "auto")
    res = pipe(synthetic())
    for k in ("sky_mask", "veg_mask", "edges", "edges_thin", "edges_binary"):
        assert k in res and res[k].shape == (1008, 1776), k
    print("OK - all models loaded and ran.")
