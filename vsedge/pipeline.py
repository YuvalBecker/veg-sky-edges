import os
import time

import cv2
import numpy as np
import torch

from .edges import EdgeEnsemble
from .segment import SkyVegSegmenter
from .utils import (edge_nms, hysteresis, luminance, pick_device, read_rgb, remove_small, to_numpy,
                    to_tensor)


class Pipeline:
    def __init__(self, device="auto", fp16=False, do_seg=True, do_edges=True,
                 seg_tile_short=768, include_field=False, seg_flip=True, seg_on_enhanced="auto",
                 edge_detectors=("dexined", "teed", "pidinet"), edge_scales=(0.5, 1.0), edge_tile=1024,
                 enhance=True, denoise="scunet", edge_low=0.08, edge_high=0.20, min_area_frac=1e-4,
                 verbose=True):
        self.device = pick_device(device)
        self.verbose = verbose
        self.seg_on_enhanced = seg_on_enhanced
        self.edge_low, self.edge_high = edge_low, edge_high
        self.min_area_frac = min_area_frac
        self.enhance = enhance
        t = time.time()
        self.seg = SkyVegSegmenter(self.device, tile_short=seg_tile_short, flip_tta=seg_flip,
                                   include_field=include_field, fp16=fp16) if do_seg else None
        self.edges = EdgeEnsemble(self.device, detectors=edge_detectors, scales=edge_scales,
                                  tile=edge_tile, enhance=enhance, denoise=denoise, fp16=fp16) if do_edges else None
        if self.edges is None and enhance and do_seg and seg_on_enhanced != "never":
            # segmentation of dark images still benefits from the enhanced view
            from .edges import ZeroDCEpp
            self._zdce = ZeroDCEpp(self.device)
        else:
            self._zdce = None
        self._log(f"models loaded on {self.device} in {time.time() - t:.1f}s")

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    @torch.no_grad()
    def __call__(self, rgb01):
        """rgb01: HxWx3 float RGB in [0,1] (any resolution). Returns dict of full-res numpy arrays."""
        H, W = rgb01.shape[:2]
        x = to_tensor(rgb01, self.device)
        out = {}
        L = luminance(x)
        mean_luma = float(L.mean())
        dark_frac = float((L < 0.12).float().mean())
        is_dark = mean_luma < 0.22 or dark_frac > 0.35
        out["stats"] = {"mean_luma": mean_luma, "dark_frac": dark_frac, "is_dark": is_dark}

        enh = None
        # the low-light branch only matters if a meaningful part of the frame is dark
        needs_enh = is_dark or float((L < 0.20).float().mean()) > 0.01
        out["stats"]["enhanced_view"] = bool(self.enhance and needs_enh)
        if self.enhance and needs_enh:
            t = time.time()
            if self.edges is not None:
                enh, info = self.edges.enhanced_view(x)
                out["stats"].update(info)
            elif self._zdce is not None:
                enh = self._zdce(x)
            self._log(f"  enhancement {time.time() - t:.1f}s")
        if enh is not None:
            out["enhanced"] = to_numpy(enh[0]).transpose(1, 2, 0)

        if self.seg is not None:
            t = time.time()
            use_enh = enh is not None and (self.seg_on_enhanced == "always" or
                                           (self.seg_on_enhanced == "auto" and is_dark))
            views = [enh] if use_enh else []
            guide = luminance(enh) if use_enh else L
            r = self.seg(x, extra_views=views, guide=guide)
            lab = to_numpy(r["label"]).astype(np.uint8)
            min_area = int(self.min_area_frac * H * W)
            sky = remove_small(lab == 1, min_area)
            veg = remove_small(lab == 2, min_area)
            out.update(sky_mask=sky, veg_mask=veg,
                       sky_prob=to_numpy(r["sky_prob"][0, 0]), veg_prob=to_numpy(r["veg_prob"][0, 0]))
            self._log(f"  segmentation {time.time() - t:.1f}s (enhanced view used: {use_enh})")

        if self.edges is not None:
            t = time.time()
            e = self.edges(x, enhanced=enh)
            soft = e["edges"]
            thin = edge_nms(soft)
            soft_np, thin_np = to_numpy(soft[0, 0]), to_numpy(thin[0, 0])
            out.update(edges=soft_np, edges_thin=thin_np,
                       edges_binary=hysteresis(thin_np, self.edge_low, self.edge_high),
                       dark_weight=to_numpy(e["dark_weight"][0, 0]))
            self._log(f"  edges {time.time() - t:.1f}s")
        return out


# ---------------------------------------------------------------------------------- saving
def _u8(a):
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def make_overlay(rgb01, res):
    base = res.get("enhanced", rgb01) if res["stats"]["is_dark"] else rgb01
    vis = base.copy()
    if "sky_mask" in res:
        for m, col in ((res["sky_mask"], (0.15, 0.55, 1.0)), (res["veg_mask"], (0.1, 0.9, 0.2))):
            vis[m] = 0.45 * vis[m] + 0.55 * np.array(col, np.float32)
    if "edges_binary" in res:
        vis[res["edges_binary"]] = (1.0, 0.1, 0.1)
    return vis


def make_panel(rgb01, res, width=1600):
    tiles = [rgb01]
    if "enhanced" in res:
        tiles.append(res["enhanced"])
    if "sky_mask" in res:
        lab = np.zeros(rgb01.shape, np.float32)
        lab[res["sky_mask"]] = (0.15, 0.55, 1.0)
        lab[res["veg_mask"]] = (0.1, 0.8, 0.2)
        tiles.append(lab)
    if "edges" in res:
        tiles.append(np.repeat(np.clip(res["edges"] * 1.5, 0, 1)[..., None], 3, 2))
        tiles.append(np.repeat(np.clip(res["edges_thin"] * 2.0, 0, 1)[..., None], 3, 2))
    tiles.append(make_overlay(rgb01, res))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(rgb01))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    panel = np.vstack(rows)
    s = width / panel.shape[1]
    return cv2.resize(panel, (width, int(panel.shape[0] * s)), interpolation=cv2.INTER_AREA)


def save_results(rgb01, res, out_dir, stem, save_enhanced=True, save_panel=True):
    os.makedirs(out_dir, exist_ok=True)
    p = lambda suffix: os.path.join(out_dir, f"{stem}_{suffix}")
    w = lambda name, img: cv2.imwrite(p(name), img)
    if "sky_mask" in res:
        w("sky_mask.png", res["sky_mask"].astype(np.uint8) * 255)
        w("veg_mask.png", res["veg_mask"].astype(np.uint8) * 255)
        w("sky_prob.png", _u8(res["sky_prob"]))
        w("veg_prob.png", _u8(res["veg_prob"]))
        label = np.zeros(res["sky_mask"].shape, np.uint8)
        label[res["sky_mask"]] = 1
        label[res["veg_mask"]] = 2
        w("labels.png", label)  # 0 other, 1 sky, 2 vegetation
    if "edges" in res:
        w("edges.png", _u8(res["edges"]))
        w("edges_thin.png", _u8(res["edges_thin"]))
        w("edges_binary.png", res["edges_binary"].astype(np.uint8) * 255)
        np.save(p("edges.npy"), res["edges"].astype(np.float16))
    if save_enhanced and "enhanced" in res:
        w("enhanced.jpg", cv2.cvtColor(_u8(res["enhanced"]), cv2.COLOR_RGB2BGR))
    w("overlay.jpg", cv2.cvtColor(_u8(make_overlay(rgb01, res)), cv2.COLOR_RGB2BGR))
    if save_panel:
        w("panel.jpg", cv2.cvtColor(_u8(make_panel(rgb01, res)), cv2.COLOR_RGB2BGR))


def process_file(pipe, path, out_dir, **kw):
    rgb = read_rgb(path)
    t = time.time()
    res = pipe(rgb)
    stem = os.path.splitext(os.path.basename(path))[0]
    save_results(rgb, res, out_dir, stem, **kw)
    pipe._log(f"{path}: {rgb.shape[1]}x{rgb.shape[0]} done in {time.time() - t:.1f}s  {res['stats']}")
    return res
