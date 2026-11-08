#!/usr/bin/env python3
"""Sky + vegetation segmentation and dark-robust deep edge maps.

    python run.py image.jpg                 # -> ./out/image_*.png
    python run.py folder_of_images/ -o res  # every image in the folder
    python run.py img.png --fp16            # faster on CUDA

All weights are bundled in ./weights - nothing is downloaded at runtime.
"""
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vsedge.pipeline import Pipeline, process_file  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def collect(inputs):
    files = []
    for p in inputs:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(IMG_EXT))
        else:
            files += sorted(glob.glob(p)) or [p]
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files, globs or folders")
    ap.add_argument("-o", "--out", default="out", help="output folder (default ./out)")
    ap.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | mps | cpu")
    ap.add_argument("--fp16", action="store_true", help="CUDA autocast (faster, ~identical output)")
    g = ap.add_argument_group("what to compute")
    g.add_argument("--no-seg", action="store_true", help="skip sky/vegetation segmentation")
    g.add_argument("--no-edges", action="store_true", help="skip edge map")
    g = ap.add_argument_group("segmentation")
    g.add_argument("--seg-tile-short", type=int, default=768,
                   help="short side the image is resized to for the tiled 480px pass (default 768; "
                        "use 1008 to run tiles at native resolution of a 1008x1776 frame - slower, finer)")
    g.add_argument("--include-field", action="store_true", help="also count ADE 'field' as vegetation")
    g.add_argument("--no-seg-flip", action="store_true", help="disable horizontal-flip TTA")
    g.add_argument("--seg-on-enhanced", choices=["auto", "always", "never"], default="auto",
                   help="also segment the low-light-enhanced view (auto = only for dark images)")
    g.add_argument("--min-area-frac", type=float, default=1e-4,
                   help="drop isolated mask blobs smaller than this fraction of the image")
    g = ap.add_argument_group("edges")
    g.add_argument("--edge-detectors", default="dexined,teed,pidinet",
                   help="comma list from dexined,teed,pidinet")
    g.add_argument("--edge-scales", default="0.5,1.0", help="comma list of scales, e.g. 0.5,1.0,1.5")
    g.add_argument("--edge-tile", type=int, default=1024,
                   help="max crop size for the edge nets (lower it if you run out of GPU memory)")
    g.add_argument("--no-enhance", action="store_true", help="disable the Zero-DCE++ dark-region view")
    g.add_argument("--denoiser", choices=["scunet", "nlm", "none"], default="scunet",
                   help="denoiser applied to the low-light-enhanced view (default scunet)")
    g.add_argument("--edge-low", type=float, default=0.08, help="hysteresis low threshold (binary edges)")
    g.add_argument("--edge-high", type=float, default=0.20, help="hysteresis high threshold (binary edges)")
    g = ap.add_argument_group("outputs")
    g.add_argument("--no-panel", action="store_true", help="don't write the *_panel.jpg preview")
    g.add_argument("--no-save-enhanced", action="store_true", help="don't write *_enhanced.jpg")
    a = ap.parse_args()

    files = collect(a.inputs)
    if not files:
        sys.exit("no input images found")
    pipe = Pipeline(
        device=a.device, fp16=a.fp16, do_seg=not a.no_seg, do_edges=not a.no_edges,
        seg_tile_short=a.seg_tile_short, include_field=a.include_field, seg_flip=not a.no_seg_flip,
        seg_on_enhanced=a.seg_on_enhanced,
        edge_detectors=tuple(s.strip() for s in a.edge_detectors.split(",") if s.strip()),
        edge_scales=tuple(float(s) for s in a.edge_scales.split(",")), edge_tile=a.edge_tile,
        enhance=not a.no_enhance, denoise=a.denoiser, edge_low=a.edge_low, edge_high=a.edge_high,
        min_area_frac=a.min_area_frac,
    )
    for f in files:
        process_file(pipe, f, a.out, save_enhanced=not a.no_save_enhanced, save_panel=not a.no_panel)


if __name__ == "__main__":
    main()
