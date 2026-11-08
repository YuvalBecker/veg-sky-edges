# veg-sky-edges

Sky + vegetation segmentation and a dark-robust deep edge map for full-resolution frames
(designed around ~1008×1776, works at any size). **All weights are in the repo** — clone and run,
nothing is downloaded at runtime.

```bash
git clone https://github.com/YuvalBecker/veg-sky-edges.git
cd veg-sky-edges
pip install -r requirements.txt          # torch, timm, einops, opencv-python, numpy
python check_install.py                  # loads every model, runs a synthetic 1008x1776 frame
python run.py path/to/image.png          # -> ./out/image_*.png
python run.py path/to/folder -o results --fp16   # whole folder, CUDA autocast
```

## Outputs (all at the input's full resolution)

| file | content |
|---|---|
| `*_sky_mask.png`, `*_veg_mask.png` | binary masks (0/255) |
| `*_sky_prob.png`, `*_veg_prob.png` | soft confidence after guided-filter refinement |
| `*_labels.png` | 0 = other, 1 = sky, 2 = vegetation |
| `*_edges.png` / `*_edges.npy` | soft edge probability (8-bit PNG / fp16 array) |
| `*_edges_thin.png` | NMS-thinned edges (1-px, same protocol as BSDS/BIPED evaluation) |
| `*_edges_binary.png` | hysteresis-thresholded thin edges (`--edge-low/--edge-high`) |
| `*_enhanced.jpg` | low-light-enhanced + denoised view (only when the frame has dark regions) |
| `*_overlay.jpg`, `*_panel.jpg` | visual check |

## How it works

### Segmentation — DPT-Hybrid (ViT-B/R50) trained on ADE20K
ADE20K is used because it has explicit `sky`, `tree`, `grass`, `plant`, `flower`, `palm` classes
(vegetation = sum of those; `--include-field` adds `field`). DPT-Hybrid scores 49.0 mIoU on ADE20K
(better than DPT-Large there).

Resolution / cropping for 1008×1776:
1. **Global pass** – whole frame resized to short side 480 (the training crop size) for scene context.
2. **Tiled pass** – frame resized to short side `--seg-tile-short` (768 default; 1008 = native res for a
   1008×1776 frame), cut into overlapping 480×480 crops (stride 320). Each crop runs at the net's native
   size; outputs are stitched back with linear-ramp blending weights (no seams, borders keep full weight).
3. Both passes + horizontal-flip TTA are averaged as probabilities, reduced to {sky, vegetation,
   best-other-class}, **bilinearly upsampled to full resolution and refined with a guided filter on the
   full-res image** (so mask boundaries follow real image edges — branches against sky), then arg-maxed.
4. For dark frames the low-light-enhanced view is segmented too and averaged in, and used as the
   guided-filter guide.

### Edges — deep ensemble, built for dark regions
* **Detectors:** DexiNed (BIPED) + TEED (BIPED) + PiDiNet (BSDS, carv4/sa/dil), each multi-scale
  (`--edge-scales 0.5,1.0` default), averaged. Each net call is reflect-padded so frame borders don't fire.
  TEED's constant ~0.44 output floor is removed with a fixed affine so tiles and views agree.
* **Dark-region branch:**
  1. **Zero-DCE++** estimates per-pixel enhancement curves (deep, zero-reference low-light enhancement).
  2. Enhancement multiplies sensor noise, demosaic/JPEG artefacts and quantisation error by the local gain,
     so the enhanced view is denoised with **SCUNet** (trained on realistic camera degradations: blind,
     handles spatially-correlated noise and compression), blended in proportionally to the local gain.
     (`--denoiser nlm` uses gain-scaled non-local means instead.)
  3. Edges on the enhanced view use a **scale-consensus** (geometric mean over scales): noise edges are
     scale-specific, real boundaries persist.
  4. A **significance gate** from the *original* frame: gradient magnitude vs. a noise reference measured
     per intensity bin and per scale (handles correlated noise), floored by the white-noise + 1-LSB
     quantisation prediction. Below that floor no method can separate structure from noise, so those
     enhanced-view responses are suppressed.
  5. Per-pixel fusion by local luminance of the original (smoothstep between 0.06 and 0.30): bright
     regions use the original-view edges, dark regions use `max(enhanced, original)`.
* **Post-processing:** NMS (port of Dollár's `edgesNms`) and hysteresis.
* Large frames are processed in overlapping tiles (`--edge-tile`, default 1024) with the same blending.

## Useful options

```
--device cuda|cuda:1|mps|cpu       default: auto
--fp16                             CUDA autocast for all nets
--seg-tile-short 1008              finer masks (tiles at native res of a 1008x1776 frame), slower
--include-field                    count ADE "field" as vegetation
--seg-on-enhanced auto|always|never
--edge-detectors dexined,teed,pidinet
--edge-scales 0.5,1.0,1.5          classic multi-scale protocol; 1.5 needs more GPU memory
--denoiser scunet|nlm|none         denoiser for the enhanced (dark) view
--no-enhance                       disable the whole dark-region branch
--edge-tile 512                    lower if you run out of GPU memory
--edge-low 0.08 --edge-high 0.20   hysteresis thresholds for *_edges_binary.png
--no-seg / --no-edges
```

Python API:

```python
from vsedge.pipeline import Pipeline
from vsedge.utils import read_rgb
pipe = Pipeline(device="cuda", fp16=True)
res = pipe(read_rgb("frame.png"))      # HxWx3 float RGB in [0,1] -> dict of full-res numpy arrays
res["sky_mask"], res["veg_mask"], res["edges"], res["edges_thin"], res["edges_binary"]
```

## Speed
Measured on a 2-core CPU (no GPU) for one 1008×1776 frame: ~75 s for a bright frame,
~100–170 s for a dark frame (SCUNet and the second edge view dominate). On a GPU it is far faster;
use `--fp16` on CUDA.

## Known limitations
* Very dark frames from heavily compressed video (flat, blocky macroblocks) still produce some
  axis-aligned block edges: those steps really exist in the data. Clean camera frames do not have this.
* "Vegetation" follows ADE20K semantics: dry grass on a hillside can be labelled `hill/earth`.
* Weights are stored fp16 (DPT split into <100 MB parts, re-assembled and sha256-checked at load); the
  difference from fp32 is negligible (max abs edge diff ~5e-3).

## Third-party models and licenses
| component | source | license |
|---|---|---|
| DPT-Hybrid ADE20K | isl-org/DPT | MIT |
| DexiNed (BIPED) | xavysp/DexiNed, weights via opencv_zoo (block-int8, dequantised to fp16) | MIT |
| TEED (BIPED) | xavysp/TEED | MIT |
| SCUNet real-noise | cszn/SCUNet | Apache-2.0 |
| PiDiNet (BSDS) | hellozhuo/pidinet | **research use only** (commercial use: contact authors) |
| Zero-DCE++ | Li-Chongyi/Zero-DCE_extension | **CC BY-NC 4.0** (non-commercial) |

For commercial use, drop PiDiNet (`--edge-detectors dexined,teed`) and Zero-DCE++ (`--no-enhance`)
or replace them. Each `vsedge/third_party/*` folder keeps its original license.
