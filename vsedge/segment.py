"""Sky + vegetation segmentation with DPT-Hybrid (ViT-R50) trained on ADE20K (150 classes).

Resolution handling for ~1008x1776 inputs (any size works):
  1. global pass  - whole image resized so its short side = `global_short` (480, DPT's training
                    crop size) -> scene-level context (sky vs. water vs. wall ...).
  2. tiled pass   - image resized so its short side = `tile_short` (default 768) and cut into
                    overlapping `tile` x `tile` crops (480, stride tile-overlap). Each crop runs at
                    the network's native resolution, outputs are blended back with linear ramps.
                    -> thin branches, small foliage gaps, sky holes.
  3. both passes (+ horizontal-flip TTA) are averaged in probability space, reduced to
     {sky, vegetation, best-other-class}, upsampled to the full input resolution and refined
     with a guided filter on the full-res image, then arg-maxed into the final masks.
"""
import torch
import torch.nn.functional as F

from . import weights
from .third_party.dpt.models import DPTSegmentationModel
from .utils import guided_filter, luminance, pad_to_multiple, resize, scaled_size, tiled_apply

# ADE20K (SceneParsing-150) 0-based indices of the DPT output channels
ADE_SKY = [2]
ADE_VEGETATION = {
    "tree": 4,
    "grass": 9,
    "plant": 17,
    "flower": 66,
    "palm": 72,
}
ADE_OPTIONAL_VEGETATION = {"field": 29}  # cultivated field: grass/crops but also bare farmland


class SkyVegSegmenter:
    def __init__(self, device, global_short=480, tile_short=768, tile=480, overlap=160,
                 flip_tta=True, include_field=False, fp16=False):
        self.device = device
        self.global_short, self.tile_short = global_short, tile_short
        self.tile, self.overlap, self.flip_tta = tile, overlap, flip_tta
        self.fp16 = fp16 and device.type == "cuda"
        veg = dict(ADE_VEGETATION)
        if include_field:
            veg.update(ADE_OPTIONAL_VEGETATION)
        self.veg_idx = sorted(veg.values())
        self.sky_idx = ADE_SKY
        other = torch.ones(150, dtype=torch.bool)
        other[self.veg_idx + self.sky_idx] = False
        self.other_mask = other.to(device)

        net = DPTSegmentationModel(150, backbone="vitb_rn50_384")
        net.load_state_dict(weights.load_state_dict("dpt_hybrid_ade20k_fp16.pt"))
        self.net = net.eval().to(device)

    # ------------------------------------------------------------------ network call
    @torch.no_grad()
    def _probs(self, x01):
        """x01: (1,3,h,w) RGB in [0,1] (any size) -> (1,150,h,w) class probabilities."""
        x = (x01 - 0.5) / 0.5
        xp, (h, w) = pad_to_multiple(x, 32)
        with torch.autocast("cuda", enabled=self.fp16):
            logits = self.net(xp)
            if self.flip_tta:
                logits = logits + torch.flip(self.net(torch.flip(xp, [3])), [3])
        return F.softmax(logits.float()[:, :, :h, :w], dim=1)

    def _reduce(self, p):
        """(1,150,h,w) -> (1,3,h,w): P(sky), P(vegetation)=sum of veg classes, max P(other class)."""
        sky = p[:, self.sky_idx].sum(1, keepdim=True)
        veg = p[:, self.veg_idx].sum(1, keepdim=True)
        oth = p[:, self.other_mask].amax(1, keepdim=True)
        return torch.cat([sky, veg, oth], 1)

    @torch.no_grad()
    def _canvas_probs(self, x01):
        H, W = x01.shape[-2:]
        ch, cw = scaled_size(H, W, short_side=self.tile_short)
        # global context pass
        g = resize(x01, scaled_size(H, W, short_side=self.global_short), "bicubic").clamp(0, 1)
        pg = resize(self._probs(g), (ch, cw))
        # high-resolution tiled pass
        c = resize(x01, (ch, cw), "bicubic").clamp(0, 1)
        pt = tiled_apply(self._probs, c, self.tile, self.overlap, 150)
        return self._reduce(0.5 * pg + 0.5 * pt)

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def __call__(self, x01, extra_views=(), guide=None, gf_radius=None, gf_eps=1e-3):
        """x01: (1,3,H,W) RGB in [0,1] at full resolution.
        extra_views: other photometric versions of the same image (e.g. low-light enhanced)
                     whose predictions are averaged in.
        guide: (1,1,H,W) image used for guided-filter refinement (default: luma of x01).
        Returns dict with full-res tensors sky_prob, veg_prob (1,1,H,W) and label (H,W) in
        {0: other, 1: sky, 2: vegetation}."""
        H, W = x01.shape[-2:]
        views = [x01] + list(extra_views)
        r = sum(self._canvas_probs(v) for v in views) / len(views)
        r = resize(r, (H, W), "bilinear")
        if guide is None:
            guide = luminance(x01)
        if gf_radius is None:
            gf_radius = max(2, round(min(H, W) / 128))
        r = guided_filter(guide, r, gf_radius, gf_eps).clamp_min(0)
        r = r / r.sum(1, keepdim=True).clamp_min(1e-6)
        lab = r.argmax(1)[0]  # 0 sky, 1 veg, 2 other
        label = torch.zeros_like(lab)
        label[lab == 0] = 1
        label[lab == 1] = 2
        return {"sky_prob": r[:, 0:1], "veg_prob": r[:, 1:2], "label": label}
