"""Deep edge map that stays reliable in dark regions.

Detectors (ensemble, each run multi-scale):
  * DexiNed  (BIPED)          - dense, crisp low/mid-level edges
  * TEED     (BIPED)          - tiny, sharp edges, complements DexiNed
  * PiDiNet  (BSDS, carv4)    - pixel-difference convs, strong on texture-less/low-contrast boundaries

Photometric views (the dark-area part):
  * original image
  * Zero-DCE++ low-light enhanced image (deep curve estimation), optionally denoised with NLM at a
    strength derived from the *measured* noise level so amplified sensor noise does not become edges.
The two views are fused per pixel with a weight driven by the local (blurred) luminance: in bright
regions the original image dominates, in dark regions the enhanced view takes over.  Edge networks
respond to absolute contrast, so this is what recovers structure in shadows / night scenes.

Outputs: soft edge probability, NMS-thinned edges, and hysteresis-binarised edges, all full-res.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import weights
from .third_party.pidinet.config import config_model_converted
from .third_party.pidinet.pidinet import PiDiNet
from .third_party.scunet.network_scunet import SCUNet
from .third_party.teed.ted import TED
from .third_party.zerodce.model import enhance_net_nopool
from .utils import (edge_snr, estimate_noise_sigma, luminance, noise_level_function, pad_to_multiple, resize, scaled_size, tiled_apply,
                    to_numpy, to_tensor)

DEXINED_MEAN_BGR = (103.5, 116.2, 123.6)
TEED_MEAN_BGR = (104.007, 116.669, 122.679)
TEED_FLOOR = 0.44
BORDER = 24  # reflect-pad before each network call so image borders do not fire as edges
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ZeroDCEpp:
    """Zero-DCE++ low-light enhancement; curve maps estimated at 1/scale and applied at full res."""

    def __init__(self, device, scale=12):
        self.net = enhance_net_nopool(scale)
        self.net.load_state_dict(weights.load_state_dict("zerodce_pp.pth"))
        self.net = self.net.eval().to(device)
        self.scale = scale

    @torch.no_grad()
    def __call__(self, x01):
        n = self.net
        H, W = x01.shape[-2:]
        xd = resize(x01, (max(1, H // self.scale), max(1, W // self.scale)), "bilinear")
        x1 = n.relu(n.e_conv1(xd))
        x2 = n.relu(n.e_conv2(x1))
        x3 = n.relu(n.e_conv3(x2))
        x4 = n.relu(n.e_conv4(x3))
        x5 = n.relu(n.e_conv5(torch.cat([x3, x4], 1)))
        x6 = n.relu(n.e_conv6(torch.cat([x2, x5], 1)))
        xr = torch.tanh(n.e_conv7(torch.cat([x1, x6], 1)))
        xr = F.interpolate(xr, size=(H, W), mode="bilinear", align_corners=False)
        return n.enhance(x01, xr).clamp(0, 1)


class EdgeEnsemble:
    def __init__(self, device, detectors=("dexined", "teed", "pidinet"), scales=(0.5, 1.0),
                 tile=1024, overlap=128, enhance=True, denoise="scunet",
                 dark_lo=0.06, dark_hi=0.30, gate_k=1.0, fp16=False):
        self.device = device
        self.scales, self.tile, self.overlap = tuple(scales), tile, overlap
        self.enhance = enhance
        self.denoise = {True: "scunet", False: "none", None: "none"}.get(denoise, denoise)
        self.scunet = None
        if enhance and self.denoise == "scunet":
            m = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
            m.load_state_dict(weights.load_state_dict("scunet_color_real_psnr_fp16.pth"))
            self.scunet = m.eval().to(device)
        self.dark_lo, self.dark_hi = dark_lo, dark_hi
        self.gate_k = gate_k  # 0 disables the contrast gate
        self.fp16 = fp16 and device.type == "cuda"
        self.dets = {}
        for d in detectors:
            if d == "dexined":
                m = torch.jit.load(weights.path("dexined_biped_fp16.ts"), map_location="cpu").float()
                self.dets[d] = m.eval().to(device)
            elif d == "teed":
                m = TED()
                m.load_state_dict(weights.load_state_dict("teed_biped.pth"))
                self.dets[d] = m.eval().to(device)
            elif d == "pidinet":
                m = PiDiNet(60, config_model_converted("carv4"), dil=24, sa=True, convert=True)
                m.load_state_dict(weights.load_state_dict("pidinet_bsds_carv4_converted.pth"))
                self.dets[d] = m.eval().to(device)
            else:
                raise ValueError(f"unknown edge detector {d}")
        self.zdce = ZeroDCEpp(device) if enhance else None

    # ------------------------------------------------------------------ single detector
    def _run(self, name, x01):
        """x01 (1,3,h,w) RGB [0,1] -> (1,1,h,w) edge probability."""
        net = self.dets[name]
        h, w = x01.shape[-2:]
        b = min(BORDER, h - 1, w - 1)
        xb = F.pad(x01, (b, b, b, b), mode="reflect") if b > 0 else x01
        xp, _ = pad_to_multiple(xb, 16)
        if name in ("dexined", "teed"):
            mean = DEXINED_MEAN_BGR if name == "dexined" else TEED_MEAN_BGR
            x = torch.flip(xp, [1]) * 255.0 - torch.tensor(mean, device=xp.device).view(1, 3, 1, 1)
        else:
            x = (xp - torch.tensor(IMAGENET_MEAN, device=xp.device).view(1, 3, 1, 1)) / \
                torch.tensor(IMAGENET_STD, device=xp.device).view(1, 3, 1, 1)
        with torch.autocast("cuda", enabled=self.fp16):
            if name == "dexined":
                e = torch.sigmoid(net(x)[:, -1:])  # fused (block_cat) output
            elif name == "teed":
                # TEED's fused sigmoid has a constant ~0.44 floor on flat regions (the authors
                # min-max normalise per image); remove it with a fixed affine so tiles/views agree
                e = ((torch.sigmoid(net(x)[-1]) - TEED_FLOOR) / (1 - TEED_FLOOR)).clamp(0, 1)
            else:
                e = net(x)[-1]  # already sigmoid
        return e.float()[:, :, b:b + h, b:b + w]

    @torch.no_grad()
    def _view_edges(self, x01, scale_consensus=False):
        """Detector-mean per scale, combined over scales at full resolution.
        scale_consensus=False: arithmetic mean over scales.
        scale_consensus=True : geometric mean over scales - a response must persist across scales.
            Noise-induced edges are scale-specific while real boundaries survive downscaling, so this
            is used for the (noisy) enhanced view."""
        H, W = x01.shape[-2:]
        per_scale = []
        for s in self.scales:
            xs = x01 if s == 1.0 else resize(x01, scaled_size(H, W, scale=s), "bicubic").clamp(0, 1)
            acc = 0
            for name in self.dets:
                acc = acc + tiled_apply(lambda t, nm=name: self._run(nm, t), xs, self.tile, self.overlap, 1)
            per_scale.append(resize(acc / len(self.dets), (H, W), "bilinear").clamp(0, 1))
        if scale_consensus and len(per_scale) > 1:
            return torch.exp(sum(torch.log(p.clamp_min(1e-4)) for p in per_scale) / len(per_scale))
        return sum(per_scale) / len(per_scale)

    def _denoise(self, x01, enh):
        """Denoise the enhanced view where enhancement amplified the signal.

        Enhancement multiplies sensor noise, demosaicing/JPEG artefacts and quantisation error by the
        local gain, producing blotchy, spatially correlated noise that edge networks happily trace.
        * scunet (default): SCUNet trained on realistic camera degradations (blind; handles correlated
          noise and compression), run tiled at full resolution.
        * nlm: OpenCV non-local means, strength = gain x measured noise of the original.
        The result is blended in proportionally to the local gain, so untouched bright regions keep
        their original detail."""
        g0 = to_numpy(luminance(x01)[0, 0])
        g1 = to_numpy(luminance(enh)[0, 0])
        gain = cv2.GaussianBlur(g1, (0, 0), 4) / (cv2.GaussianBlur(g0, (0, 0), 4) + 2.0 / 255.0)
        amplified = gain > 1.5
        info = {"median_gain": float(np.median(gain[amplified])) if amplified.any() else 1.0}
        if self.denoise == "none" or amplified.mean() < 0.005:
            return enh, info
        if self.denoise == "scunet":
            with torch.autocast("cuda", enabled=self.fp16):
                den = tiled_apply(lambda t: self.scunet(t).float(), enh, 512, 64, 3).clamp(0, 1)
        elif self.denoise == "nlm":
            nlf = noise_level_function(g0) * 255.0
            sigma = max(float(np.median(nlf[amplified])), 0.5) * info["median_gain"]
            sigma = max(sigma, estimate_noise_sigma(g1) * 255.0)
            info["nlm_h"] = h = float(np.clip(sigma, 3, 30))
            u8 = (to_numpy(enh[0]).transpose(1, 2, 0) * 255 + 0.5).clip(0, 255).astype(np.uint8)
            den = cv2.fastNlMeansDenoisingColored(u8, None, h, h, 5, 21).astype(np.float32) / 255.0
            den = to_tensor(den, enh.device)
        else:
            raise ValueError(f"unknown denoiser {self.denoise}")
        alpha = torch.from_numpy(np.clip((gain - 1.5) / 3.0, 0, 1).astype(np.float32))[None, None].to(enh.device)
        return alpha * den + (1 - alpha) * enh, info

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def enhanced_view(self, x01):
        if self.zdce is None:
            return None, None
        return self._denoise(x01, self.zdce(x01))

    @torch.no_grad()
    def __call__(self, x01, enhanced=None):
        """x01: (1,3,H,W) full-res RGB [0,1]. Returns dict(edges=(1,1,H,W) soft map, dark_weight)."""
        H, W = x01.shape[-2:]
        # local luminance of the ORIGINAL image decides which view to trust
        L = luminance(x01)
        k = max(3, int(round(min(H, W) / 64)) | 1)
        L = F.avg_pool2d(F.pad(L, (k // 2,) * 4, mode="replicate"), k, stride=1)
        wdark = ((self.dark_hi - L) / (self.dark_hi - self.dark_lo)).clamp(0, 1)
        wdark = wdark * wdark * (3 - 2 * wdark)  # smoothstep
        e_orig = self._view_edges(x01)
        if enhanced is None or float(wdark.max()) < 0.02:  # nothing dark -> enhanced view unused
            return {"edges": e_orig, "dark_weight": wdark if enhanced is not None else torch.zeros_like(e_orig)}
        e_enh = self._view_edges(enhanced, scale_consensus=True) * self._contrast_gate(x01)
        # bright: original; dark: enhanced view, never weaker than what the original already sees
        e = (1 - wdark) * e_orig + wdark * torch.maximum(e_enh, e_orig)
        return {"edges": e, "dark_weight": wdark}

    def _contrast_gate(self, x01):
        """Physical floor for dark regions: an enhanced-view edge is only trusted where the ORIGINAL
        image has a gradient that is significant w.r.t. its own (intensity-dependent) noise level.
        Below that floor nothing can tell structure from noise / quantisation, so it is suppressed."""
        g = to_numpy(luminance(x01)[0, 0])
        snr = edge_snr(g, noise_level_function(g))
        lo, hi = 1.5 * self.gate_k, 3.5 * self.gate_k
        t = np.clip((snr - lo) / max(hi - lo, 1e-6), 0, 1)
        t = t * t * (3 - 2 * t)
        t = cv2.dilate(t, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))  # nets localise +-2px
        t = cv2.GaussianBlur(t, (0, 0), 1.0)
        return torch.from_numpy(t.astype(np.float32))[None, None].to(x01.device)
