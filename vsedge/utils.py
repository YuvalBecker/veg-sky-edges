import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- device / io
def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype == np.uint16:  # keep 16-bit precision, map to [0,1]
        rgb = img[:, :, ::-1].astype(np.float32) / 65535.0
    else:
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb)


def to_tensor(rgb01, device):
    return torch.from_numpy(np.ascontiguousarray(rgb01)).permute(2, 0, 1)[None].float().to(device)


def to_numpy(t):
    return t.detach().float().cpu().numpy()


def luminance(x):
    """x: (B,3,H,W) RGB in [0,1] -> (B,1,H,W) Rec.709 luma."""
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


# --------------------------------------------------------------------------- padding / resizing
def pad_to_multiple(x, m, mode="reflect"):
    h, w = x.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph == 0 and pw == 0:
        return x, (h, w)
    # reflect padding needs pad < size; fall back to replicate for tiny inputs
    if mode == "reflect" and (ph >= h or pw >= w):
        mode = "replicate"
    return F.pad(x, (0, pw, 0, ph), mode=mode), (h, w)


def resize(x, size, mode="bilinear"):
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    if mode in ("bilinear", "bicubic"):
        return F.interpolate(x, size=size, mode=mode, align_corners=False,
                             antialias=size[0] < x.shape[-2] or size[1] < x.shape[-1])
    return F.interpolate(x, size=size, mode=mode)


def scaled_size(h, w, short_side=None, scale=None):
    s = scale if scale is not None else short_side / min(h, w)
    return max(1, int(round(h * s))), max(1, int(round(w * s)))


# --------------------------------------------------------------------------- tiling
def _tile_starts(n, tile, stride):
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile + 1, stride))
    if starts[-1] != n - tile:
        starts.append(n - tile)
    return starts


def _ramp(n, overlap, taper_start, taper_end, device):
    """1-D blending weight: linear ramps of length `overlap` on the sides that overlap a
    neighbouring tile, flat (=1) elsewhere, so image borders keep full weight."""
    w = torch.ones(n, device=device)
    k = min(overlap, n // 2)
    if k > 0:
        r = torch.linspace(0, 1, k + 2, device=device)[1:-1]
        if taper_start:
            w[:k] = r
        if taper_end:
            w[n - k:] = torch.flip(r, [0])
    return w


def tiled_apply(fn, x, tile, overlap, out_channels):
    """Run `fn` on overlapping crops of x (B,C,H,W) and stitch the (B,K,h,w) outputs back
    to full resolution with a linear-ramp blending window (no seams).
    fn must return an output with the same spatial size as its input crop."""
    B, _, H, W = x.shape
    th, tw = min(tile, H), min(tile, W)
    if th == H and tw == W:
        return fn(x).float()
    out = torch.zeros(B, out_channels, H, W, device=x.device)
    norm = torch.zeros(1, 1, H, W, device=x.device)
    for y0 in _tile_starts(H, th, max(1, th - overlap)):
        wy = _ramp(th, overlap, y0 > 0, y0 + th < H, x.device)
        for x0 in _tile_starts(W, tw, max(1, tw - overlap)):
            wx = _ramp(tw, overlap, x0 > 0, x0 + tw < W, x.device)
            w = (wy[:, None] * wx[None, :])[None, None]
            y = fn(x[:, :, y0:y0 + th, x0:x0 + tw]).float()
            out[:, :, y0:y0 + th, x0:x0 + tw] += y * w
            norm[:, :, y0:y0 + th, x0:x0 + tw] += w
    return out / norm.clamp_min(1e-6)


# --------------------------------------------------------------------------- guided filter
def _box(x, r):
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def guided_filter(guide, src, r, eps):
    """Edge-aware upsampling refinement (He et al.). guide: (1,1,H,W), src: (1,C,H,W)."""
    mean_I = _box(guide, r)
    mean_p = _box(src, r)
    cov_Ip = _box(guide * src, r) - mean_I * mean_p
    var_I = _box(guide * guide, r) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return _box(a, r) * guide + _box(b, r)


# --------------------------------------------------------------------------- edge post-processing
def _conv_tri(x, r):
    """Triangle filter (as in Dollar's edges toolbox) used before orientation estimation."""
    if r <= 0:
        return x
    k = torch.cat([torch.arange(1, r + 2), torch.arange(r, 0, -1)]).float().to(x.device)
    k = k / k.sum()
    x = F.pad(x, (r, r, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return F.conv2d(x, k.view(1, 1, -1, 1))


def _grad(x):
    gy, gx = torch.gradient(x, dim=(-2, -1))
    return gy, gx


def edge_nms(E, r=1, s=5, m=1.01):
    """Non-maximum suppression of a soft edge map (port of edgesNms from Dollar's toolbox,
    the same protocol used to evaluate HED / DexiNed / PiDiNet).  E: (1,1,H,W) in [0,1]."""
    Es = _conv_tri(E, 4)
    Oy, Ox = _grad(Es)
    Oyy, _ = _grad(Oy)
    Oxy, Oxx = _grad(Ox)
    O = torch.remainder(torch.atan(Oyy * torch.sign(-Oxy) / (Oxx + 1e-5)), math.pi)
    _, _, H, W = E.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=E.device, dtype=E.dtype),
                            torch.arange(W, device=E.device, dtype=E.dtype), indexing="ij")
    co, si = torch.cos(O)[0, 0], torch.sin(O)[0, 0]
    keep = torch.ones_like(E, dtype=torch.bool)
    for d in [-r, r] if r == 1 else [d for d in range(-r, r + 1) if d != 0]:
        gx = (xx + d * co) / max(W - 1, 1) * 2 - 1
        gy = (yy + d * si) / max(H - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], -1)[None]
        nb = F.grid_sample(E, grid, mode="bilinear", padding_mode="border", align_corners=True)
        keep &= E * m >= nb
    out = E * keep
    # suppress noisy responses at the image boundary
    if s > 0:
        ramp = torch.ones(H, W, device=E.device)
        for i in range(s):
            f = i / s
            ramp[i, :] = torch.minimum(ramp[i, :], torch.tensor(f, device=E.device))
            ramp[H - 1 - i, :] = torch.minimum(ramp[H - 1 - i, :], torch.tensor(f, device=E.device))
            ramp[:, i] = torch.minimum(ramp[:, i], torch.tensor(f, device=E.device))
            ramp[:, W - 1 - i] = torch.minimum(ramp[:, W - 1 - i], torch.tensor(f, device=E.device))
        out = out * ramp
    return out


def hysteresis(E, low, high):
    """Binary edges: keep connected weak (>low) segments that contain a strong (>high) pixel."""
    weak = (E > low).astype(np.uint8)
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    strong_labels = np.unique(lab[E > high])
    keep = np.zeros(n, dtype=bool)
    keep[strong_labels] = True
    keep[0] = False
    return keep[lab]


def estimate_noise_sigma(gray01):
    """Immerkaer's fast noise estimate on a [0,1] grayscale image (returns sigma in [0,1] units)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = gray01.shape
    conv = cv2.filter2D(gray01.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)
    return float(np.sqrt(math.pi / 2) * np.abs(conv[1:-1, 1:-1]).sum() / (6 * (w - 2) * (h - 2)))


def remove_small(mask, min_area):
    if min_area <= 0:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab]


_rng = np.random.default_rng(0).standard_normal(400000)
_a = np.abs(_rng)
_TRIM_HALFNORMAL = float(_a[_a <= np.quantile(_a, 0.9)].mean())  # ~0.6655
del _rng, _a


def noise_level_function(gray01, bins=16):
    """Per-pixel noise std (sensor noise is intensity dependent): robust Immerkaer/MAD estimate
    per intensity bin, interpolated back to every pixel.  gray01: HxW float [0,1]."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(gray01.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT))
    level = cv2.GaussianBlur(gray01.astype(np.float32), (0, 0), 2)
    edges_ = np.quantile(level, np.linspace(0, 1, bins + 1))
    centers, sig = [], []
    for i in range(bins):
        m = (level >= edges_[i]) & (level <= edges_[i + 1])
        if m.sum() < 500:
            continue
        # trimmed mean of |N(0, 36 s^2)|: robust to real structure (top 10% dropped) and, unlike a
        # median, still works on quantised 8-bit data where most Laplacian responses are exactly 0
        v = r[m]
        v = v[v <= np.quantile(v, 0.9)]
        sig.append(v.mean() / (6.0 * _TRIM_HALFNORMAL))
        centers.append(float(np.median(level[m])))
    if not sig:
        return np.full_like(gray01, estimate_noise_sigma(gray01), dtype=np.float32)
    sig = np.array(sig)
    return np.interp(level, centers, sig).astype(np.float32)


_GRAD_NOISE_GAIN = {}


def _grad_noise_gain(sigma):
    """Std of |Gaussian-derivative| response to unit white noise (computed once, numerically)."""
    if sigma not in _GRAD_NOISE_GAIN:
        rng = np.random.default_rng(0)
        n = rng.standard_normal((512, 512)).astype(np.float32)
        b = cv2.GaussianBlur(n, (0, 0), sigma)
        gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        _GRAD_NOISE_GAIN[sigma] = float(gx[32:-32, 32:-32].std())
    return _GRAD_NOISE_GAIN[sigma]


def edge_snr(gray01, sigma_map=None, scales=(1.0, 2.0, 3.0), bins=16):
    """Gradient significance of the ORIGINAL image, max over scales.

    Real camera noise after ISP/JPEG is spatially correlated (blotchy), so a white-noise model
    under-estimates it at coarser scales.  Instead the noise reference is measured *at each
    scale*: per intensity bin, the median gradient magnitude (most pixels of a dark region are
    noise/texture-only) is the typical background response, floored by the white-noise +
    quantisation prediction.  SNR = |grad| / reference."""
    g = gray01.astype(np.float32)
    level = cv2.GaussianBlur(g, (0, 0), 3)
    qs = np.quantile(level, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(qs, level, side="right") - 1, 0, bins - 1)
    if sigma_map is None:
        sigma_map = noise_level_function(g)
    out = np.zeros_like(g)
    for s in scales:
        b = cv2.GaussianBlur(g, (0, 0), s)
        gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        mag = np.sqrt(gx * gx + gy * gy)
        white = np.sqrt(sigma_map ** 2 + (1.0 / 255.0) ** 2 / 12.0) * _grad_noise_gain(s) * np.sqrt(2.0)
        med = np.array([np.median(mag[idx == i]) if np.any(idx == i) else 0.0 for i in range(bins)],
                       np.float32)
        ref = np.maximum(med[idx], white)
        out = np.maximum(out, mag / np.maximum(ref, 1e-6))
    return out
