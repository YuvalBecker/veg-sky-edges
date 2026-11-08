"""Locating and loading the bundled weights (all stored inside the repo under ./weights).

Large checkpoints are stored as fp16 and split into <100 MB parts so the repo can be
cloned with plain git (no LFS). Parts are re-assembled in memory and verified by sha256.
"""
import glob
import hashlib
import io
import os

import torch

WEIGHTS_DIR = os.environ.get(
    "VSEDGE_WEIGHTS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
)

SHA256 = {
    # sha256 of the re-assembled fp16 DPT checkpoint
    "dpt_hybrid_ade20k_fp16.pt": "8750350d03babd87b3f24dbc486e1ca66e92398b21de81d293a512d806b4022b",
}


def path(name):
    p = os.path.join(WEIGHTS_DIR, name)
    if not os.path.exists(p) and not glob.glob(p + ".part*"):
        raise FileNotFoundError(
            f"Missing weight file '{name}' in {WEIGHTS_DIR}. "
            "Did the clone finish? (git status / git fsck) - or set VSEDGE_WEIGHTS."
        )
    return p


def load_bytes(name):
    """Return the file contents, re-assembling '<name>.partNN' chunks if needed."""
    p = path(name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            data = f.read()
    else:
        parts = sorted(glob.glob(p + ".part*"))
        data = b"".join(open(q, "rb").read() for q in parts)
    if name in SHA256:
        h = hashlib.sha256(data).hexdigest()
        if h != SHA256[name]:
            raise RuntimeError(f"Checksum mismatch for {name}: got {h}, expected {SHA256[name]} (corrupt clone?)")
    return data


def load_state_dict(name):
    sd = torch.load(io.BytesIO(load_bytes(name)), map_location="cpu", weights_only=True)
    return {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
