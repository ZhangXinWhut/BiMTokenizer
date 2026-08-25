"""Extract a smaller Leech codebook by uniform random sampling.

The source file contains the complete first shell of the Leech lattice:
196,560 normalized vectors with shape ``(196560, 24)``. Since the quantizer
performs nearest-neighbor search over codebook rows, any sampled collection of
rows is a valid smaller codebook. The random seed is fixed by default so that
the generated subset is reproducible.
"""
import numpy as np

SRC = "cache/leech_lattices_normalized.npy"  # Original 196,560 x 24 codebook
OUT = "cache/leech_subset_32768.npy"         # Output subset
M = 32768                                     # 2^15 entries; 15 bits/token
SEED = 0                                      # Preserve the default seed

cb = np.load(SRC)                              # Shape: (196560, 24)
N = cb.shape[0]
assert cb.shape[1] == 24, "The Leech codebook must have dimension 24"
print(
    f"Source codebook: {cb.shape}, dtype={cb.dtype}, "
    f"first-row norm≈{np.linalg.norm(cb[0]):.4f}"
)
assert M <= N, "The subset cannot be larger than the source codebook"

rng = np.random.default_rng(SEED)

# Sample rows uniformly without replacement. The full Leech shell is nearly
# uniform on the unit sphere, so this produces a representative fixed codebook.
idx = rng.choice(N, size=M, replace=False)
sub = cb[idx]

# Preserve unit-vector normalization (slicing should already preserve it).
sub = sub / np.linalg.norm(sub, axis=1, keepdims=True)
np.save(OUT, sub)
print(f"Saved subset -> {OUT}, shape={sub.shape}, dtype={sub.dtype}")
print(
    "Checks: "
    f"unique rows={np.unique(sub, axis=0).shape[0]}, "
    f"all unit vectors={np.allclose(np.linalg.norm(sub, axis=1), 1)}"
)
print(f"\nNext, update the config with:\n  codebook_size: {M}\n  codebook_path: \"{OUT}\"")
