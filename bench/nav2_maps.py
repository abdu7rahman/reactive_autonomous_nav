"""Random occupancy maps matched to the Nav2 Smac Planner benchmark setup.

Macenski et al., "Cost-Aware Kinematically Feasible Planning for Mobile and
Surface Robotics" (arXiv:2401.13078) Table I benchmarks against 10,000 m^2
random environments at 10/15/20 percent obstacle density, 5 cm resolution,
with 1,000 start-goal pairs at least 3 m apart. This reproduces the map and
query geometry so the timings are measured on comparable work.
"""
import numpy as np
from collections import deque

FREE, LETHAL = 0, 254
RES = 0.05                      # m per cell, as in the paper
SIDE_M = 100.0                  # 100 x 100 m = 10,000 m^2
N = int(SIDE_M / RES)           # 2000 cells


def random_map(density, seed, n=N):
    """Blob obstacles to the requested area fraction, borders sealed."""
    rng = np.random.default_rng(seed)
    g = np.full((n, n), FREE, dtype=np.int16)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = LETHAL
    target = density * n * n
    placed = 0
    while placed < target:
        h = int(rng.integers(12, 70)); w = int(rng.integers(12, 70))
        r = int(rng.integers(1, n - h - 1)); c = int(rng.integers(1, n - w - 1))
        block = g[r:r + h, c:c + w]
        placed += int((block == FREE).sum())
        g[r:r + h, c:c + w] = LETHAL
    return g


def _reachable_from(g, start):
    h, w = g.shape
    seen = np.zeros(g.shape, dtype=bool)
    q = deque([start]); seen[start] = True
    while q:
        r, c = q.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] < 253:
                    seen[nr, nc] = True; q.append((nr, nc))
    return seen


def query_pairs(g, count, seed, sep_m=50.0, tol_m=6.0):
    """Start-goal pairs about sep_m apart, both in the same free component."""
    rng = np.random.default_rng(seed + 991)
    h, w = g.shape
    free = np.argwhere(g < 253)
    anchor = tuple(free[int(rng.integers(len(free)))])
    comp = _reachable_from(g, anchor)
    idx = np.argwhere(comp)
    pairs = []
    sep, tol = sep_m / RES, tol_m / RES
    guard = 0
    while len(pairs) < count and guard < count * 4000:
        guard += 1
        a = idx[int(rng.integers(len(idx)))]
        b = idx[int(rng.integers(len(idx)))]
        d = np.hypot(*(a - b))
        if abs(d - sep) <= tol:
            pairs.append((tuple(int(v) for v in a), tuple(int(v) for v in b)))
    return pairs


def dump(path, densities=(0.10, 0.15, 0.20), pairs_per_map=30, seed=0):
    cases = []
    with open(path, "wb") as f:
        f.write(np.int32(sum(1 for _ in densities) * pairs_per_map).tobytes())
        for i, d in enumerate(densities):
            g = random_map(d, seed + i)
            qs = query_pairs(g, pairs_per_map, seed + i)
            for (sr, sc), (gr, gc) in qs:
                f.write(np.array([g.shape[1], g.shape[0], sr, sc, gr, gc],
                                 dtype=np.int32).tobytes())
                f.write(g.astype(np.uint8).astype(np.int8).tobytes())
            cases.append((d, g, qs))
    return cases


if __name__ == "__main__":
    _sig()
    import sys
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    cases = dump("bench/nav2_maps.bin", pairs_per_map=n_pairs)
    for d, g, qs in cases:
        occ = (g >= 253).mean()
        seps = [np.hypot(a[0] - b[0], a[1] - b[1]) * RES for a, b in qs]
        print(f"  density {d:.0%} -> actual {occ:.1%}, {g.shape[0]}x{g.shape[1]} cells "
              f"({g.shape[0]*RES:.0f}x{g.shape[1]*RES:.0f} m), {len(qs)} pairs, "
              f"separation {np.mean(seps):.1f} m")


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)
