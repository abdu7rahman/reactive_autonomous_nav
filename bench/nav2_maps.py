"""Random occupancy maps matched to the Nav2 Smac Planner benchmark setup.

Macenski et al., "Cost-Aware Kinematically Feasible Planning for Mobile and
Surface Robotics" (arXiv:2401.13078) Table I benchmarks against 10,000 m^2
random environments at 10/15/20 percent obstacle density, 5 cm resolution,
with 1,000 start-goal pairs at least 3 m apart. This reproduces the map and
query geometry so the timings are measured on comparable work.
"""
import numpy as np
from collections import deque


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)

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


def query_pairs(g, count, seed, min_sep_m=3.0):
    """Uniform random start-goal pairs in one free component, min_sep_m apart.

    The paper's wording is the specification: "we generated 1,000 verified
    start-goal pairs with minimum separation of 3m". This drew pairs from a
    50 +/- 6 m band instead, which is not the same experiment and not the one
    the docstring above claimed. Uniform pairs in a 100 x 100 m square average
    52.1 m apart, which is why their l_path column reads 49.65 to 52.60 m; a
    band at 50 m reproduces that mean and throws away the spread around it,
    and A* time is superlinear in separation, so the tails are exactly the
    part that decides a mean. The band also made the l_path column useless as
    a cross-check, because it agreed with the paper by construction.
    """
    rng = np.random.default_rng(seed + 991)
    free = np.argwhere(g < 253)
    anchor = tuple(free[int(rng.integers(len(free)))])
    comp = _reachable_from(g, anchor)
    idx = np.argwhere(comp)
    min_sep = min_sep_m / RES
    pairs = []
    guard = 0
    while len(pairs) < count and guard < count * 100:
        guard += 1
        a = idx[int(rng.integers(len(idx)))]
        b = idx[int(rng.integers(len(idx)))]
        if np.hypot(*(a - b)) >= min_sep:
            pairs.append((tuple(int(v) for v in a), tuple(int(v) for v in b)))
    return pairs


# First word of the dump. The old layout led with a positive case count and
# repeated the whole grid for every query, which at 2000 x 2000 is 4 MB a
# query: the paper's 1,000 pairs a density would have been a 12 GB file, and
# the bench ran 8 instead. Grouped, the three maps cost 12 MB however many
# queries hang off them. Negative so bench_astar can never mistake it for a
# count.
MAGIC_GROUPED = -2


def dump(path, densities=(0.10, 0.15, 0.20), pairs_per_map=1000, seed=0):
    cases = []
    with open(path, "wb") as f:
        f.write(np.int32(MAGIC_GROUPED).tobytes())
        f.write(np.int32(sum(1 for _ in densities)).tobytes())
        for i, d in enumerate(densities):
            g = random_map(d, seed + i)
            qs = query_pairs(g, pairs_per_map, seed + i)
            f.write(np.array([g.shape[1], g.shape[0]], dtype=np.int32).tobytes())
            f.write(g.astype(np.uint8).astype(np.int8).tobytes())
            f.write(np.int32(len(qs)).tobytes())
            for (sr, sc), (gr, gc) in qs:
                f.write(np.array([sr, sc, gr, gc], dtype=np.int32).tobytes())
            cases.append((d, g, qs))
    return cases


if __name__ == "__main__":
    _sig()
    import sys
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    cases = dump("bench/nav2_maps.bin", pairs_per_map=n_pairs)
    for d, g, qs in cases:
        occ = (g >= 253).mean()
        seps = [np.hypot(a[0] - b[0], a[1] - b[1]) * RES for a, b in qs]
        print(f"  density {d:.0%} -> actual {occ:.1%}, {g.shape[0]}x{g.shape[1]} cells "
              f"({g.shape[0]*RES:.0f}x{g.shape[1]*RES:.0f} m), {len(qs)} pairs, "
              f"separation {np.mean(seps):.1f} m")


