"""Deterministic costmaps shared by the Python and C++ benchmarks.

Values follow the costmap convention the planners already assume:
0 free, 100 inflated, 254 lethal. Stored as int8 so the byte layout matches
nav_msgs/OccupancyGrid.data -- 254 round-trips through int8 as -2, which both
implementations read back as lethal.
"""
import numpy as np
from collections import deque

FREE, INFLATE, LETHAL = 0, 100, 254


def make_map(w, h, n_obstacles, seed):
    rng = np.random.default_rng(seed)
    g = np.full((h, w), FREE, dtype=np.int16)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = LETHAL

    for _ in range(n_obstacles):
        bw = int(rng.integers(w // 24, w // 8))
        bh = int(rng.integers(h // 24, h // 8))
        c = int(rng.integers(4, w - bw - 4))
        r = int(rng.integers(4, h - bh - 4))
        g[r - 2:r + bh + 2, c - 2:c + bw + 2] = np.maximum(
            g[r - 2:r + bh + 2, c - 2:c + bw + 2], INFLATE)
        g[r:r + bh, c:c + bw] = LETHAL

    start, goal = (3, 3), (h - 4, w - 4)
    for rc in (start, goal):
        g[rc[0] - 2:rc[0] + 3, rc[1] - 2:rc[1] + 3] = FREE
    return g, start, goal


def reachable(g, start, goal):
    """8-connected BFS, so a map is only used if a path actually exists."""
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    q = deque([start]); seen[start] = True
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] < 253:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return False


def suite():
    out = []
    for w, h, nobs in ((128, 128, 14), (256, 256, 26), (384, 384, 42)):
        seed = 0
        while True:
            g, s, gl = make_map(w, h, nobs, seed)
            if reachable(g, s, gl):
                out.append((f"{w}x{h}", g, s, gl))
                break
            seed += 1
    return out


def dump(path):
    """Binary the C++ harness reads: [n](w,h,sr,sc,gr,gc,int8 data)*"""
    with open(path, "wb") as f:
        cases = suite()
        f.write(np.int32(len(cases)).tobytes())
        for name, g, s, gl in cases:
            h, w = g.shape
            f.write(np.array([w, h, s[0], s[1], gl[0], gl[1]], dtype=np.int32).tobytes())
            f.write(g.astype(np.uint8).astype(np.int8).tobytes())
    return suite()


if __name__ == "__main__":
    for name, g, s, gl in dump("bench/maps.bin"):
        occ = float((g >= 253).mean())
        print(f"{name}: {occ*100:.1f}% lethal, start {s} -> goal {gl}")


def local_costmap(n=120, res=0.05, seed=7):
    """A 6x6 m local costmap with inflated obstacles, robot-centred at (0,0)."""
    rng = np.random.default_rng(seed)
    g = np.zeros((n, n), dtype=np.int16)
    for _ in range(10):
        r, c = int(rng.integers(6, n - 14)), int(rng.integers(6, n - 14))
        bh, bw = int(rng.integers(5, 12)), int(rng.integers(5, 12))
        g[r - 4:r + bh + 4, c - 4:c + bw + 4] = np.maximum(
            g[r - 4:r + bh + 4, c - 4:c + bw + 4], 90)
        g[r:r + bh, c:c + bw] = LETHAL
    mid = n // 2
    g[mid - 6:mid + 7, mid - 6:mid + 7] = 0          # keep the robot footprint clear
    origin = (-(n * res) / 2.0, -(n * res) / 2.0)
    return g, res, origin


def dump_local(path="bench/local.bin"):
    g, res, origin = local_costmap()
    with open(path, "wb") as f:
        n = g.shape[0]
        f.write(np.array([n, n], dtype=np.int32).tobytes())
        f.write(np.array([res, origin[0], origin[1]], dtype=np.float64).tobytes())
        f.write(g.astype(np.uint8).astype(np.int8).tobytes())
    return g, res, origin
