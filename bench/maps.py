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


def maze(cells=12, cell_px=10, wall=2, seed=0):
    """Recursive-backtracker maze scaled to a costmap.

    Forces a winding solution, so a planner that quietly returns
    start->goal straight-line is caught rather than flattered.
    """
    rng = np.random.default_rng(seed)
    visited = np.zeros((cells, cells), dtype=bool)
    # walls[r][c] = (north_open, west_open)
    open_n = np.zeros((cells, cells), dtype=bool)
    open_w = np.zeros((cells, cells), dtype=bool)
    stack = [(0, 0)]; visited[0, 0] = True
    while stack:
        r, c = stack[-1]
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < cells and 0 <= nc < cells and not visited[nr, nc]:
                nbrs.append((nr, nc, dr, dc))
        if not nbrs:
            stack.pop(); continue
        nr, nc, dr, dc = nbrs[int(rng.integers(len(nbrs)))]
        if dr == -1:   open_n[r, c] = True
        elif dr == 1:  open_n[nr, nc] = True
        elif dc == -1: open_w[r, c] = True
        else:          open_w[nr, nc] = True
        visited[nr, nc] = True
        stack.append((nr, nc))

    pitch = cell_px + wall
    n = cells * pitch + wall
    g = np.full((n, n), LETHAL, dtype=np.int16)
    for r in range(cells):
        for c in range(cells):
            r0, c0 = wall + r * pitch, wall + c * pitch
            g[r0:r0 + cell_px, c0:c0 + cell_px] = FREE
            if open_n[r, c]:
                g[r0 - wall:r0, c0:c0 + cell_px] = FREE
            if open_w[r, c]:
                g[r0:r0 + cell_px, c0 - wall:c0] = FREE
    start = (wall + cell_px // 2, wall + cell_px // 2)
    goal = (wall + (cells - 1) * pitch + cell_px // 2,
            wall + (cells - 1) * pitch + cell_px // 2)
    return g, start, goal


def rooms(n=200, seed=3, door=18):
    """Open space broken by long walls with offset doorways.

    door is the opening width in cells. The default is fine for scoring a
    global planner, which is a point. A controller carrying a footprint needs
    a wider one: inflating the jambs by the inscribed radius eats door - 2*r,
    and a doorway that lands on a crossing wall is narrower still.
    """
    rng = np.random.default_rng(seed)
    g = np.full((n, n), FREE, dtype=np.int16)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = LETHAL
    ys = [k * n // 5 for k in range(1, 5)]
    for y in ys:
        g[y - 2:y + 2, :] = LETHAL
        for _ in range(2):
            d = int(rng.integers(8, n - door - 8))
            g[y - 2:y + 2, d:d + door] = FREE
    x = 2 * n // 3
    g[:, x - 2:x + 2] = LETHAL
    for _ in range(3):                      # several doorways so the split stays passable
        d = int(rng.integers(8, n - door - 8))
        # keep the opening clear of a crossing wall, which would halve it
        for y in ys:
            if d - 6 < y < d + door + 6:
                d = min(n - door - 8, y + 8)
        g[d:d + door, x - 2:x + 2] = FREE
    return g, (5, 5), (n - 6, n - 6)


def hard_suite():
    """Maps where a straight start-to-goal line is definitely blocked."""
    out = []
    for cells, seed in ((10, 1), (14, 4)):
        g, s, gl = maze(cells=cells, cell_px=10, wall=2, seed=seed)
        out.append((f"maze-{g.shape[0]}", g, s, gl))
    seed = 0
    while True:                              # first seed whose doorways actually connect
        g, s, gl = rooms(200, seed=seed)
        if reachable(g, s, gl):
            out.append(("rooms-200", g, s, gl)); break
        seed += 1
    return out


def inflate(g, radius_cells=4, lethal=LETHAL, inscribed_cells=0):
    """Costmap inflation, the way nav2_costmap_2d's inflation layer works.

    Lethal cells keep their value; cells within radius get a cost that decays
    with distance. Without this a global planner hugs walls and any
    path-tracking controller with a lookahead clips the corner -- which is a
    property of the map, not of the controller.

    inscribed_cells reproduces the other half of that layer, the one that
    matters to a local controller: nav2 raises every cell closer than the
    robot's inscribed radius to lethal, which is what makes a centre-point
    collision check equivalent to a footprint check. Leave it at 0 for the
    global planners -- they are scored on paths, and a lethal band that thick
    closes corridors the robot can physically drive through.
    """
    g = g.copy()
    h, w = g.shape
    lethal_mask = g >= lethal
    dist = np.full((h, w), np.inf)
    dist[lethal_mask] = 0.0
    # two-pass chamfer distance transform
    for r in range(h):
        for c in range(w):
            best = dist[r, c]
            if r: best = min(best, dist[r-1, c] + 1)
            if c: best = min(best, dist[r, c-1] + 1)
            if r and c: best = min(best, dist[r-1, c-1] + 1.414)
            if r and c + 1 < w: best = min(best, dist[r-1, c+1] + 1.414)
            dist[r, c] = best
    for r in range(h - 1, -1, -1):
        for c in range(w - 1, -1, -1):
            best = dist[r, c]
            if r + 1 < h: best = min(best, dist[r+1, c] + 1)
            if c + 1 < w: best = min(best, dist[r, c+1] + 1)
            if r + 1 < h and c + 1 < w: best = min(best, dist[r+1, c+1] + 1.414)
            if r + 1 < h and c: best = min(best, dist[r+1, c-1] + 1.414)
            dist[r, c] = best
    band = (~lethal_mask) & (dist <= radius_cells)
    scaled = (AVOID_COST_INFLATE * np.exp(-1.2 * dist)).astype(np.int16)
    g[band] = np.maximum(g[band], scaled[band])
    if inscribed_cells:
        g[(~lethal_mask) & (dist <= inscribed_cells)] = lethal
    return g


AVOID_COST_INFLATE = 240


INSCRIBED_CELLS = 3        # 0.15 m at 0.05 m/cell, a TurtleBot4's inscribed radius


def controller_suite():
    """Maps with corridors a real robot fits down.

    The tight maze is the right stress test for a global planner but not for a
    local controller: its corridors are 0.5 m across, narrower than a
    TurtleBot4 plus the 0.4 m lookahead pure pursuit steers at. Comparing a
    tracker against a gap it cannot physically turn in measures the map.

    Reachability is checked against the inscribed band, not the raw walls, so
    every map here is passable by something with a footprint rather than by a
    point.
    """
    def passable(g, s, gl):
        return reachable(inflate(g.astype(np.int16), radius_cells=4,
                                 inscribed_cells=INSCRIBED_CELLS), s, gl)

    out = []
    g, s, gl = maze(cells=6, cell_px=22, wall=3, seed=2)
    assert passable(g, s, gl), "maze-wide closed by the inscribed band"
    out.append((f"maze-wide-{g.shape[0]}", g, s, gl))
    seed = 0
    while True:
        g, s, gl = rooms(200, seed=seed, door=30)
        if passable(g, s, gl):
            out.append(("rooms-200", g, s, gl)); break
        seed += 1
        assert seed < 60, "no rooms seed survives the inscribed band"
    return out
