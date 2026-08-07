#!/usr/bin/env python3
"""Times this repo's C++ A* on maps built to the Nav2 benchmark's geometry.

Reference numbers come from Table I of

  S. Macenski, et al. "Cost-Aware Kinematically Feasible Planning for Mobile
  and Surface Robotics." arXiv:2401.13078

which benchmarks the Nav2 Smac Planners against NavFn and SBPL ARA* on
10,000 m^2 random occupancy maps at 10/15/20 percent obstacle density, 5 cm
resolution, 1,000 start-goal pairs at least 3 m apart, on an AMD Ryzen 5 5600X.

bench/nav2_maps.py reproduces that map and query geometry. What it cannot
reproduce is their CPU or their planner's scope, so read the note at the end
before quoting any of this.

    python3 bench/nav2_maps.py 8 && ./bench/bench_astar bench/nav2_maps.bin 3
"""
import json, statistics, subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Table I, transcribed. t is milliseconds, l is metres.
NAV2 = {
    "10%": {"Hybrid-A*": (39.07, 51.41), "State Lattice": (42.31, 51.40),
            "Smac 2D-A*": (66.15, 50.96), "SBPL ARA*": (5640, 51.99), "NavFn": (71.11, 52.60)},
    "15%": {"Hybrid-A*": (40.73, 51.10), "State Lattice": (43.25, 51.15),
            "Smac 2D-A*": (85.63, 50.45), "SBPL ARA*": (6587, 54.87), "NavFn": (66.45, 52.50)},
    "20%": {"Hybrid-A*": (38.77, 50.78), "State Lattice": (39.40, 50.51),
            "Smac 2D-A*": (88.82, 49.65), "SBPL ARA*": (6633, 53.68), "NavFn": (61.02, 52.25)},
}


def main(pairs_per_map=8):
    subprocess.run([sys.executable, os.path.join(HERE, "nav2_maps.py"), str(pairs_per_map)],
                   cwd=ROOT, check=True)
    out = subprocess.run([os.path.join(HERE, "bench_astar"),
                          os.path.join("bench", "nav2_maps.bin"), "3"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    rows = json.loads(out)
    n = pairs_per_map
    groups = {"10%": rows[0:n], "15%": rows[n:2*n], "20%": rows[2*n:3*n]}

    print("\nThis repo, C++ A*, on Nav2-geometry maps")
    print(f"  {'density':>8}{'median ms':>11}{'mean ms':>10}{'path m':>9}{'expanded':>10}")
    mine = {}
    for d, grp in groups.items():
        ms = sorted(r["ms"] for r in grp)
        length = statistics.mean(r["path_cells"] for r in grp) * 0.05
        mine[d] = (statistics.median(ms), length)
        print(f"  {d:>8}{statistics.median(ms):>11.1f}{statistics.mean(ms):>10.1f}"
              f"{length:>9.1f}{statistics.mean(r['expanded'] for r in grp):>10.0f}")

    print("\nNav2 Table I, same map geometry, AMD Ryzen 5 5600X")
    print(f"  {'density':>8}" + "".join(f"{k:>16}" for k in NAV2["10%"]))
    for d in NAV2:
        print(f"  {d:>8}" + "".join(f"{v[0]:>15.1f}ms" for v in NAV2[d].values()))

    print("\n  ratio vs Nav2 Smac 2D-A* (their ms / ours):")
    for d in NAV2:
        print(f"    {d}: {NAV2[d]['Smac 2D-A*'][0] / mine[d][0]:.1f}x")

    print("""
Read with the caveats. Their CPU is much faster than the one these numbers were
taken on, which flatters this repo. Against that, the Smac 2D-A* they measure is
cost-aware and returns a smoothed path, and NavFn solves a full navigation
function -- both do more work per call than a plain octile A*. The honest claim
is that this planner is in the same order of magnitude on equivalent maps, not
that it beats Nav2.""")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
