"""Prints the comparison table in bench/README.md from the four result files."""
import json, os
d = os.path.dirname(__file__)
L = lambda n: json.load(open(os.path.join(d, n)))
pa, ca, pd_, cd = L("py_astar.json"), L("cpp_astar.json"), L("py_dwa.json"), L("cpp_dwa.json")

print("\nA* global planner — median of 5 runs")
print(f"{'map':>10} {'python':>10} {'c++':>9} {'speedup':>9} {'py exp':>9} {'c++ exp':>9}")
for p, c in zip(pa, ca):
    print(f"{p['map']:>10} {p['ms']:9.2f}ms {c['ms']:8.3f}ms {p['ms']/c['ms']:8.0f}x "
          f"{p['expanded']:9d} {c['expanded']:9d}")

print("\nDWA rollout + score — median of 25 runs, horizon T=25")
print(f"{'window':>21} {'N':>6} {'python':>10} {'c++':>9} {'per-traj speedup':>18}")
for p, c in zip(pd_, cd):
    per = (p['ms'] / p['N']) / (c['ms'] / c['N'])
    print(f"{p['window']:>21} {p['N']:6d} {p['ms']:9.4f}ms {c['ms']:8.4f}ms {per:17.0f}x")
