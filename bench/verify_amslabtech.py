#!/usr/bin/env python3
"""Checks that baseline_amslabtech.cpp still holds upstream's code.

The other two C++ baselines are #included straight from the fetched source.
amslabtech's is a ROS node, so its scoring functions had to be transcribed out
of it -- which is only fair to call a baseline if the transcription is actually
theirs. This diffs each lifted function body against the upstream file, ignoring
whitespace, and prints the substitutions that are expected.

    bash bench/fetch_baselines.sh && python3 bench/verify_amslabtech.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.join(HERE, ".third_party", "amslabtech_dwa_planner.cpp")
MINE = os.path.join(HERE, "baseline_amslabtech.cpp")

# The stub boundary: upstream's types replaced by the shims at the top of
# baseline_amslabtech.cpp. Everything else has to match.
EXPECTED = [("Eigen::Vector3d", "Vector3d")]

FNS = [
    ("void DWAPlanner::motion(State &state, const double velocity, const double yawrate)",
     "static void motion(State& state, const double velocity, const double yawrate)"),
    ("DWAPlanner::Window DWAPlanner::calc_dynamic_window(void)",
     "static Window calc_dynamic_window(void)"),
    ("float DWAPlanner::calc_to_goal_cost(const std::vector<State> &traj, const Eigen::Vector3d &goal)",
     "static float calc_to_goal_cost(const std::vector<State>& traj, const Vector3d& goal)"),
    ("float DWAPlanner::calc_speed_cost(const std::vector<State> &traj)",
     "static float calc_speed_cost(const std::vector<State>& traj)"),
    ("std::vector<DWAPlanner::State> DWAPlanner::generate_trajectory(const double velocity, const double yawrate)",
     "static std::vector<State> generate_trajectory(const double velocity, const double yawrate)"),
]


def body(src, sig):
    i = src.index(sig)
    j = src.index("{", i + len(sig))
    depth, k = 0, j
    while True:
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    return re.sub(r"\s+", " ", src[j:k + 1]).strip()


def main():
    if not os.path.exists(UP):
        print("run bash bench/fetch_baselines.sh first", file=sys.stderr)
        return 2
    up, mine = open(UP).read(), open(MINE).read()
    bad = 0
    print("  %-24s %s" % ("function", "against amslabtech/dwa_planner@master"))
    for a, b in FNS:
        name = a.split("DWAPlanner::")[-1].split("(")[0]
        want, got = body(up, a), body(mine, b)
        if want == got:
            print("  %-24s identical" % name)
            continue
        for src, dst in EXPECTED:                       # apply the stub swaps
            want = want.replace(src, dst)
        if want == got:
            subs = ", ".join("%s -> %s" % p for p in EXPECTED)
            print("  %-24s identical after the stub swap (%s)" % (name, subs))
        else:
            bad += 1
            print("  %-24s DIFFERS" % name)
            print("     upstream: %s" % want[:160])
            print("     here    : %s" % got[:160])
    print("\n%s" % ("every lifted body is upstream's" if not bad
                    else "%d function(s) have drifted" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
