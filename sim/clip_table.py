"""Print the ten-clip table from the runs, rather than from memory.

    clip_table.py [run/log/batch.txt ...]

Every number in it comes out of the batch log and the files on disk: the
playback factor each clip was encoded at, the wall second of its capture the
robot arrived on, and the simulated seconds that took.  The README's first
version of this table was typed from the terminal and was wrong within a day
-- the clips were re-recorded and the arrivals moved by up to forty seconds,
and nothing in the document knew.

Where the numbers come from, in drive.sh's own terms: the capture starts, four
seconds later the goal goes out, and the arrival is grepped out of the
controller's log with its wall stamp.  So `END - 4` is the wall second of the
capture the robot arrived on and `(END - 8) / SPEED` is how many simulated
seconds it took from the goal, which is the only one of the three that is
comparable between runs -- the others carry whatever the machine was doing.

A clip recorded more than once appears more than once in the log; the last
block wins, because that is the run whose gif is on disk.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import os
import re
import sys

GIF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gif')
ORDER = ['planner-astar', 'planner-theta_star', 'planner-smac', 'planner-rrt',
         'planner-rrt_smac_hybrid', 'controller-dwa', 'controller-pure_pursuit',
         'controller-stanley', 'controller-teb', 'controller-mppi']
PREROLL = 4.0      # drive.sh: sleep 4 between the capture opening and the goal
PAD = 4.0          # drive.sh: END = reach - T0 + 4


def parse(paths):
    runs = {}
    tag = None
    for path in paths:
        if not os.path.exists(path):
            continue
        for line in open(path, errors='replace'):
            m = re.match(r'=== (\S+) :', line)
            if m:
                tag = m.group(1)
                if 'already recorded' not in line:
                    runs.setdefault(tag, {}).clear()
                continue
            if tag is None:
                continue
            m = re.search(r'playing back ([0-9.]+)x', line)
            if m:
                runs.setdefault(tag, {})['speed'] = float(m.group(1))
            m = re.search(r'reached ([0-9.]+)s into the capture', line)
            if m:
                runs.setdefault(tag, {})['end'] = float(m.group(1))
    return runs


def main():
    paths = sys.argv[1:] or [
        os.path.join(os.path.dirname(GIF), 'run', 'log', 'batch.txt'),
        os.path.join(os.path.dirname(GIF), 'run', 'astar_fix.log')]
    runs = parse(paths)
    # controller-dwa is astar's clip under a second name -- all.sh copies it
    if 'planner-astar' in runs:
        runs.setdefault('controller-dwa', {}).update(runs['planner-astar'])

    print('| clip | arrived | playback | size |')
    print('|---|---|---|---|')
    total = 0
    for tag in ORDER:
        r = runs.get(tag, {})
        f = os.path.join(GIF, f'{tag}.gif')
        kb = os.path.getsize(f) / 1048576.0 if os.path.exists(f) else 0.0
        total += kb
        if 'end' in r and 'speed' in r:
            sim = (r['end'] - PAD - PREROLL) / r['speed']
            when = f'{sim:.1f} s'
            speed = f'{r["speed"]:.1f}x'
        else:
            when = speed = '-'
        print(f'| {tag} | {when} | {speed} | {kb:.2f} MB |')
    print(f'\n{len(ORDER)} clips, {total:.1f} MB total')
    missing = [t for t in ORDER
               if not os.path.exists(os.path.join(GIF, f'{t}.gif'))]
    if missing:
        print('missing: ' + ', '.join(missing))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
