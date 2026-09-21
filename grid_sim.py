"""Grid-world simulator: drive a robot with the learned policy and record the trajectory."""
import numpy as np

# 8 headings, clockwise from North. (dx, dy) with y pointing DOWN on screen.
DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
HEADING_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SENSOR_OFFSETS = {"front": 0, "left": -2, "right": 2, "back": 4}   # in 45-degree steps

TURN_OPTIONS = {
    "Straight": 0,
    "Slight left (45°)": -1,
    "Slight right (45°)": 1,
    "Sharp left (90°)": -2,
    "Sharp right (90°)": 2,
}
LAYOUTS = ["Empty room", "Room with pillar", "L-shaped room", "Ring corridor"]


def make_layout(name, w, h):
    """Boolean wall map (h x w); True = wall. The outer border is always wall."""
    walls = np.zeros((h, w), dtype=bool)
    walls[0, :] = walls[-1, :] = True
    walls[:, 0] = walls[:, -1] = True
    if name == "Room with pillar":
        pw, ph = max(2, w // 5), max(2, h // 4)
        x0, y0 = w // 2 - pw // 2, h // 2 - ph // 2
        walls[y0:y0 + ph, x0:x0 + pw] = True
    elif name == "L-shaped room":
        walls[: h // 2, int(w * 0.55):] = True
    elif name == "Ring corridor" and h >= 10 and w >= 10:
        walls[4:h - 4, 4:w - 4] = True
    return walls


def is_free(walls, x, y):
    h, w = walls.shape
    return 0 <= x < w and 0 <= y < h and not walls[y, x]


def ray(walls, x, y, heading, max_steps=200):
    """Distance (in cells) from the robot centre to the first wall along a heading."""
    dx, dy = DIRS[heading % 8]
    k = 0
    while k < max_steps and is_free(walls, x + (k + 1) * dx, y + (k + 1) * dy):
        k += 1
    return (k + 0.5) * float(np.hypot(dx, dy))


def sense(walls, x, y, heading, cell_m):
    """Simulated front/left/right/back distances, in cells and in metres."""
    cells = {s: ray(walls, x, y, heading + off) for s, off in SENSOR_OFFSETS.items()}
    return cells, {s: v * cell_m for s, v in cells.items()}


def action_to_turn(name):
    """Best-effort guess of the turn encoded in an action label (user can override)."""
    n = name.lower()
    mag = 2 if "sharp" in n else 1 if "slight" in n else 0
    if mag and "right" in n:
        return mag
    if mag and "left" in n:
        return -mag
    return 0


def find_state(uniq, bins):
    """Exact state if it was seen in the data, else the closest observed bin-combination."""
    return int(np.abs(uniq - bins).sum(axis=1).argmin())


def rollout(walls, start, heading, n_steps, cell_m, sensor_map, feats, edges, uniq,
            medians, values, best_actions, turns):
    """Run the greedy policy in the grid and record every step.

    sensor_map: {"front": feature_name or None, ...}; features that are not fed by a
    simulated sensor are held at their dataset median.
    turns: {action_name: turn in 45-degree steps}
    """
    x, y = start
    hd = heading
    steps = []
    for t in range(n_steps + 1):
        cells, meters = sense(walls, x, y, hd, cell_m)
        fv = dict(medians)
        for sname, feat in sensor_map.items():
            if feat:
                fv[feat] = meters[sname]
        bins = np.array([int(np.searchsorted(edges[f], fv[f], side="right")) for f in feats])
        s = find_state(uniq, bins)
        action = best_actions[s]
        rec = dict(step=t, x=x, y=y, heading=hd, cells=cells, meters=meters, state=s,
                   value=float(values[s]), action=action, collided=False)
        steps.append(rec)
        if t == n_steps:
            break
        nh = (hd + turns.get(action, 0)) % 8
        dx, dy = DIRS[nh]
        if is_free(walls, x + dx, y + dy):
            x, y = x + dx, y + dy
        else:
            rec["collided"] = True          # bumped into a wall: turn in place, no move
        hd = nh
    return steps
