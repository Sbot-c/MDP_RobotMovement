"""The simulated room, the robot's motion and its four simplified distance sensors.

Map units: 50 units = 1 metre. Angles in degrees, measured clockwise on screen
(y points down), so +90 is the robot's right-hand side.
"""
import math
import numpy as np

POLY = [(0, 0), (190, 0), (245, 70), (388, 70), (400, 0), (570, 0), (610, 90), (750, 95),
        (752, 430), (785, 440), (785, 520), (560, 520), (555, 478), (285, 478), (280, 515), (0, 515)]
BLOCK = dict(x=300, y=230, w=170, h=90)
U = 50                                  # map units per metre
START = dict(x=40.0, y=150.0, h=-90.0)  # entrance, facing north, wall on the left
EXIT = dict(x=640, y=430, w=145, h=90)  # exit bay in front of the gate
RADIUS = 16                             # robot body radius (units)
MAXR = 5.0                              # sensor range (m)

ACTIONS = ["Move-Forward", "Slight-Right-Turn", "Sharp-Right-Turn", "Slight-Left-Turn"]
ACT_SHORT = ["FWD", "SLT-R", "SHP-R", "SLT-L"]
MOVES = [dict(turn=0, fwd=11), dict(turn=10, fwd=10), dict(turn=30, fwd=4), dict(turn=-10, fwd=10)]
SENSORS = ["front", "left", "right", "back"]
COLS = ["SD_front", "SD_left", "SD_right", "SD_back"]


def _edges():
    e = []
    def add(p):
        for i in range(len(p)):
            e.append((p[i], p[(i + 1) % len(p)]))
    add(POLY)
    b = BLOCK
    add([(b["x"], b["y"]), (b["x"] + b["w"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (b["x"], b["y"] + b["h"])])
    return e


EDGES = _edges()
EA = np.array([a for a, _ in EDGES], dtype=float)   # edge start points
EB = np.array([b for _, b in EDGES], dtype=float)   # edge end points
EX = EB[:, 0] - EA[:, 0]
EY = EB[:, 1] - EA[:, 1]
EL2 = EX * EX + EY * EY

# 7 rays spread over each 60 degree arc: the dataset's "minimum reading in a 60 degree arc"
_OFF = np.array([0, -90, 90, 180], dtype=float)
_SPREAD = np.arange(-30, 31, 10, dtype=float)
_RAY_OFFSETS = (_OFF[:, None] + _SPREAD[None, :]).ravel()   # 28 rays, sensor-major


class Rng:
    """Small seeded xorshift generator so runs are repeatable."""

    def __init__(self, seed):
        s = seed & 0xFFFFFFFF
        self.s = s or 1

    def __call__(self):
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        self.s = s & 0xFFFFFFFF
        return self.s / 4294967296.0

    def gauss(self):
        u = self() or 1e-9
        v = self()
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)


def sense(px, py, deg, noise=0.0, rng=None):
    """Return [front, left, right, back] distances in metres."""
    ang = np.radians(deg + _RAY_OFFSETS)
    dx = np.cos(ang)[:, None]
    dy = np.sin(ang)[:, None]
    den = dx * EY[None, :] - dy * EX[None, :]
    ax = EA[:, 0][None, :] - px
    ay = EA[:, 1][None, :] - py
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ax * EY[None, :] - ay * EX[None, :]) / den
        u = (ax * dy - ay * dx) / den
    ok = (np.abs(den) >= 1e-9) & (t > 0.001) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, np.inf)
    per_ray = t.min(axis=1).reshape(4, len(_SPREAD))
    per_sensor = per_ray.min(axis=1)
    out = []
    for m in per_sensor:
        v = min(MAXR, (m if np.isfinite(m) else MAXR * U) / U)
        if noise and rng is not None:
            v = max(0.05, v + noise * rng.gauss())
        out.append(float(v))
    return out


def wall_dist(px, py):
    t = np.where(EL2 > 0, ((px - EA[:, 0]) * EX + (py - EA[:, 1]) * EY) / np.where(EL2 > 0, EL2, 1), 0)
    t = np.clip(t, 0, 1)
    return float(np.min(np.hypot(px - (EA[:, 0] + t * EX), py - (EA[:, 1] + t * EY))))


def inside(px, py):
    c = False
    n = len(POLY)
    j = n - 1
    for i in range(n):
        a, b = POLY[i], POLY[j]
        if (a[1] > py) != (b[1] > py) and px < (b[0] - a[0]) * (py - a[1]) / (b[1] - a[1]) + a[0]:
            c = not c
        j = i
    bl = BLOCK
    if bl["x"] < px < bl["x"] + bl["w"] and bl["y"] < py < bl["y"] + bl["h"]:
        return False
    return c


def step(pose, a, rng=None, head_noise=0.0):
    """Apply one action. A move into a wall is cancelled and counted as a bump."""
    mv = MOVES[a]
    h = pose["h"] + mv["turn"] + (head_noise * rng.gauss() if (rng is not None and head_noise) else 0)
    rad = math.radians(h)
    nx = pose["x"] + math.cos(rad) * mv["fwd"]
    ny = pose["y"] + math.sin(rad) * mv["fwd"]
    bump = False
    if not inside(nx, ny) or wall_dist(nx, ny) < RADIUS:
        bump = True
        nx, ny = pose["x"], pose["y"]
    return dict(x=nx, y=ny, h=((h + 540) % 360) - 180, bump=bump)


def at_exit(p):
    return p["x"] >= EXIT["x"] and p["y"] >= EXIT["y"]
