"""Turning sensor readings into an MDP: states, rewards and transition probabilities."""
import numpy as np
import pandas as pd

from .room import ACTIONS, COLS, SENSORS

NAMES = {
    "front": ["near", "mid", "far"],
    "left": ["too close", "close", "on track", "far", "lost"],
    "right": ["near", "clear"],
    "back": ["near", "clear"],
}
NB = [len(NAMES[s]) for s in SENSORS]
NS = int(np.prod(NB))          # 3 x 5 x 2 x 2 = 60 states
NA = len(ACTIONS)
MIN_SEEN = 2                   # fewer samples than this and a state-action pair counts as unseen

DEFAULT_REWARDS = dict(step=-1.0, onTrack=0.8, tooClose=-2.0, blocked=-2.0, lost=-1.0,
                       unseen=-3.0, bump=-5.0, exit=100.0)


class Bands:
    """Distance band edges (metres) for each sensor."""

    def __init__(self, front_near=0.9, track_lo=0.5, track_hi=0.9):
        self.front_near, self.track_lo, self.track_hi = front_near, track_lo, track_hi
        self.edges = {
            "front": [front_near, front_near + 0.7],
            "left": [max(0.1, track_lo - 0.2), track_lo, track_hi, max(track_hi + 0.3, 2.5)],
            "right": [0.6],
            "back": [0.8],
        }

    def key(self):
        return (self.front_near, self.track_lo, self.track_hi)

    def describe(self, sensor):
        ed, nm = self.edges[sensor], NAMES[sensor]
        out = []
        for i, n in enumerate(nm):
            if i == 0:
                out.append(f"{n} (<{ed[0]:.2f})")
            elif i == len(ed):
                out.append(f"{n} (≥{ed[i-1]:.2f})")
            else:
                out.append(f"{n} ({ed[i-1]:.2f}–{ed[i]:.2f})")
        return ", ".join(out)

    def state_of(self, sv):
        s = 0
        for k, name in enumerate(SENSORS):
            b = 0
            for e in self.edges[name]:
                if sv[k] >= e:
                    b += 1
            s = s * NB[k] + b
        return s

    def states_of(self, X):
        """Vectorised state_of for an (n, 4) array."""
        s = np.zeros(len(X), dtype=int)
        for k, name in enumerate(SENSORS):
            b = np.searchsorted(np.array(self.edges[name]), X[:, k], side="right")
            s = s * NB[k] + b
        return s


def decode(s):
    b = [0, 0, 0, 0]
    for k in range(3, -1, -1):
        b[k] = s % NB[k]
        s //= NB[k]
    return b


def state_name(s):
    b = decode(s)
    return [NAMES[n][b[k]] for k, n in enumerate(SENSORS)]


def state_label(s):
    n = state_name(s)
    return f"#{s}  front {n[0]}, left {n[1]}, right {n[2]}, back {n[3]}"


def state_reward(s, W):
    """Reward for arriving in state s."""
    b = decode(s)
    f, l = b[0], b[1]
    r = W["step"]
    if f == 0:
        return r + W["blocked"]
    if l == 0:
        return r + W["tooClose"]
    if l == 2:
        return r + W["onTrack"]
    if l >= 4:
        return r + W["lost"]
    if l == 3:
        return r + W["lost"] * 0.5
    return r


def load_dataset(path):
    """Read sensor_readings_4.csv (with or without a header row)."""
    first = open(path, encoding="utf-8").readline()
    has_header = not first.split(",")[0].strip().replace(".", "", 1).isdigit()
    df = pd.read_csv(path, header=0 if has_header else None)
    if not has_header:
        df.columns = COLS + ["Class"]
    df.columns = [c.strip() for c in df.columns]
    df["Class"] = df["Class"].astype(str).str.strip()
    df = df[df["Class"].isin(ACTIONS)].reset_index(drop=True)
    df["action"] = df["Class"].map({a: i for i, a in enumerate(ACTIONS)})
    return df


def estimate(df, bands):
    """Count transitions between consecutive rows (the data is one 9 Hz time series)."""
    X = df[COLS].to_numpy(dtype=float)
    a = df["action"].to_numpy(dtype=int)
    st = bands.states_of(X)
    visits = np.bincount(st, minlength=NS).astype(float)
    cnt = np.zeros((NS, NA, NS))
    np.add.at(cnt, (st[:-1], a[:-1], st[1:]), 1)
    tot = cnt.sum(axis=2)
    return dict(cnt=cnt, tot=tot, visits=visits, states=st)


def build_model(est, W):
    """P[s, a, s'] and expected reward R[s, a]. Unseen pairs stay put and get W['unseen']."""
    r = np.array([state_reward(s, W) for s in range(NS)])
    P = np.zeros((NS, NA, NS))
    R = np.zeros((NS, NA))
    seen = est["tot"] >= MIN_SEEN
    for s in range(NS):
        for a in range(NA):
            if seen[s, a]:
                P[s, a] = est["cnt"][s, a] / est["tot"][s, a]
                R[s, a] = P[s, a] @ r
            else:
                P[s, a, s] = 1.0
                R[s, a] = W["unseen"]
    return dict(P=P, R=R, seen=seen, r=r)
