"""Value iteration (MDP), fitted value iteration (ADP), Hooke-Jeeves, and room rollouts."""
import math
import numpy as np

from . import room
from .model import NS, NA, NB, decode, state_reward


def q_values(M, V, gamma):
    return M["R"] + gamma * (M["P"] @ V)


def greedy(M, V, gamma):
    return np.argmax(q_values(M, V, gamma), axis=1)


# 1. Value iteration ----------------------------------------------------------
def value_iteration(M, gamma, tol, max_iter=5000, record=False):
    V = np.zeros(NS)
    hist, Vs = [], [V.copy()] if record else None
    it, delta = 0, math.inf
    while it < max_iter:
        V2 = q_values(M, V, gamma).max(axis=1)
        delta = float(np.max(np.abs(V2 - V)))
        V = V2
        it += 1
        hist.append(delta)
        if record:
            Vs.append(V.copy())
        if delta < tol:
            break
    return dict(V=V, pi=greedy(M, V, gamma), iters=it, delta=delta, hist=hist,
                converged=delta < tol, Vs=Vs)


# 2. Approximate DP: fitted value iteration with a linear value function ------
NF = 1 + sum(NB)


def features():
    Phi = np.zeros((NS, NF))
    for s in range(NS):
        Phi[s, 0] = 1
        o = 1
        for k, b in enumerate(decode(s)):
            Phi[s, o + b] = 1
            o += NB[k]
    return Phi


def adp(M, est, gamma, tol, max_iter=1000, ridge=1e-3):
    Phi = features()
    w8 = est["visits"] + 0.5
    A = Phi.T @ (Phi * w8[:, None]) + ridge * np.eye(NF)
    Vh = np.zeros(NS)
    w = np.zeros(NF)
    hist, it, delta = [], 0, math.inf
    while it < max_iter:
        y = q_values(M, Vh, gamma).max(axis=1)      # Bellman backup
        w = np.linalg.solve(A, Phi.T @ (w8 * y))    # weighted least-squares fit
        V2 = Phi @ w
        delta = float(np.max(np.abs(V2 - Vh)))
        Vh = V2
        it += 1
        hist.append(delta)
        if delta < tol:
            break
    return dict(V=Vh, w=w, pi=greedy(M, Vh, gamma), iters=it, delta=delta, hist=hist,
                converged=delta < tol, nf=NF)


# Policies ------------------------------------------------------------------
def table_policy(pi, bands):
    return lambda sv: int(pi[bands.state_of(sv)])


def rule(sv, th):
    if sv[0] < th[0]:
        return 2          # Sharp-Right-Turn
    if sv[1] < th[1]:
        return 1          # Slight-Right-Turn
    if sv[1] > th[2]:
        return 3          # Slight-Left-Turn
    return 0              # Move-Forward


def rule_policy(th):
    return lambda sv: rule(sv, th)


# Rollouts in the room ------------------------------------------------------
def rollout(policy, W, bands, seed=2026, max_t=600, light=False):
    rng = room.Rng(seed)
    p = dict(room.START)
    path = [dict(x=p["x"], y=p["y"], h=p["h"])]
    acts, sens, bumps, cum, cum_good = [], [], [], [0.0], [0]
    ret, good, exited = 0.0, 0, False
    for t in range(max_t):
        sv = room.sense(p["x"], p["y"], p["h"], 0.02, rng)
        sens.append(sv)
        a = policy(sv)
        acts.append(a)
        p = room.step(p, a, rng, 1.5)
        if p["bump"]:
            bumps.append(t + 1)
            ret += W["bump"]
        s2 = bands.state_of(room.sense(p["x"], p["y"], p["h"]))
        ret += state_reward(s2, W)
        if decode(s2)[1] == 2:
            good += 1
        path.append(dict(x=p["x"], y=p["y"], h=p["h"]))
        cum.append(ret)
        cum_good.append(good)
        if room.at_exit(p):
            exited = True
            ret += W["exit"]
            break
    n = len(acts)
    if not light:
        sens.append(room.sense(p["x"], p["y"], p["h"]))
    return dict(path=path, acts=acts, sens=sens, bumps=bumps, ret=ret, exit=exited, steps=n,
                wall=100 * good / n if n else 0, cum=cum, cum_good=cum_good)


# 3. Hooke-Jeeves pattern search over the controller's three thresholds -----
LO, HI = [0.4, 0.2, 0.6], [3.0, 2.0, 4.0]


def hooke_jeeves(W, bands, x0=(0.8, 0.4, 1.0), step=0.25, min_step=0.01, max_eval=400):
    evals, hist, cache = 0, [], {}

    def clamp(v):
        return [max(LO[i], min(HI[i], math.floor(z * 1000 + 0.5) / 1000)) for i, z in enumerate(v)]

    def f(v):
        nonlocal evals
        v = clamp(v)
        key = tuple(v)
        if key in cache:
            return cache[key]
        evals += 1
        val = rollout(rule_policy(v), W, bands, max_t=500, light=True)["ret"]
        cache[key] = val
        hist.append(val)
        return val

    def explore(base, fb, h):
        xb, best = list(base), fb
        for i in range(len(xb)):
            t = list(xb); t[i] += h
            ft = f(t)
            if ft > best:
                xb, best = clamp(t), ft
                continue
            t = list(xb); t[i] -= h
            ft = f(t)
            if ft > best:
                xb, best = clamp(t), ft
        return xb, best

    x = clamp(list(x0))
    fx = f(x)
    h = step
    moves = 0
    while h >= min_step and evals < max_eval:
        ex, ef = explore(x, fx, h)
        if ef > fx:
            while evals < max_eval:                  # pattern moves while they keep improving
                xp = clamp([2 * z - x[i] for i, z in enumerate(ex)])
                x, fx = ex, ef
                moves += 1
                ex2, ef2 = explore(xp, f(xp), h)
                if ef2 > fx:
                    ex, ef = ex2, ef2
                else:
                    break
        else:
            h /= 2
    best, b = [], -math.inf
    for v in hist:
        b = max(b, v)
        best.append(b)
    return dict(theta=x, f=fx, evals=evals, hist=hist, best=best, moves=moves)


def label_match(df, policy):
    from .room import COLS
    X = df[COLS].to_numpy(dtype=float)
    a = df["action"].to_numpy()
    ok = sum(1 for i in range(len(X)) if policy(list(X[i])) == a[i])
    return 100 * ok / len(X) if len(X) else 0.0
