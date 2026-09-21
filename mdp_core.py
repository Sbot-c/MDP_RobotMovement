"""Core MDP / Value Iteration logic (no Streamlit here, so it is easy to test)."""
import numpy as np
import pandas as pd


def compute_edges(x, n_bins, method="quantile"):
    """Bin edges (n_bins-1 inner cut points) for one feature."""
    if method == "quantile":
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    else:
        edges = np.linspace(x.min(), x.max(), n_bins + 1)[1:-1]
    return np.unique(edges)


def discretize(df, feats, edges):
    """Return an (n_rows, n_feats) array of bin indices."""
    return np.column_stack(
        [np.searchsorted(edges[f], df[f].to_numpy(dtype=float), side="right") for f in feats]
    )


def encode_states(bins):
    """Map each unique bin-combination to a state id 0..S-1 (observed states only)."""
    uniq, inv = np.unique(bins, axis=0, return_inverse=True)
    return uniq, inv.ravel()


def estimate_mdp(sidx, aidx, n_states, n_actions, group=None, alpha=0.0):
    """Estimate P[a, s, s'] by counting consecutive rows (t -> t+1).

    The action taken at row t is the label of row t.
    Returns P and n_sa (how often each (a, s) pair was observed).
    """
    s, a, s2 = sidx[:-1], aidx[:-1], sidx[1:]
    valid = np.ones(len(s), dtype=bool) if group is None else (group[:-1] == group[1:])
    counts = np.zeros((n_actions, n_states, n_states))
    np.add.at(counts, (a[valid], s[valid], s2[valid]), 1)
    n_sa = counts.sum(axis=2)
    counts += alpha                                   # optional Laplace smoothing
    tot = counts.sum(axis=2, keepdims=True)
    P = np.divide(counts, tot, out=np.zeros_like(counts), where=tot > 0)
    ai, si = np.where(tot[..., 0] == 0)               # never-seen (a, s): self-loop
    P[ai, si, si] = 1.0
    return P, n_sa


def value_iteration(P, R, gamma=0.9, tol=1e-6, max_iter=1000, mask=None):
    """Standard value iteration. P: (A,S,S), R: (A,S). Returns V, policy, deltas, Q."""
    n_a, n_s, _ = P.shape
    V = np.zeros(n_s)
    deltas = []
    for _ in range(max_iter):
        Q = R + gamma * (P @ V)                       # (A, S)
        if mask is not None:
            Q = np.where(mask, Q, -np.inf)
        V_new = Q.max(axis=0)
        delta = float(np.max(np.abs(V_new - V)))
        deltas.append(delta)
        V = V_new
        if delta < tol:
            break
    Q = R + gamma * (P @ V)
    if mask is not None:
        Q = np.where(mask, Q, -np.inf)
    return V, Q.argmax(axis=0), deltas, Q


def wall_following_reward(means, front_col, side_col, target, collision_dist):
    """State reward: closer to the target wall distance is better; too close = collision."""
    side, front = means[side_col], means[front_col]
    r = (1 - (side - target).abs() / max(target, 1e-9)).clip(-1, 1)
    r[(front < collision_dist) | (side < collision_dist)] = -5.0
    return r.to_numpy()
