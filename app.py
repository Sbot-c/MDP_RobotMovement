"""Streamlit app: MDP + Value Iteration on a Kaggle sensor dataset (wall-following robot)."""
import glob
import io
import os

import numpy as np
import pandas as pd
import streamlit as st

from mdp_core import (compute_edges, discretize, encode_states, estimate_mdp,
                      value_iteration, wall_following_reward)

st.set_page_config(page_title="MDP Value Iteration", layout="wide")
st.title("MDPs – Wall-following robot navigation")
st.caption("Build an MDP from a sensor dataset and solve it with Value Iteration.")

DEFAULT_NAMES = {
    5: ["SD_front", "SD_left", "SD_right", "SD_back", "Class"],   # sensor_readings_4.csv
    3: ["SD_front", "SD_left", "Class"],                           # sensor_readings_2.csv
    25: [f"US{i}" for i in range(1, 25)] + ["Class"],              # sensor_readings_24.csv
}


def pick(name, cols):
    return cols.index(name) if name in cols else 0


@st.cache_data(show_spinner=False)
def read_table(src):
    buf = lambda: io.BytesIO(src) if isinstance(src, bytes) else src
    first = pd.read_csv(buf(), header=None, nrows=1)
    has_header = not pd.api.types.is_numeric_dtype(first.iloc[:, 0])   # auto-detect header
    df = pd.read_csv(buf(), header=0 if has_header else None)
    if not has_header:
        df.columns = DEFAULT_NAMES.get(df.shape[1], [f"col_{i}" for i in range(df.shape[1])])
    return df


@st.cache_data(show_spinner="Downloading from Kaggle...")
def kaggle_download(slug):
    import kagglehub                       # pip install kagglehub
    return kagglehub.dataset_download(slug)


# ------------------------------------------------------------------ 1. data
src = None
with st.sidebar:
    st.header("1. Data")
    source = st.radio("Source", ["Upload CSV", "Local path", "Kaggle (kagglehub)"])
    if source == "Upload CSV":
        up = st.file_uploader("CSV file", type=["csv", "data"])
        src = up.getvalue() if up else None
    elif source == "Local path":
        p = st.text_input("Path to a CSV file", "sensor_readings_4.csv")
        src = p if os.path.isfile(p) else None
    else:
        slug = st.text_input("Kaggle dataset slug", "uciml/wall-following-robot")
        if st.button("Download"):
            try:
                st.session_state["kpath"] = kaggle_download(slug)
            except Exception as e:
                st.error(f"Download failed: {e}")
        if st.session_state.get("kpath"):
            files = sorted(glob.glob(os.path.join(st.session_state["kpath"], "**", "*.csv"), recursive=True))
            src = st.selectbox("File", files) if files else None

if src is None:
    st.info("Choose a dataset in the sidebar to begin.")
    st.stop()

df = read_table(src)
num_cols = df.select_dtypes("number").columns.tolist()
st.subheader("Dataset preview")
st.write(f"{df.shape[0]} rows x {df.shape[1]} columns")
st.dataframe(df.head())

# ------------------------------------------------------- 2. states & actions
with st.sidebar:
    st.header("2. States and actions")
    cols = df.columns.tolist()
    action_col = st.selectbox("Action / label column", cols, index=len(cols) - 1)
    cand = [c for c in num_cols if c != action_col]
    feats = st.multiselect("State features (sensors)", cand, default=cand[:4])
    n_bins = st.slider("Bins per feature", 2, 6, 3)
    method = st.radio("Binning", ["quantile", "uniform"], horizontal=True)
    group_col = st.selectbox("Episode / run column (optional)", ["(none)"] + cols)
    restrict = st.checkbox("Only allow actions seen in the data for each state", True)

if not feats:
    st.warning("Pick at least one state feature.")
    st.stop()
if n_bins ** len(feats) > 5000:
    st.warning(f"Up to {n_bins ** len(feats):,} states - reduce features or bins.")

actions = sorted(df[action_col].astype(str).unique())

# ------------------------------------------------------------ 3. reward
with st.sidebar:
    st.header("3. Reward")
    mode = st.radio("Reward source", ["Distance-based (wall following)", "Column in dataset"])
    if mode.startswith("Distance"):
        front_col = st.selectbox("Front distance column", num_cols, index=pick("SD_front", num_cols))
        side_col = st.selectbox("Wall-side distance column", num_cols, index=pick("SD_left", num_cols))
        target = st.number_input("Target wall distance", value=float(df[side_col].median()),
                                 key=f"target_{side_col}")
        coll = st.number_input("Collision distance (penalty below this)",
                               value=float(df[front_col].quantile(0.05)), key=f"coll_{front_col}")
    else:
        reward_col = st.selectbox("Reward column", num_cols)
    with st.expander("Action costs (smoothness)"):
        costs = np.array([st.number_input(a, value=0.2 if "sharp" in a.lower() else 0.05 if "slight" in a.lower() else 0.0,
                                          key=f"cost_{a}") for a in actions])

# ---------------------------------------------------------- 4. algorithm
st.sidebar.header("4. Value Iteration")
gamma = st.sidebar.slider("Discount factor (gamma)", 0.0, 0.999, 0.9)
tol = st.sidebar.number_input("Convergence threshold (tolerance)", value=1e-6, format="%.1e")
max_iter = st.sidebar.number_input("Max iterations", value=1000, step=100)

if st.sidebar.button("Run Value Iteration", type="primary"):
    edges = {f: compute_edges(df[f].to_numpy(dtype=float), n_bins, method) for f in feats}
    uniq, sidx = encode_states(discretize(df, feats, edges))
    aidx = df[action_col].astype(str).map({a: i for i, a in enumerate(actions)}).to_numpy()
    group = None if group_col == "(none)" else df[group_col].to_numpy()

    P, n_sa = estimate_mdp(sidx, aidx, len(uniq), len(actions), group)
    mask = n_sa > 0 if restrict else np.ones_like(n_sa, dtype=bool)
    mask[:, ~mask.any(axis=0)] = True                      # states with no observed action

    means = df.groupby(sidx)[num_cols].mean()
    r_state = (wall_following_reward(means, front_col, side_col, target, coll)
               if mode.startswith("Distance") else means[reward_col].to_numpy())
    R = P @ r_state - costs[:, None]                       # R[a, s]

    V, policy, deltas, _ = value_iteration(P, R, gamma, tol, int(max_iter), mask)

    out = pd.DataFrame(uniq, columns=[f"{f}_bin" for f in feats])
    out.insert(0, "state_id", range(len(uniq)))
    out["optimal_value"] = V
    out["best_action"] = [actions[i] for i in policy]
    out["samples"] = np.bincount(sidx, minlength=len(uniq))
    out.to_csv("optimal_value_function.csv", index=False)  # required output file
    st.session_state["res"] = dict(out=out, deltas=deltas, edges=edges, uniq=uniq, feats=feats,
                                   agree=float((policy[sidx] == aidx).mean()), tol=tol, actions=actions)

# ------------------------------------------------------------ results
res = st.session_state.get("res")
if res:
    out = res["out"]
    st.subheader("Results")
    c = st.columns(4)
    c[0].metric("States", len(out))
    c[1].metric("Iterations", len(res["deltas"]))
    c[2].metric("Final delta", f"{res['deltas'][-1]:.2e}", "converged" if res["deltas"][-1] < res["tol"] else "not converged")
    c[3].metric("Policy = dataset label", f"{res['agree']:.1%}")

    left, right = st.columns(2)
    left.markdown("**Convergence (max change in V per sweep)**")
    left.line_chart(pd.Series(res["deltas"], name="delta"))
    right.markdown("**Optimal action per state**")
    right.bar_chart(out["best_action"].value_counts())

    st.dataframe(out)
    st.download_button("Download optimal_value_function.csv", out.to_csv(index=False),
                       "optimal_value_function.csv", "text/csv")

    st.subheader("Try a sensor reading")
    vals, sl = {}, st.columns(len(res["feats"]))
    for col, f in zip(sl, res["feats"]):
        lo, hi = float(df[f].min()), float(df[f].max())
        vals[f] = col.slider(f, lo, hi, float(df[f].median()))
    b = [int(np.searchsorted(res["edges"][f], vals[f], side="right")) for f in res["feats"]]
    hit = np.where((res["uniq"] == b).all(axis=1))[0]
    if len(hit):
        row = out.iloc[hit[0]]
        st.success(f"State {int(row.state_id)}: value = {row.optimal_value:.3f}, best action = **{row.best_action}**")
    else:
        st.warning("This combination of bins never appears in the dataset (unseen state).")
