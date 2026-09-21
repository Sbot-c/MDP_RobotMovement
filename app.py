"""Wall-following robot navigation: MDP value iteration, ADP and Hooke-Jeeves.

Run locally with:  streamlit run app.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from mdp import model, room, solvers
from mdp.arena import METHODS, arena_html

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "sensor_readings_4.csv"
OUT_CSV = ROOT / "optimal_value_function.csv"
COLORS = {"mdp": "#2b64ff", "hj": "#e07800", "adp": "#d42a90"}
ICON = ["↑", "↗", "→", "↖"]
PLOT = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
            font=dict(family="Bricolage Grotesque, sans-serif", color="#241a4b"),
            margin=dict(l=10, r=10, t=10, b=10))

st.set_page_config(page_title="Wall-following robot: MDP dashboard", page_icon="🤖", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600&family=Press+Start+2P&display=swap');
:root{
  --ink:#f1ecff; --deep:#140f28; --panel:#1c1440; --card:#241a4b;
  --mdp:#2b64ff; --hj:#e07800; --adp:#d42a90; --glow:#8b7bff;
}
.stMarkdown p, .stMarkdown li, label p, .stCaption {font-family:"Bricolage Grotesque", sans-serif;}
h1, h2, h3 {font-family:"Press Start 2P", monospace !important; font-weight:400 !important; letter-spacing:0; color:var(--ink) !important;}
h1 {font-size:1.45rem !important; line-height:1.5 !important; text-shadow:3px 3px 0 rgba(0,0,0,.55), 0 0 18px rgba(139,123,255,.55);}
h2 {font-size:1rem !important; line-height:1.6 !important;}
h3 {font-size:.8rem !important; line-height:1.6 !important; color:var(--glow) !important;}

/* --- arcade cabinet backdrop: scanlines + neon glow blobs behind everything --- */
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {background:transparent;}
[data-testid="stAppViewContainer"] {
  background:
    repeating-linear-gradient(0deg, rgba(255,255,255,.02) 0px, rgba(255,255,255,.02) 1px, transparent 1px, transparent 3px),
    radial-gradient(1100px 550px at 12% -8%, rgba(139,123,255,.18), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(212,42,144,.14), transparent 55%),
    radial-gradient(700px 500px at 50% 100%, rgba(43,100,255,.12), transparent 55%),
    var(--deep);
}

/* --- sidebar: lit-up control panel --- */
section[data-testid="stSidebar"] {
  background:linear-gradient(180deg, var(--panel), var(--deep));
  border-right:3px solid var(--glow); box-shadow:6px 0 22px rgba(139,123,255,.15);
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
  border-bottom:2px dashed rgba(139,123,255,.4); padding-bottom:8px;
}

/* --- metric "readout" cards, glowing like the HUD in the arena --- */
div[data-testid="stMetric"] {
  border:3px solid var(--glow); border-radius:6px; background:#fff; padding:10px 14px;
  box-shadow:5px 5px 0 rgba(0,0,0,.5), 0 0 16px rgba(139,123,255,.35);
}
div[data-testid="stMetricValue"] {font-family:"Press Start 2P", monospace; font-size:1.05rem; color:#241a4b !important;}
div[data-testid="stMetricLabel"] {color:#3b2d8f !important;}

/* --- tabs: arcade cartridge slots --- */
.stTabs [data-baseweb="tab-list"] {
  gap:6px; background:var(--panel); padding:8px; border-radius:8px; border:3px solid var(--card);
}
.stTabs [data-baseweb="tab"] {
  font-family:"Press Start 2P", monospace; font-size:.6rem; color:var(--ink);
  background:rgba(255,255,255,.04); border:2px solid rgba(139,123,255,.35); border-radius:4px; padding:10px 12px;
}
.stTabs [aria-selected="true"] {
  background:var(--glow) !important; color:#140f28 !important; border-color:var(--glow) !important;
  box-shadow:3px 3px 0 rgba(0,0,0,.5);
}

/* --- plot / dataframe panels: framed like little screens on the cabinet --- */
div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
  border:3px solid rgba(139,123,255,.55); border-radius:6px; padding:6px;
  background:rgba(255,255,255,.02); box-shadow:4px 4px 0 rgba(0,0,0,.4);
}

/* --- expanders --- */
div[data-testid="stExpander"] {
  border:2px solid rgba(139,123,255,.4) !important; border-radius:6px; background:rgba(255,255,255,.03);
}
div[data-testid="stExpander"] summary {font-family:"Press Start 2P", monospace; font-size:.62rem; color:var(--glow);}

/* --- buttons: chunky arcade buttons --- */
.stDownloadButton button, .stButton button {
  font-family:"Press Start 2P", monospace; font-size:.6rem; border:2px solid var(--glow) !important;
  border-radius:5px; box-shadow:3px 3px 0 rgba(0,0,0,.5); transition:transform .08s ease;
}
.stDownloadButton button:hover, .stButton button:hover {transform:translate(-1px,-1px); box-shadow:4px 4px 0 rgba(0,0,0,.5);}

/* --- inputs get a faint neon edge to match the cabinet --- */
div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input, .stSlider {
  border-color:rgba(139,123,255,.35) !important;
}

::-webkit-scrollbar {height:10px; width:10px;}
::-webkit-scrollbar-track {background:var(--deep);}
::-webkit-scrollbar-thumb {background:var(--glow); border-radius:5px;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data + training
@st.cache_data
def get_data():
    return model.load_dataset(DATA)


@st.cache_data(show_spinner=False)
def train_model_based(band_key, rewards, gamma, tol):
    bands = model.Bands(*band_key)
    df = get_data()
    W = dict(rewards)
    t0 = time.perf_counter()
    est = model.estimate(df, bands)
    M = model.build_model(est, W)
    t_model = time.perf_counter() - t0
    t1 = time.perf_counter()
    vi = solvers.value_iteration(M, gamma, tol, record=True)
    t_vi = time.perf_counter() - t1
    t2 = time.perf_counter()
    ad = solvers.adp(M, est, gamma, tol)
    t_adp = time.perf_counter() - t2
    ref = solvers.value_iteration(M, gamma, 1e-10, max_iter=20000)
    visited = [s for s in range(model.NS) if est["visits"][s] > 0]
    acc = []
    for t in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        r = solvers.value_iteration(M, gamma, t, max_iter=20000)
        acc.append(dict(tol=t, iters=r["iters"], delta=r["delta"], bound=gamma * r["delta"] / (1 - gamma),
                        err=float(np.max(np.abs(r["V"] - ref["V"]))),
                        diff=int(sum(r["pi"][s] != ref["pi"][s] for s in visited))))
    runs, match = {}, {}
    for key, res in (("mdp", vi), ("adp", ad)):
        pol = solvers.table_policy(res["pi"], bands)
        runs[key] = solvers.rollout(pol, W, bands)
        match[key] = solvers.label_match(df, pol)
    return dict(est=est, M=M, vi=vi, adp=ad, acc=acc, visited=visited, runs=runs, match=match,
                time=dict(mdp=t_model + t_vi, adp=t_model + t_adp))


@st.cache_data(show_spinner=False)
def train_hj(band_key, rewards, x0, step):
    bands = model.Bands(*band_key)
    W = dict(rewards)
    t0 = time.perf_counter()
    hj = solvers.hooke_jeeves(W, bands, x0=x0, step=step)
    t = time.perf_counter() - t0
    pol = solvers.rule_policy(hj["theta"])
    return dict(hj=hj, run=solvers.rollout(pol, W, bands), match=solvers.label_match(get_data(), pol), time=t)


df = get_data()

# ---------------------------------------------------------------- sidebar: inputs
with st.sidebar:
    st.header("Input")
    st.caption("Change anything here and all three methods retrain.")
    gamma = st.slider("Discount factor (gamma)", 0.50, 0.99, 0.90, 0.01)
    tol = st.select_slider("Convergence threshold (tolerance)", options=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                           value=1e-4, format_func=lambda v: f"{v:g}")
    st.subheader("States")
    front = st.number_input("Front counts as blocked below (m)", 0.4, 2.0, 0.9, 0.05)
    c1, c2 = st.columns(2)
    lo = c1.number_input("On-track band from (m)", 0.3, 2.0, 0.5, 0.05)
    hi = c2.number_input("to (m)", 0.5, 2.4, 0.9, 0.05)
    if hi <= lo + 0.1:
        st.error("The on-track band must end at least 0.1 m after it starts. Using 0.5 to 0.9 m.")
        lo, hi = 0.5, 0.9
    with st.expander("Rewards"):
        st.caption("Given when the robot lands in a state. Bump and exit only apply in the room run.")
        labels = dict(step="Every step", onTrack="On-track band", tooClose="Too close to the wall",
                      blocked="Front blocked", lost="Wall lost (half this when far)",
                      unseen="Action never seen in the data", bump="Bump (room run)", exit="Reach the exit (room run)")
        W = {k: st.number_input(labels[k], value=float(v), step=0.1, key="w_" + k) for k, v in model.DEFAULT_REWARDS.items()}
    with st.expander("Hooke-Jeeves starting guess"):
        h0 = st.number_input("Sharp right if front under (m)", 0.4, 3.0, 0.8, 0.05)
        h1 = st.number_input("Drift right if left under (m)", 0.2, 2.0, 0.4, 0.05)
        h2 = st.number_input("Drift left if left over (m)", 0.6, 4.0, 1.0, 0.05)
        hstep = st.number_input("First step size (m)", 0.02, 1.0, 0.25, 0.05)

band_key = (float(front), float(lo), float(hi))
bands = model.Bands(*band_key)
rewards = tuple(sorted(W.items()))
with st.spinner("Training MDP, ADP and Hooke-Jeeves…"):
    R = train_model_based(band_key, rewards, gamma, tol)
    H = train_hj(band_key, rewards, (h0, h1, h2), hstep)
vi, ad, hj, est, M = R["vi"], R["adp"], H["hj"], R["est"], R["M"]
runs = dict(mdp=R["runs"]["mdp"], hj=H["run"], adp=R["runs"]["adp"])
match = dict(mdp=R["match"]["mdp"], hj=H["match"], adp=R["match"]["adp"])
times = dict(mdp=R["time"]["mdp"], hj=H["time"], adp=R["time"]["adp"])
visited = R["visited"]

# ---------------------------------------------------------------- hero: the arena
st.title("Wall-following robot")
st.markdown("An MDP built from the Kaggle wall-following sensor data, solved by value iteration, "
            "and raced against ADP and Hooke-Jeeves in the same room.")
components.html(arena_html(runs, bands, dict(mdp=vi["V"], adp=ad["V"]), W["exit"]), height=520, scrolling=True)
st.caption("The robot's real room is not in the dataset, so this room is a stand-in: policies learned from the "
           "real readings are tested here, at 9 steps a second like the real sensors.")

tabs = st.tabs(["Dataset", "Input", "How it works", "Output", "Accuracy", "Side by side"])

# ---------------------------------------------------------------- dataset
with tabs[0]:
    st.subheader("Dataset")
    st.markdown(f"**{len(df):,} rows** from `sensor_readings_4.csv`: a SCITOS G5 robot following the wall "
                "clockwise with the wall on its left, for 4 rounds, logged 9 times a second. "
                "Each feature is the shortest ultrasound reading across a 60° arc, in metres.")
    c1, c2 = st.columns(2)
    with c1:
        share = df["Class"].value_counts(normalize=True).reindex(room.ACTIONS) * 100
        fig = go.Figure(go.Bar(x=share.values, y=[f"{ICON[i]} {a}" for i, a in enumerate(room.ACTIONS)],
                               orientation="h", marker_color="#8b7bff",
                               text=[f"{v:.1f}%" for v in share.values], textposition="outside"))
        fig.update_layout(**PLOT, height=240, xaxis=dict(title="Share of rows (%)", range=[0, share.max() * 1.25]), yaxis=dict(autorange="reversed"))
        st.markdown("**Movement classes**")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Features**")
        desc = df[room.COLS].describe().T[["min", "mean", "50%", "max"]].rename(columns={"50%": "median"})
        st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)
    with st.expander("First rows of the file"):
        st.dataframe(df[room.COLS + ["Class"]].head(20), use_container_width=True)

# ---------------------------------------------------------------- input
with tabs[1]:
    st.subheader("Input")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**States.** Each sensor is split into bands, giving **{model.NS} states** "
                    f"({' × '.join(map(str, model.NB))}). The data visits **{len(visited)}** of them.")
        st.markdown("\n".join(f"- {s.capitalize()}: {bands.describe(s)}" for s in room.SENSORS))
        st.markdown("**Actions.** " + ", ".join(f"{ICON[i]} {a}" for i, a in enumerate(room.ACTIONS)))
    with c2:
        st.markdown(f"**Discount factor** γ = {gamma:.2f}  \n**Tolerance** = {tol:g}")
        st.markdown("**Rewards**")
        st.dataframe(pd.DataFrame({"Reward": [labels[k] for k in W], "Value": list(W.values())}),
                     hide_index=True, use_container_width=True)
    st.markdown("**Transition probabilities**")
    st.caption("Counted from consecutive rows: after this state and action, where did the robot end up next?")
    order = sorted(visited, key=lambda s: -est["visits"][s])
    c1, c2 = st.columns([3, 2])
    s_sel = c1.selectbox("From state", order, format_func=lambda s: f"{model.state_label(s)} ({int(est['visits'][s])} rows)")
    a_sel = c2.radio("Action", range(4), format_func=lambda a: f"{ICON[a]} {room.ACTIONS[a]}", horizontal=True)
    T = est["tot"][s_sel, a_sel]
    if T < model.MIN_SEEN:
        st.info(f"The robot took this action from this state {int(T)} time(s) in the data, too few to trust. "
                f"The model keeps it in place and gives the “never seen” reward ({W['unseen']}).")
    else:
        probs = est["cnt"][s_sel, a_sel] / T
        nz = np.argsort(-probs)[:8]
        nz = [s for s in nz if probs[s] > 0]
        fig = go.Figure(go.Bar(x=[probs[s] for s in nz], y=[model.state_label(s) for s in nz], orientation="h",
                               marker_color=COLORS["mdp"], text=[f"{probs[s]:.3f}" for s in nz], textposition="outside"))
        fig.update_layout(**PLOT, height=60 + 34 * len(nz), xaxis=dict(range=[0, 1.12], title="P(s′ | s, a)"),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"From {int(T)} transitions. Expected reward for this pair: {M['R'][s_sel, a_sel]:.3f}.")

# ---------------------------------------------------------------- how it works
with tabs[2]:
    st.subheader("How it works")
    st.markdown("Value iteration sweeps every state, keeps the best action's expected reward, and stops once "
                "no value changes by more than the tolerance.")
    st.latex(r"V_{k+1}(s) = \max_a \sum_{s'} P(s' \mid s,a)\,\big[\,r(s') + \gamma\,V_k(s')\,\big]")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Largest value change per sweep**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=vi["hist"], x=list(range(1, len(vi["hist"]) + 1)), name="MDP value iteration",
                                 line=dict(color=COLORS["mdp"], width=3)))
        fig.add_trace(go.Scatter(y=ad["hist"], x=list(range(1, len(ad["hist"]) + 1)), name="ADP fitted iteration",
                                 line=dict(color=COLORS["adp"], width=2.5, dash="dash")))
        fig.add_hline(y=tol, line=dict(color="#5a5091", dash="dot"), annotation_text=f"tolerance {tol:g}")
        fig.update_layout(**PLOT, height=320, yaxis=dict(type="log", title="max |Vₖ₊₁ − Vₖ|"), xaxis_title="Sweep",
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Value iteration stopped after {vi['iters']} sweeps; ADP after {ad['iters']}"
                   + ("" if ad["converged"] else " without settling") + ".")
    with c2:
        st.markdown("**Values as the sweeps run**")
        k = st.slider("Sweep", 0, vi["iters"], vi["iters"])
        Vk = vi["Vs"][k]
        Q = solvers.q_values(M, Vk, gamma)
        nf, nl = model.NB[0], model.NB[1]
        z, text = np.full((nf, nl), np.nan), [["" for _ in range(nl)] for _ in range(nf)]
        for f in range(nf):
            for l in range(nl):
                s = ((f * nl + l) * model.NB[2] + 1) * model.NB[3] + 1   # right clear, back clear
                if est["visits"][s] > 0:
                    z[f, l] = Vk[s]
                    a = int(np.argmax(Q[s]))
                    text[f][l] = f"{Vk[s]:.2f}<br>{ICON[a]} {room.ACT_SHORT[a]}"
                else:
                    text[f][l] = "no data"
        fig = go.Figure(go.Heatmap(z=z, x=[f"left {n}" for n in model.NAMES["left"]],
                                   y=[f"front {n}" for n in model.NAMES["front"]], text=text,
                                   texttemplate="%{text}", colorscale=[[0, "#eef1ff"], [1, "#2b64ff"]],
                                   showscale=False, xgap=4, ygap=4))
        fig.update_layout(**PLOT, height=300, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Right and back sensors clear. Each tile shows V(s) and the best action at that sweep.")
    c1, c2 = st.columns(2)
    with c1:
        agree = sum(vi["pi"][s] == ad["pi"][s] for s in visited)
        st.markdown("**ADP (approximate dynamic programming)**")
        st.markdown(f"Swaps the table of {model.NS} values for a linear model with {ad['nf']} weights: one per "
                    "sensor band plus a constant. Each sweep does the same Bellman backup, then refits the weights "
                    f"by weighted least squares. Its policy agrees with MDP on **{agree} of {len(visited)}** visited states.")
    with c2:
        th = hj["theta"]
        st.markdown("**Hooke-Jeeves pattern search**")
        st.markdown(f"Ignores the model and tunes a simple controller by trial runs in the room: sharp right if the "
                    f"front is under **{th[0]:.2f} m**, drift right if the left wall is under **{th[1]:.2f} m**, "
                    f"otherwise drift left if it is over **{th[2]:.2f} m**. It probes each setting up and down, "
                    f"jumps in the direction that worked, and halves the step when nothing helps "
                    f"({hj['evals']} trial runs). It only finds the best setting near its starting guess.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=hj["hist"], mode="markers", marker=dict(color=COLORS["hj"], size=5, opacity=.35), name="Each try"))
        fig.add_trace(go.Scatter(y=hj["best"], mode="lines", line=dict(color=COLORS["hj"], width=3), name="Best so far"))
        fig.update_layout(**PLOT, height=240, xaxis_title="Trial run", yaxis_title="Return", legend=dict(orientation="h", y=-0.35))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- output
with tabs[3]:
    st.subheader("Output")
    out = pd.DataFrame({
        "state": range(model.NS),
        **{f"{c}_band": [model.state_name(s)[k] for s in range(model.NS)] for k, c in enumerate(room.COLS)},
        "optimal_value": np.round(vi["V"], 6),
        "best_action": [room.ACTIONS[a] for a in vi["pi"]],
        "rows_in_data": est["visits"].astype(int),
    })
    csv = out.to_csv(index=False)
    try:
        OUT_CSV.write_text(csv)
    except OSError:
        pass
    st.markdown("The optimal value for each state, saved as `optimal_value_function.csv`.")
    c1, c2 = st.columns([2, 3])
    c1.download_button("Download optimal_value_function.csv", csv, "optimal_value_function.csv", "text/csv")
    only = c2.checkbox("Only states seen in the data", True)
    view = out.assign(adp_value=np.round(ad["V"], 3))
    st.dataframe(view[view.rows_in_data > 0] if only else view, hide_index=True, use_container_width=True, height=420)

# ---------------------------------------------------------------- accuracy
with tabs[4]:
    st.subheader("Accuracy")
    st.markdown(f"A smaller tolerance means more sweeps and values closer to the true optimum. When the last change "
                f"is Δ, every value is within γΔ/(1−γ) of optimal; with γ = {gamma:.2f} that is {gamma/(1-gamma):.1f} × Δ. "
                "Actual error is measured against a run to 1e-10.")
    acc = pd.DataFrame(R["acc"])
    acc["policy_differs"] = acc["diff"].astype(str) + f" of {len(visited)} states"
    acc = acc.rename(columns=dict(tol="Tolerance", iters="Sweeps", delta="Last change Δ", bound="Guaranteed within",
                                  err="Actual max error", policy_differs="Policy differs in")).drop(columns="diff")
    st.dataframe(acc.style.format({"Tolerance": "{:g}", "Last change Δ": "{:.2e}", "Guaranteed within": "{:.2e}",
                                   "Actual max error": "{:.2e}"})
                 .apply(lambda r: ["background-color:#8b7bff;color:#140f28;font-weight:600" if abs(r["Tolerance"] - tol) < 1e-12 else "" for _ in r], axis=1),
                 hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- comparison
with tabs[5]:
    st.subheader("Side by side")
    st.caption("Label match is checked against all real readings; everything else comes from the run in the room.")
    cols = st.columns(3)
    for col, m in zip(cols, METHODS):
        r = runs[m["id"]]
        col.metric(m["name"], f"{r['steps']} steps" if r["exit"] else "No exit")
        col.caption(f"Return {r['ret']:.1f}, {len(r['bumps'])} bumps, {match[m['id']]:.1f}% label match")
    table = pd.DataFrame({
        m["name"]: {
            "Reached the exit": "Yes" if runs[m["id"]]["exit"] else "No (600-step limit)",
            "Steps to the exit": str(runs[m["id"]]["steps"]) if runs[m["id"]]["exit"] else "–",
            "Wall bumps": str(len(runs[m["id"]]["bumps"])),
            "Steps in the on-track band": f"{runs[m['id']]['wall']:.0f}%",
            "Total return": f"{runs[m['id']]['ret']:.1f}",
            "Training time": f"{times[m['id']]*1000:.0f} ms",
            "Matches dataset labels": f"{match[m['id']]:.1f}%",
            "What it learns": {"mdp": f"{model.NS} state values", "hj": "3 thresholds", "adp": f"{ad['nf']} weights"}[m["id"]],
        } for m in METHODS
    })
    st.dataframe(table, use_container_width=True)
