"""Command-line version of the exercise: value iteration on the wall-following dataset.

    python value_iteration.py --gamma 0.9 --tol 1e-4

Inputs: transition probabilities and rewards (built from the data), discount factor,
convergence threshold, states and action space. Output: optimal_value_function.csv.
"""
import argparse
from pathlib import Path

import pandas as pd

from mdp import model, room, solvers

ROOT = Path(__file__).parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "sensor_readings_4.csv"))
    p.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    p.add_argument("--tol", type=float, default=1e-4, help="convergence threshold")
    p.add_argument("--out", default=str(ROOT / "optimal_value_function.csv"))
    a = p.parse_args()

    df = model.load_dataset(a.data)
    bands = model.Bands()
    est = model.estimate(df, bands)
    M = model.build_model(est, model.DEFAULT_REWARDS)
    vi = solvers.value_iteration(M, a.gamma, a.tol)

    out = pd.DataFrame({
        "state": range(model.NS),
        **{f"{c}_band": [model.state_name(s)[k] for s in range(model.NS)] for k, c in enumerate(room.COLS)},
        "optimal_value": vi["V"].round(6),
        "best_action": [room.ACTIONS[x] for x in vi["pi"]],
        "rows_in_data": est["visits"].astype(int),
    })
    out.to_csv(a.out, index=False)
    pol = solvers.table_policy(vi["pi"], bands)
    print(f"Rows: {len(df)}   States: {model.NS} ({int((est['visits'] > 0).sum())} visited)   Actions: {len(room.ACTIONS)}")
    print(f"Converged in {vi['iters']} sweeps (last change {vi['delta']:.2e} < tolerance {a.tol:g})")
    print(f"Error bound: every value within {a.gamma * vi['delta'] / (1 - a.gamma):.2e} of optimal")
    print(f"Policy matches the dataset labels on {solvers.label_match(df, pol):.1f}% of rows")
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
