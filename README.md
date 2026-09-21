# Wall-following robot navigation with MDPs

A Streamlit dashboard for the exercise *MDPs – Wall-following Robot navigation*. It builds a Markov
Decision Process from the Kaggle wall-following dataset, solves it with **value iteration**, saves
the optimal value of every state to `optimal_value_function.csv`, and compares the result with
**ADP** (approximate dynamic programming) and **Hooke-Jeeves** pattern search in an animated arena.

## What's in the repository

```
├── app.py                        Streamlit dashboard
├── value_iteration.py            Command-line version: runs value iteration and saves the CSV
├── optimal_value_function.csv    Output of value iteration with the default settings
├── mdp/
│   ├── room.py                   Simulated room, robot motion, the four 60° sensors
│   ├── model.py                  States, rewards, transition probabilities from the data
│   ├── solvers.py                Value iteration, ADP, Hooke-Jeeves, room runs
│   └── arena.py                  Animated arcade arena (HTML component)
├── data/sensor_readings_4.csv    Kaggle dataset (SD_front, SD_left, SD_right, SD_back, Class)
├── requirements.txt
└── .streamlit/config.toml        Theme
```

## How it maps to the exercise

| Exercise item | Where it lives |
|---|---|
| **Input:** transition probabilities, rewards, discount factor (gamma), convergence threshold (tolerance), states and action space | Sidebar and the *Input* tab. States are distance bands of the four sensors (3 × 5 × 2 × 2 = 60 states); actions are the four movement classes; transition probabilities are counted from consecutive rows of the data. |
| **Output:** optimal value for each state, saved as `optimal_value_function.csv` | *Output* tab (table and download button) and `value_iteration.py` |
| **Accuracy:** depends on the convergence threshold | *Accuracy* tab: sweeps, error bound γΔ/(1−γ) and actual error for tolerances 0.1 to 0.000001 |
| **How it works:** value iteration updates state values until convergence | *How it works* tab: Bellman update, convergence chart, and a slider to replay the sweeps |
| **Dataset:** Kaggle, sensor readings with movement class labels | *Dataset* tab |

The additions:

- **ADP** – fitted value iteration with a linear value function (13 weights instead of 60 values).
- **Hooke-Jeeves** – direct search over three thresholds of a simple controller, scored by trial runs in the room.
- **Side by side** – steps to the exit, bumps, time in the on-track band, total return, training time and agreement with the dataset labels.

The dataset does not include the robot's real room, so the arena is a stand-in: policies learned
from the real readings are tested in a simulated room at 9 steps a second, like the real sensors.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Command-line version only:

```bash
python value_iteration.py --gamma 0.9 --tol 1e-4
```

## Deploy on Streamlit Community Cloud

1. Create a new **public** repository on GitHub (for example `wall-following-mdp`).
2. Upload every file and folder above, keeping the same structure, including the hidden
   `.streamlit` folder. On github.com: *Add file → Upload files*, drag the folders in, then *Commit changes*.
   Or from a terminal inside the project folder:
   ```bash
   git init
   git add .
   git commit -m "Wall-following robot MDP dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/wall-following-mdp.git
   git push -u origin main
   ```
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and choose **Create app**.
4. Pick the repository, branch `main`, and main file path `app.py`. Under *Advanced settings*, choose Python 3.11 or 3.12.
5. Click **Deploy**. The first build takes a few minutes; you get a link like `https://<name>.streamlit.app`.

## Dataset

Wall-Following Robot Navigation Data (Kaggle, originally UCI Machine Learning Repository).
A SCITOS G5 robot followed the wall clockwise for 4 rounds, logging 9 readings a second.
`sensor_readings_4.csv` holds the minimum ultrasound distance in 60° arcs at the front, left,
right and back, plus the movement class: Move-Forward, Slight-Right-Turn, Sharp-Right-Turn, Slight-Left-Turn.
