# AUTOCAR — Autonomous Car Simulator

A simple, educational Python project demonstrating a **Model Predictive
Controller (MPC)** navigating a multi‑lane highway. All logic is contained
in `main.py`, making the code easy to read, modify, and extend for
experiments with planning, control, and traffic interactions.

## 🚗 About the Simulation

`main.py` creates a real‑time view of an ego vehicle (green) traveling
on a three‑lane road. Surrounding traffic is randomly spawned and moves
relative to the ego car’s frame. The controller consists of two MPC
routines:

1. **Longitudinal MPC** — keeps the ego car close to a target speed and
   maintains a safe gap to the lead vehicle, with comfort penalties for
   acceleration and jerk.
2. **Lateral MPC** — performs smooth lane tracking and lane changes while
   respecting lateral acceleration limits.

Additional logic automatically initiates lane changes when a slower car
appears ahead, provided the adjacent lane is clear. Acceleration and
jerk are plotted live using `matplotlib`.

## ✨ Key Features

- CPU‑only implementation using `pygame` for graphics and `cvxpy`/`OSQP`
  as the solver
- Configurable highway with 3 lanes and adjustable dimensions
- Dynamic traffic spawn and simple kinematic model for other vehicles
- Safety circle visualisation around ego car
- HUD displaying speed, acceleration, current/target lane, and gap
- Tuning constants at the top of `main.py` allow easy experimentation

## 🛠 Requirements

- Python 3.10 or newer (works on Windows, macOS, Linux)
- Python packages listed in `pyproject.toml`:
  `pygame`, `cvxpy`, `numpy`, `matplotlib`

## ⚡ Setup

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1   # Windows
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Running

```powershell
python main.py
```

After launching, a window displays the highway. The ego car uses MPC to
maneuver; close the window or press **ESC** to quit.

## ⚙️ Configuration

Open `main.py` and modify constants near the top:

- `WIDTH`, `HEIGHT`, `FPS` – rendering parameters
- `DT`, `HORIZON` – time step and MPC horizon
- `LANES`, `LANE_Y` – lane count and vertical positions
- `V_TARGET`, `D_SAFE`, `A_MIN`, `A_MAX` – speed/distance/acceleration
  thresholds
- `LAT_ACC_MAX` – lateral acceleration cap
- `SPAWN_COOLDOWN` – spacing between new traffic vehicles
- `PLOT_EVERY` – how often the acceleration/jerk plots update

Values are treated in a 1‑pixel = 1‑meter scale to keep the math simple.

## 🧩 Development & Extensions

- Break `main.py` into modules (e.g. controllers, models, visualization)
- Add longitudinal control terms (braking, fuel efficiency, comfort)
- Swap in a different solver or LP/QP formulation
- Implement reinforcement learning or save replay data
- Capture video frames or add a simple web UI

## 🤝 Contributing

Bug reports, enhancements, and pull requests are welcome. The project
originated as a teaching demo; keep the code clear and well‑documented. An
MIT or other permissive license is recommended if sharing publicly.

---

Have fun experimenting with MPC on the highway! 🚦

