# Autonomous Car Simulator

This workspace contains a simple CPU‑only car simulator built with
`pygame`. The vehicle follows a straight road and must avoid randomly
placed obstacles using a combination of **reinforcement learning (Q‑learning)**
and a lightweight **model predictive controller (MPC)** fallback.

## Features

* Car kinematic model (bicycle model) with steering constraints.
* A **curved lane** defined by a sinusoidal centerline; boundaries are
  drawn and the car is physically constrained so it cannot leave the
  lane during simulation.
* Obstacles spawn relative to the current lane center rather than a
  fixed straight line.
* Camera offset keeps the vehicle near the left edge of the window.
* MPC sampler that evaluates candidate steering angles over a short
  horizon and penalises collisions, lane deviation, or approaching the
  boundary. This controller is used at runtime to plan and follow the
  lane; it guarantees trajectory planning independent of the RL agent.
* (Optional) RL agent with discretized states is trained offline but
  currently not used in the live simulation — it can be enabled later
  for experimentation.
* Training loop runs at startup; epsilon is decayed and you can adjust
  `TRAIN_EPISODES`/`max_steps` in `main.py` for more thorough learning.

## Running

Activate your virtual environment and run:

```powershell
& .\.venv\Scripts\Activate.ps1
python main.py
```

The program will print training progress in the terminal and then open
a window showing the curving lane and the car driving through the
obstacle field. The simulation **will automatically stop** if the car
bumps an obstacle or wanders outside the lane; you can also close the
window or hit the close button to exit manually.

## Customization

* Adjust `OBS_COUNT`, `HORIZON`, and `STEER_ACTIONS` in `main.py` for
  more obstacles, longer planning horizon, or finer steering resolution.
* You can comment out the training section to reuse a previously
  learned `rl_agent.q` table (add serialization yourself if desired).
* Replace the Q‑learning agent with a neural network for a deeper RL
  setup, or integrate a proper MPC solver (CVXOPT, OSQP, etc.).

---

Feel free to experiment!
