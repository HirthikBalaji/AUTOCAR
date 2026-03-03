import pygame
import cvxpy as cp
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# ====================== CONFIG ======================
WIDTH, HEIGHT = 1200, 600
FPS = 60

DT = 0.2
HORIZON = 10

LANES = 3
LANE_Y = [180, 300, 420]

V_TARGET = 30.0
D_SAFE = 150.0
A_MIN, A_MAX = -5.0, 4.0
LAT_ACC_MAX = 1.5

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MPC Autonomous Highway")
clock = pygame.time.Clock()

# ---------- Plot setup ----------
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4))
fig.tight_layout(pad=2.0)
acc_hist = deque([0] * 50, maxlen=50)
jerk_hist = deque([0] * 50, maxlen=50)
prev_acc = 0.0
plot_counter = 0
PLOT_EVERY = 15  # Update plot every 15 frames to avoid slowdown

# ===================================================
class Car:
    def __init__(self, x, lane, v, color=(200, 200, 200)):
        self.x = float(x)
        self.lane = lane
        self.y = float(LANE_Y[lane])
        self.v = float(v)
        self.vy = 0.0   # lateral velocity — persisted across frames
        self.a = 0.0
        self.color = color

ego = Car(200, 1, 20, color=(0, 220, 80))
traffic = []

SPAWN_COOLDOWN = 90  # frames between spawns
spawn_timer = 0

def spawn_traffic():
    global spawn_timer
    spawn_timer -= 1
    if spawn_timer > 0:
        return
    if len(traffic) < 8:
        lane = random.randint(0, LANES - 1)
        x = random.randint(600, 1100)
        v = random.uniform(14, 24)
        # Avoid spawning on top of existing cars
        if all(abs(c.x - x) > 80 or c.lane != lane for c in traffic):
            traffic.append(Car(x, lane, v))
        spawn_timer = SPAWN_COOLDOWN

# ===================================================
def mpc_longitudinal(v0, d0, v_lead):
    """
    v0     : ego current velocity
    d0     : distance to lead car (ego.x to lead.x)
    v_lead : lead car velocity (used for gap prediction)
    """
    a = cp.Variable(HORIZON)
    cost = 0
    constraints = [a >= A_MIN, a <= A_MAX]

    v = v0
    d = d0
    for k in range(HORIZON):
        v_next = v + a[k] * DT
        # Gap evolves: lead moves at v_lead, ego moves at v
        d_next = d + (v_lead - v) * DT

        # Penalize deviation from target speed
        cost += 2.0 * cp.square(v_next - V_TARGET)
        # Soft safety distance penalty (quadratic barrier)
        cost += 20.0 * cp.square(cp.pos(D_SAFE - d_next))
        # Comfort: penalize large accelerations
        cost += 0.5 * cp.square(a[k])
        # Jerk penalty (consecutive acc difference)
        if k > 0:
            cost += 0.3 * cp.square(a[k] - a[k - 1])

        v = v_next
        d = d_next

    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if a.value is not None:
            return float(a.value[0])
    except Exception:
        pass
    return 0.0


def mpc_lateral(y0, vy0, y_target):
    """
    y0       : current lateral position (pixels)
    vy0      : current lateral velocity (pixels/s) — MUST be passed in from state
    y_target : target lane centre (pixels)
    Returns  : (ay_cmd, predicted_vy_next)
    """
    ay = cp.Variable(HORIZON)
    cost = 0
    constraints = []

    y  = float(y0)
    vy = float(vy0)   # warm-start from real lateral velocity
    for k in range(HORIZON):
        vy = vy + ay[k] * DT
        y  = y  + vy  * DT
        cost += 5.0 * cp.square(y - y_target)   # track lane centre
        cost += 0.1 * cp.square(vy)             # damp lateral speed → kills overshoot
        cost += 0.8 * cp.square(ay[k])          # penalise large commands
        if k > 0:
            cost += 0.5 * cp.square(ay[k] - ay[k-1])  # jerk penalty → smooth transitions
        constraints += [ay[k] >= -LAT_ACC_MAX, ay[k] <= LAT_ACC_MAX]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if ay.value is not None:
            ay0 = float(ay.value[0])
            vy_next = vy0 + ay0 * DT
            return ay0, vy_next
    except Exception:
        pass
    return 0.0, vy0 * 0.9   # fallback: damp velocity to settle


def find_lead():
    leads = [c for c in traffic if c.lane == ego.lane and c.x > ego.x]
    return min(leads, key=lambda c: c.x) if leads else None


def lane_is_safe(lane):
    """Check if a lane is clear enough for a lane change."""
    for c in traffic:
        if c.lane == lane:
            if abs(c.x - ego.x) < 120:
                return False
    return True


def update_plots(acc):
    global prev_acc, plot_counter
    jerk = (acc - prev_acc) / DT
    prev_acc = acc
    acc_hist.append(acc)
    jerk_hist.append(jerk)

    plot_counter += 1
    if plot_counter % PLOT_EVERY != 0:
        return

    ax1.clear()
    ax2.clear()
    ax1.plot(list(acc_hist), color='tomato', linewidth=1.5)
    ax1.axhline(0, color='white', linewidth=0.5, linestyle='--')
    ax1.set_ylabel("Accel (m/s²)", fontsize=8)
    ax1.set_ylim(A_MIN - 1, A_MAX + 1)
    ax1.set_facecolor('#1e1e2e')
    fig.patch.set_facecolor('#1e1e2e')

    ax2.plot(list(jerk_hist), color='cornflowerblue', linewidth=1.5)
    ax2.axhline(0, color='white', linewidth=0.5, linestyle='--')
    ax2.set_ylabel("Jerk (m/s³)", fontsize=8)
    ax2.set_facecolor('#1e1e2e')

    plt.pause(0.001)


# ====================== DRAWING HELPERS ======================
font = pygame.font.SysFont("monospace", 16)

def draw_road():
    # Road background
    pygame.draw.rect(screen, (45, 45, 50), (0, LANE_Y[0] - 60, WIDTH, LANE_Y[-1] - LANE_Y[0] + 120))
    # Lane dividers
    for i, y in enumerate(LANE_Y):
        if i < len(LANE_Y) - 1:
            mid_y = (LANE_Y[i] + LANE_Y[i + 1]) // 2
            for x in range(0, WIDTH, 60):
                pygame.draw.rect(screen, (200, 180, 50), (x, mid_y - 3, 35, 5))
    # Road edges
    pygame.draw.line(screen, (255, 255, 255), (0, LANE_Y[0] - 55), (WIDTH, LANE_Y[0] - 55), 3)
    pygame.draw.line(screen, (255, 255, 255), (0, LANE_Y[-1] + 55), (WIDTH, LANE_Y[-1] + 55), 3)

def draw_car(x, y, color, label=""):
    rect = pygame.Rect(int(x) - 24, int(y) - 12, 48, 24)
    pygame.draw.rect(screen, color, rect, border_radius=5)
    pygame.draw.rect(screen, (255, 255, 255), rect, 1, border_radius=5)
    if label:
        surf = font.render(label, True, (0, 0, 0))
        screen.blit(surf, (int(x) - surf.get_width() // 2, int(y) - surf.get_height() // 2))

def draw_hud():
    texts = [
        f"Speed : {ego.v * 3.6:.1f} km/h",
        f"Accel : {ego.a:+.2f} m/s²",
        f"Lane  : {ego.lane + 1}",
        f"Target: {target_lane + 1}",
    ]
    lead = find_lead()
    if lead:
        texts.append(f"Gap   : {lead.x - ego.x:.0f} m")
    else:
        texts.append("Gap   : free")

    for i, t in enumerate(texts):
        surf = font.render(t, True, (200, 255, 200))
        screen.blit(surf, (10, 10 + i * 22))


# ====================== MAIN LOOP ======================
running = True
target_lane = ego.lane
lane_change_cooldown = 0  # Prevent rapid lane switching

while running:
    clock.tick(FPS)
    screen.fill((20, 20, 25))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    spawn_traffic()

    # --- Longitudinal MPC ---
    lead = find_lead()
    if lead:
        d_rel = lead.x - ego.x
        v_lead = lead.v
    else:
        d_rel = 500.0
        v_lead = V_TARGET  # No lead: assume open road

    ego.a = mpc_longitudinal(ego.v, d_rel, v_lead)
    ego.v = max(0.0, min(ego.v + ego.a * DT, V_TARGET * 1.5))

    # World scrolls: keep ego visually at x=200
    world_move = ego.v * DT  # how far ego moved this frame (meters → pixels, 1:1 scale)

    # --- Lane Change Logic ---
    lane_change_cooldown = max(0, lane_change_cooldown - 1)

    if lead and (lead.x - ego.x) < 90 and lane_change_cooldown == 0:
        # Try adjacent lanes first (prefer minimal change)
        candidates = sorted(range(LANES), key=lambda l: abs(l - ego.lane))
        for l in candidates:
            if l != ego.lane and lane_is_safe(l):
                target_lane = l
                lane_change_cooldown = 120  # ~2 seconds at 60fps
                break

    # If we've arrived at target lane, confirm lane
    if ego.y == float(LANE_Y[target_lane]):
        ego.lane = target_lane

    # --- Lateral MPC ---
    lat_acc, ego.vy = mpc_lateral(ego.y, ego.vy, float(LANE_Y[target_lane]))
    ego.y += ego.vy * DT
    # Clamp to road bounds
    ego.y = max(float(LANE_Y[0]) - 40, min(float(LANE_Y[-1]) + 40, ego.y))

    # Snap + zero velocity once close enough to avoid micro-oscillation
    if abs(ego.y - LANE_Y[target_lane]) < 1.5:
        ego.y  = float(LANE_Y[target_lane])
        ego.vy = 0.0

    # --- Update traffic positions ---
    for c in traffic[:]:
        # In ego's reference frame: traffic moves left by (ego speed - traffic speed)
        c.x -= (ego.v - c.v) * DT
        if c.x < -200 or c.x > WIDTH + 400:
            traffic.remove(c)

    # --- Drawing ---
    draw_road()

    # Safety circle around ego
    pygame.draw.circle(screen, (0, 200, 220), (int(ego.x), int(ego.y)), int(D_SAFE), 1)

    # Traffic cars
    for c in traffic:
        col = (220, 80, 80) if (c.lane == ego.lane and 0 < c.x - ego.x < D_SAFE * 1.5) else (180, 180, 180)
        draw_car(c.x, c.y, col)

    # Ego car
    draw_car(ego.x, ego.y, ego.color, "EGO")

    draw_hud()
    update_plots(ego.a)
    pygame.display.flip()

pygame.quit()
plt.close()