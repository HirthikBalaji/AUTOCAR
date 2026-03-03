import pygame
import random
import math
import cvxpy as cp
import numpy as np

# ================== CONFIG ==================
WIDTH, HEIGHT = 1000, 600
LANE_Y = [220, 300, 380]     # 3 lanes
DT = 0.1

V_TARGET = 24.0
D_SAFE = 45.0
A_MIN, A_MAX = -3.5, 2.5
HORIZON = 8

EGO_COLOR = (0, 170, 255)
TRAFFIC_COLOR = (255, 80, 80)
ROAD_COLOR = (120, 120, 120)
# ============================================

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3-Lane ACC + Lane Change (ML + MPC)")
clock = pygame.time.Clock()

# ================= VEHICLE ==================
class Vehicle:
    def __init__(self, x, lane, speed):
        self.x = x
        self.lane = lane
        self.y = LANE_Y[lane]
        self.v = speed
        self.w, self.h = 40, 20

    def rect(self, cam_x):
        return pygame.Rect(self.x - cam_x, self.y - 10, self.w, self.h)

    def draw(self, cam_x, color):
        pygame.draw.rect(screen, color, self.rect(cam_x))

# ================= MPC (LONGITUDINAL) ==================
def mpc_accel(v_ego, d_rel):
    a = cp.Variable(HORIZON)
    v = v_ego
    d = d_rel
    cost = 0

    for k in range(HORIZON):
        v = v + a[k] * DT
        d = d - v * DT

        cost += (
            (v - V_TARGET) ** 2 +
            4 * cp.pos(D_SAFE - d) ** 2 +
            0.15 * a[k] ** 2
        )

    constraints = [a >= A_MIN, a <= A_MAX]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)

    return float(a.value[0])

# ================= ML LANE POLICY ==================
def lane_policy(ego, traffic):
    """
    ML-style policy:
    - If too close in current lane → evaluate left & right lanes
    - Choose lane with max forward clearance
    """
    best_lane = ego.lane
    best_clearance = -1

    for lane in range(3):
        clearance = 999
        for car in traffic:
            if car.lane == lane and car.x > ego.x:
                clearance = min(clearance, car.x - ego.x)
        if clearance > best_clearance:
            best_clearance = clearance
            best_lane = lane

    return best_lane

# ================= TRAFFIC ==================
def spawn_traffic():
    cars = []
    for lane in range(3):
        for _ in range(2):
            x = random.randint(300, 1200)
            v = random.uniform(14, 22)
            cars.append(Vehicle(x, lane, v))
    return cars

# ================= MAIN ==================
ego = Vehicle(200, 1, 22)
traffic = spawn_traffic()

target_lane = ego.lane
cam_x = 0
running = True

while running:
    clock.tick(60)
    screen.fill((30, 30, 30))

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # ----- Camera follows ego -----
    cam_x = ego.x - 200

    # ----- Move traffic -----
    for car in traffic:
        if random.random() < 0.01:
            car.v += random.uniform(-2, 2)
            car.v = max(10, min(24, car.v))

        car.x += car.v * DT

        # recycle traffic
        if car.x < ego.x - 400:
            car.x = ego.x + random.randint(600, 1000)
            car.lane = random.randint(0, 2)
            car.y = LANE_Y[car.lane]
            car.v = random.uniform(14, 22)

    # ----- Relative distance in same lane -----
    d_rel = 999
    for car in traffic:
        if car.lane == ego.lane and car.x > ego.x:
            d_rel = min(d_rel, car.x - ego.x)

    # ----- ML lane decision -----
    if d_rel < D_SAFE + 10:
        target_lane = lane_policy(ego, traffic)

    # ----- Smooth lane change -----
    ego.y += 0.08 * (LANE_Y[target_lane] - ego.y)
    if abs(ego.y - LANE_Y[target_lane]) < 2:
        ego.lane = target_lane

    # ----- MPC acceleration -----
    accel = mpc_accel(ego.v, d_rel)
    ego.v += accel * DT
    ego.v = max(0, ego.v)
    ego.x += ego.v * DT

    # ----- Draw lanes -----
    for y in LANE_Y:
        pygame.draw.line(screen, ROAD_COLOR, (0, y), (WIDTH, y), 2)

    # ----- Draw traffic -----
    for car in traffic:
        car.draw(cam_x, TRAFFIC_COLOR)

    # ----- Draw ego -----
    ego.draw(cam_x, EGO_COLOR)

    pygame.display.flip()

pygame.quit()