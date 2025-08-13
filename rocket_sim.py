"""
Rocket Launch Simulator — 2D vertical flight (Pygame + Matplotlib)

Features
- Adjustable parameters: mass (dry + fuel), thrust, Isp, Cd, area, dt
- Atmosphere: exponential density model
- Gravity: varies with altitude
- Drag: quadratic (0.5*rho*v^2*Cd*A)
- Thrust ends when propellant is exhausted
- Keyboard: SPACE launch/pause, R reset, G show graphs, ESC quit
- Simple HUD: time, altitude, velocity, fuel remaining
- Plots altitude vs time and velocity vs time using Matplotlib

Run
  python rocket_sim.py

Dependencies
  pip install pygame matplotlib
"""

import math
import pygame
import sys
from dataclasses import dataclass
from typing import List, Tuple

# Optional: Matplotlib only imported when needed
plt = None  # lazy import

# ---------- Constants ----------
G0 = 9.80665            # m/s^2, sea-level gravity
R_EARTH = 6_371_000.0   # m
RHO0 = 1.225            # kg/m^3, sea-level air density
SCALE_HEIGHT = 8500.0   # m, exponential atmosphere scale height

# ---------- Configurable Parameters ----------
@dataclass
class RocketParams:
    dry_mass: float = 1_000.0          # kg (structure + engine)
    fuel_mass: float = 9_000.0         # kg
    thrust: float = 200_000.0          # N (constant while fuel remains)
    Isp: float = 280.0                 # s
    Cd: float = 0.4                    # drag coefficient
    area: float = 1.0                  # m^2
    dt: float = 0.02                   # s integration step
    max_time: float = 400.0            # s cutoff
    max_alt_plot: float = 120_000.0    # m for plotting scale

# ---------- Pygame Display Settings ----------
WIDTH, HEIGHT = 600, 800
SCREEN_SCALE = 0.002   # pixels per meter at low altitude
HUD_COLOR = (240, 240, 240)
BG_COLOR = (10, 12, 18)
ROCKET_COLOR = (200, 200, 220)
GROUND_COLOR = (40, 80, 40)
FUEL_COLOR = (120, 200, 120)

# ---------- Simulation State ----------
@dataclass
class State:
    t: float = 0.0
    h: float = 0.0
    v: float = 0.0
    m: float = 0.0
    fuel: float = 0.0
    flying: bool = False

@dataclass
class History:
    time: List[float]
    alt: List[float]
    vel: List[float]
    mass: List[float]
    thrust: List[float]
    drag: List[float]

# ---------- Physics Helpers ----------
def gravity(h: float) -> float:
    return G0 * (R_EARTH / (R_EARTH + h)) ** 2

def air_density(h: float) -> float:
    return RHO0 * math.exp(-h / SCALE_HEIGHT)

def drag_force(v: float, h: float, Cd: float, A: float) -> float:
    rho = air_density(h)
    return 0.5 * rho * v * v * Cd * A

def mass_flow(thrust: float, Isp: float) -> float:
    return thrust / (Isp * G0)

def step(params: RocketParams, s: State) -> Tuple[State, Tuple[float, float, float]]:
    dt = params.dt
    g = gravity(s.h)
    D = drag_force(s.v, s.h, params.Cd, params.area) * (1 if s.v > 0 else -1)

    T = 0.0
    mdot = 0.0
    if s.flying and s.fuel > 0:
        T = params.thrust
        mdot = mass_flow(params.thrust, params.Isp)
        dm = mdot * dt
        if dm > s.fuel:
            dm = s.fuel
            mdot = dm / dt
            T = mdot * params.Isp * G0
        s.fuel -= dm
        s.m -= dm

    a = (T - D - s.m * g) / s.m
    s.v += a * dt
    s.h += s.v * dt
    s.t += dt

    if s.h < 0 and s.v < 0:
        s.h = 0
        s.v = 0
        s.flying = False

    return s, (T, D, g)

# ---------- Rendering ----------
def world_to_screen_y(h: float, camera_alt: float) -> int:
    scale = SCREEN_SCALE / max(1.0, (camera_alt + 1000.0) / 10_000.0)
    y = HEIGHT - int(h * scale) - 50
    return y

def draw_scene(screen, font, params: RocketParams, s: State, fuels0: float, T: float, D: float, g: float):
    screen.fill(BG_COLOR)
    pygame.draw.rect(screen, GROUND_COLOR, (0, HEIGHT - 40, WIDTH, 40))

    rocket_h_px = 60
    rocket_w_px = 16
    y = world_to_screen_y(s.h, s.h)
    x = WIDTH // 2 - rocket_w_px // 2
    pygame.draw.rect(screen, ROCKET_COLOR, (x, y - rocket_h_px, rocket_w_px, rocket_h_px), border_radius=4)

    if s.flying and s.fuel > 0 and T > 0:
        flame_len = min(40, int(20 + 0.0002 * T))
        pygame.draw.polygon(screen, (255, 180, 80), [(x, y), (x + rocket_w_px, y), (x + rocket_w_px // 2, y + flame_len)])

    def bl(txt, yline):
        surf = font.render(txt, True, HUD_COLOR)
        screen.blit(surf, (10, yline))

    bl(f"t = {s.t:6.1f} s", 10)
    bl(f"h = {s.h:8.1f} m", 35)
    bl(f"v = {s.v:8.2f} m/s", 60)
    bl(f"m = {s.m:8.1f} kg", 85)
    bl(f"g = {g:5.2f} m/s²", 110)
    bl(f"T = {T:8.0f} N", 135)
    bl(f"D = {D:8.0f} N", 160)

    fuel_frac = 0 if fuels0 <= 0 else s.fuel / fuels0
    pygame.draw.rect(screen, (80, 80, 80), (WIDTH - 40, 20, 16, 200), 2, border_radius=6)
    pygame.draw.rect(screen, FUEL_COLOR, (WIDTH - 38, 20 + int((1 - fuel_frac) * 200), 12, int(fuel_frac * 200)))

    instr = ["SPACE: launch/pause", "R: reset", "G: graphs", "ESC: quit"]
    for i, t in enumerate(instr):
        surf = font.render(t, True, (180, 180, 200))
        screen.blit(surf, (10, HEIGHT - 30 * (len(instr) - i)))

# ---------- Plotting ----------
def show_graphs(hist: History, params: RocketParams):
    global plt
    if plt is None:
        import matplotlib.pyplot as plt
        globals()['plt'] = plt
    plt.figure()
    plt.plot(hist.time, hist.alt)
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude vs Time')
    plt.grid(True)
    plt.tight_layout()

    plt.figure()
    plt.plot(hist.time, hist.vel)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------- Main Loop ----------
def run(params: RocketParams):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Rocket Launch Simulator — 2D Vertical Flight')
    font = pygame.font.SysFont('consolas', 18)
    clock = pygame.time.Clock()

    def make_state():
        s = State()
        s.fuel = params.fuel_mass
        s.m = params.dry_mass + params.fuel_mass
        s.h = 0.0
        s.v = 0.0
        s.t = 0.0
        s.flying = False
        return s

    s = make_state()
    hist = History(time=[], alt=[], vel=[], mass=[], thrust=[], drag=[])
    running = True
    T = D = g = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    s.flying = not s.flying
                elif event.key == pygame.K_r:
                    s = make_state()
                    hist = History(time=[], alt=[], vel=[], mass=[], thrust=[], drag=[])
                elif event.key == pygame.K_g:
                    if hist.time:
                        show_graphs(hist, params)

        if s.t < params.max_time:
            s, (T, D, g) = step(params, s)
            hist.time.append(s.t)
            hist.alt.append(max(0.0, s.h))
            hist.vel.append(s.v)
            hist.mass.append(s.m)
            hist.thrust.append(T)
            hist.drag.append(D)

        draw_scene(screen, font, params, s, fuels0=params.fuel_mass, T=T, D=D, g=g)
        pygame.display.flip()
        clock.tick(1.0 / params.dt)

    pygame.quit()

if __name__ == '__main__':
    params = RocketParams(
        dry_mass=800.0,
        fuel_mass=2200.0,
        thrust=50000.0,
        Isp=260.0,
        Cd=0.4,
        area=0.9,
        dt=0.02,
        max_time=300.0,
    )
    run(params)
