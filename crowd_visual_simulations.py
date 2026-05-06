
"""
Crowd simulation visual prototypes for CMPE 49G presentation.

Outputs:
1. perlin_crowd_simulation.gif
2. particle_filter_station_sim.gif
3. hybrid_adaptive_prototype.gif
4. pf_rmse_summary.png

Run:
    python crowd_visual_simulations.py

Dependencies:
    numpy, matplotlib, pillow

Important:
These are didactic prototypes, not exact reproductions of the papers.
They are designed to visually explain the model ideas during a presentation.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# ============================================================
# Shared deterministic smooth value-noise / fBM implementation
# ============================================================

def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)

def make_grid(seed, size):
    rng = np.random.default_rng(seed)
    return rng.random((size, size))

def value_noise_periodic(x, y, grid):
    """
    Periodic 2D smooth value noise on [0,1]x[0,1].
    x, y can be scalar or numpy arrays.
    """
    size = grid.shape[0]
    gx = (np.asarray(x) * size) % size
    gy = (np.asarray(y) * size) % size

    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    x1 = (x0 + 1) % size
    y1 = (y0 + 1) % size

    sx = smoothstep(gx - x0)
    sy = smoothstep(gy - y0)

    v00 = grid[y0, x0]
    v10 = grid[y0, x1]
    v01 = grid[y1, x0]
    v11 = grid[y1, x1]

    ix0 = v00 * (1 - sx) + v10 * sx
    ix1 = v01 * (1 - sx) + v11 * sx
    return ix0 * (1 - sy) + ix1 * sy

def fbm_noise(x, y, t, seed=0, octaves=4, base_size=6, lacunarity=2.0, persistence=0.55, drift=(0.05, 0.03)):
    """
    Multi-octave periodic value noise used as a Perlin-like coherent field.
    This is intentionally self-contained, so no external 'noise' package is needed.
    """
    total = 0.0
    amp = 1.0
    amp_sum = 0.0
    for k in range(octaves):
        size = int(base_size * (lacunarity ** k))
        size = max(size, 2)
        grid = make_grid(seed + 1009 * k, size)
        xx = (np.asarray(x) + drift[0] * t * (0.5 + 0.2 * k)) % 1.0
        yy = (np.asarray(y) + drift[1] * t * (0.5 + 0.2 * k)) % 1.0
        total = total + amp * value_noise_periodic(xx, yy, grid)
        amp_sum += amp
        amp *= persistence
    return total / amp_sum

def wrap_positions(pos):
    pos[:, 0] %= 1.0
    pos[:, 1] %= 1.0
    return pos

# ============================================================
# 1. Perlin/noise-coordinated crowd
# ============================================================

def simulate_perlin_crowd():
    rng = np.random.default_rng(42)
    n_agents = 450
    frames = 72
    dt = 0.035
    beta = 0.86

    pos = rng.random((n_agents, 2))
    theta = rng.uniform(0, 2 * np.pi, n_agents)
    speed = rng.uniform(0.025, 0.075, n_agents)

    positions = []
    active_flags = []

    for f in range(frames):
        t = f / frames
        n_theta = fbm_noise(pos[:, 0], pos[:, 1], t, seed=10, octaves=4, base_size=5)
        n_speed = fbm_noise(pos[:, 0], pos[:, 1], t, seed=30, octaves=4, base_size=5, drift=(-0.03, 0.04))
        n_hazard = fbm_noise(pos[:, 0], pos[:, 1], t, seed=55, octaves=3, base_size=4)

        target_theta = 2 * np.pi * n_theta + rng.normal(0, 0.08, n_agents)
        # Vector-average heading to avoid wrap-around artifacts.
        vx = beta * np.cos(theta) + (1 - beta) * np.cos(target_theta)
        vy = beta * np.sin(theta) + (1 - beta) * np.sin(target_theta)
        theta = np.arctan2(vy, vx)

        target_speed = 0.015 + n_speed * 0.085
        speed = 0.90 * speed + 0.10 * target_speed

        # Hazard-like activation: not every NPC acts every frame.
        prob_active = 0.15 + 0.75 * n_hazard
        active = rng.random(n_agents) < prob_active

        step = np.column_stack([np.cos(theta), np.sin(theta)]) * speed[:, None] * dt
        # Inactive agents drift less, so activation field is visible.
        pos += step * (0.25 + 0.75 * active[:, None])
        pos = wrap_positions(pos)

        positions.append(pos.copy())
        active_flags.append(active.copy())

    # Precompute a quiver field.
    qn = 18
    xs = np.linspace(0, 1, qn)
    ys = np.linspace(0, 1, qn)
    X, Y = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=120)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")

    # Initial background field
    N0 = fbm_noise(X, Y, 0, seed=30, octaves=4, base_size=5, drift=(-0.03, 0.04))
    bg = ax.imshow(N0, extent=(0, 1, 0, 1), origin="lower", alpha=0.42)
    TH0 = 2 * np.pi * fbm_noise(X, Y, 0, seed=10, octaves=4, base_size=5)
    quiv = ax.quiver(X, Y, np.cos(TH0), np.sin(TH0), alpha=0.55, scale=35)
    scat_active = ax.scatter([], [], s=11, label="active / updating agents")
    scat_idle = ax.scatter([], [], s=5, alpha=0.35, label="inactive / low update")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    def update(f):
        t = f / frames
        N = fbm_noise(X, Y, t, seed=30, octaves=4, base_size=5, drift=(-0.03, 0.04))
        bg.set_data(N)
        TH = 2 * np.pi * fbm_noise(X, Y, t, seed=10, octaves=4, base_size=5)
        quiv.set_UVC(np.cos(TH), np.sin(TH))
        p = positions[f]
        active = active_flags[f]
        scat_active.set_offsets(p[active])
        scat_idle.set_offsets(p[~active])
        ax.set_title(
            "Perlin-like coherent field as AI coordinator\n"
            "nearby agents share direction/speed tendencies; global motion remains varied"
        )
        return bg, quiv, scat_active, scat_idle

    anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    path = OUT / "perlin_crowd_simulation.gif"
    anim.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)
    return path

# ============================================================
# 2. StationSim-like particle filter assimilation
# ============================================================

def resample_systematic(weights, rng):
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    idx = np.zeros(n, dtype=int)
    cumsum = np.cumsum(weights)
    i = j = 0
    while i < n:
        if positions[i] < cumsum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx

def station_step(pos, speeds, rng, dt=0.22, noise=0.02):
    """
    Very small StationSim-like corridor model:
    agents move from entrance to exit, with slight vertical interactions.
    """
    n = pos.shape[-2]
    new = pos.copy()

    # Move toward exit at x=10 with weak center preference.
    new[..., 0] += speeds * dt
    center_pull = (2.0 - new[..., 1]) * 0.035
    new[..., 1] += center_pull * dt

    # Minimal collision/crowding response.
    # For small n only, so O(n^2) is fine.
    if pos.ndim == 2:
        for i in range(n):
            delta = new[i] - new
            dist2 = np.sum(delta * delta, axis=1) + 1e-6
            mask = (dist2 < 0.22 ** 2) & (np.arange(n) != i)
            if np.any(mask):
                side = np.sign(np.sum(delta[mask, 1]))
                if side == 0:
                    side = rng.choice([-1, 1])
                new[i, 1] += 0.16 * side * dt
        new += rng.normal(0, noise, size=new.shape)
    else:
        p = pos.shape[0]
        new += rng.normal(0, noise, size=new.shape)

    # Boundaries and recycle agents after exit.
    new[..., 1] = np.clip(new[..., 1], 0.15, 3.85)
    crossed = new[..., 0] > 10.0
    if np.any(crossed):
        new[..., 0] = np.where(crossed, 0.0, new[..., 0])
        new[..., 1] = np.where(crossed, rng.uniform(0.7, 3.3, size=new[..., 1].shape), new[..., 1])
    return new

def simulate_particle_filter():
    rng = np.random.default_rng(7)
    n_agents = 12
    n_particles = 140
    frames = 80
    assim_every = 2
    sigma_obs = 0.35

    true_pos = np.column_stack([
        rng.uniform(0.0, 2.0, n_agents),
        rng.uniform(0.6, 3.4, n_agents)
    ])
    true_speed = rng.uniform(0.18, 0.33, n_agents)

    particles = true_pos[None, :, :] + rng.normal(0, 0.65, size=(n_particles, n_agents, 2))
    part_speed = true_speed[None, :] + rng.normal(0, 0.05, size=(n_particles, n_agents))
    part_speed = np.clip(part_speed, 0.12, 0.40)
    weights = np.ones(n_particles) / n_particles

    histories = []
    rmse_filter = []
    rmse_model_only = []

    model_only = particles[0].copy()
    model_speed = part_speed[0].copy()

    last_obs = np.full_like(true_pos, np.nan)

    for f in range(frames):
        true_pos = station_step(true_pos, true_speed, rng, noise=0.008)
        particles = station_step(particles, part_speed, rng, noise=0.028)
        model_only = station_step(model_only, model_speed, rng, noise=0.028)

        assimilated = False
        if f % assim_every == 0:
            assimilated = True
            obs = true_pos + rng.normal(0, sigma_obs, size=true_pos.shape)
            obs[:, 0] = np.clip(obs[:, 0], 0, 10)
            obs[:, 1] = np.clip(obs[:, 1], 0, 4)
            last_obs = obs.copy()

            # Gaussian likelihood from Euclidean observation error.
            diff = particles - obs[None, :, :]
            sq = np.sum(diff * diff, axis=(1, 2))
            logw = -0.5 * sq / (sigma_obs ** 2)
            logw -= np.max(logw)
            weights = np.exp(logw)
            weights += 1e-300
            weights /= np.sum(weights)

            # Resample if degeneracy is high.
            ess = 1.0 / np.sum(weights * weights)
            if ess < n_particles * 0.55:
                idx = resample_systematic(weights, rng)
                particles = particles[idx]
                part_speed = part_speed[idx] + rng.normal(0, 0.01, size=part_speed.shape)
                part_speed = np.clip(part_speed, 0.12, 0.40)
                weights = np.ones(n_particles) / n_particles
        else:
            obs = last_obs

        estimate = np.average(particles, axis=0, weights=weights)
        rmse_filter.append(float(np.sqrt(np.mean(np.sum((estimate - true_pos) ** 2, axis=1)))))
        rmse_model_only.append(float(np.sqrt(np.mean(np.sum((model_only - true_pos) ** 2, axis=1)))))

        histories.append({
            "true": true_pos.copy(),
            "particles_agent0": particles[:, 0, :].copy(),
            "estimate": estimate.copy(),
            "model_only": model_only.copy(),
            "obs": obs.copy(),
            "assimilated": assimilated,
            "rmse_filter": rmse_filter[-1],
            "rmse_model": rmse_model_only[-1],
        })

    # Animation
    fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=120)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.set_xlabel("station x")
    ax.set_ylabel("station y")
    ax.add_patch(Rectangle((0, 0), 10, 4, fill=False, linewidth=1.2))
    ax.add_patch(Rectangle((-0.15, 0.7), 0.15, 2.6, alpha=0.25))
    ax.add_patch(Rectangle((10.0, 0.7), 0.15, 2.6, alpha=0.25))
    ax.text(0.05, 3.78, "entrance", fontsize=8)
    ax.text(8.92, 3.78, "exit", fontsize=8)

    true_sc = ax.scatter([], [], s=32, label="pseudo-truth agents")
    est_sc = ax.scatter([], [], s=23, marker="D", label="PF estimate")
    obs_sc = ax.scatter([], [], s=25, marker="x", label="noisy observations")
    model_sc = ax.scatter([], [], s=18, alpha=0.45, label="model-only prediction")
    spread_sc = ax.scatter([], [], s=7, alpha=0.18, label="particle cloud for one agent")
    text = ax.text(0.02, 0.04, "", transform=ax.transAxes, fontsize=8,
                   bbox=dict(boxstyle="round", alpha=0.8))
    ax.legend(loc="lower right", fontsize=7, framealpha=0.85)

    def update(f):
        h = histories[f]
        true_sc.set_offsets(h["true"])
        est_sc.set_offsets(h["estimate"])
        model_sc.set_offsets(h["model_only"])
        obs_sc.set_offsets(h["obs"])
        spread_sc.set_offsets(h["particles_agent0"])
        marker = "ASSIMILATION: predict → compare → reweight → resample" if h["assimilated"] else "prediction step"
        text.set_text(
            f"{marker}\n"
            f"PF RMSE: {h['rmse_filter']:.3f} | model-only RMSE: {h['rmse_model']:.3f}\n"
            f"Particles: {n_particles}, agents: {n_agents}"
        )
        ax.set_title("StationSim-like particle filter data assimilation")
        return true_sc, est_sc, obs_sc, model_sc, spread_sc, text

    anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    gif_path = OUT / "particle_filter_station_sim.gif"
    anim.save(gif_path, writer=PillowWriter(fps=10))
    plt.close(fig)

    # RMSE summary chart
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
    ax.plot(rmse_filter, label="particle filter")
    ax.plot(rmse_model_only, label="model-only")
    ax.set_xlabel("simulation frame")
    ax.set_ylabel("position RMSE")
    ax.set_title("Particle filter reduces drift from pseudo-truth")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    rmse_path = OUT / "pf_rmse_summary.png"
    fig.savefig(rmse_path)
    plt.close(fig)

    return gif_path, rmse_path

# ============================================================
# 3. Hybrid adaptive prototype
# ============================================================

def simulate_hybrid_adaptive():
    rng = np.random.default_rng(123)
    n_agents = 650
    frames = 82
    dt = 0.035

    pos = np.column_stack([
        rng.uniform(0.02, 0.98, n_agents),
        rng.uniform(0.05, 0.95, n_agents)
    ])
    theta = rng.uniform(0, 2 * np.pi, n_agents)
    speed = rng.uniform(0.018, 0.050, n_agents)

    sensor_centers = np.array([[0.36, 0.35], [0.36, 0.65], [0.68, 0.50]])
    sensor_r = np.array([0.14, 0.14, 0.16])
    exits = np.array([[1.02, 0.25], [1.02, 0.75]])
    danger_center = np.array([0.54, 0.50])
    danger_r = 0.13

    histories = []
    for f in range(frames):
        t = f / frames
        event_strength = 0.0 if f < 25 else min(1.0, (f - 25) / 20)

        n_theta = fbm_noise(pos[:, 0], pos[:, 1], t, seed=501, octaves=4, base_size=6)
        n_speed = fbm_noise(pos[:, 0], pos[:, 1], t, seed=777, octaves=3, base_size=5, drift=(-0.02, 0.035))
        target_theta = 2 * np.pi * n_theta

        # Sensor promotion: agents entering zones are "tracked" by expensive assimilation.
        dist_to_sensors = np.sqrt(((pos[:, None, :] - sensor_centers[None, :, :]) ** 2).sum(axis=2))
        tracked = np.any(dist_to_sensors < sensor_r[None, :], axis=1)

        # Danger/high-risk zone after frame 25 promotes more agents.
        danger_dist = np.linalg.norm(pos - danger_center[None, :], axis=1)
        high_risk = danger_dist < (danger_r + 0.10 * event_strength)
        tracked = tracked | high_risk

        # Density feedback from sensor zones.
        density = np.mean(dist_to_sensors < sensor_r[None, :], axis=0)
        congestion = np.clip(np.max(density) / 0.22, 0, 1)

        # Background Perlin motion: coherent but not goal-accurate.
        vx = np.cos(target_theta)
        vy = np.sin(target_theta)

        # Weak exit drift for all agents, because it is a station / evacuation prototype.
        nearest_exit = exits[np.argmin(np.linalg.norm(pos[:, None, :] - exits[None, :, :], axis=2), axis=1)]
        goal_vec = nearest_exit - pos
        goal_vec /= np.linalg.norm(goal_vec, axis=1, keepdims=True) + 1e-9

        # Dynamic field adaptation: if congestion/event is high, flow is biased around the danger region.
        away = pos - danger_center[None, :]
        away /= np.linalg.norm(away, axis=1, keepdims=True) + 1e-9
        around = np.column_stack([-away[:, 1], away[:, 0]])
        near_event = np.exp(-((danger_dist / 0.28) ** 2))
        adapt_vec = (0.55 * away + 0.45 * around) * (event_strength * near_event * (0.5 + congestion))[:, None]

        # PF-tracked agents receive stronger correction toward estimated safe flow.
        base_vec = np.column_stack([vx, vy])
        direction = 0.45 * base_vec + 0.35 * goal_vec + adapt_vec
        direction[tracked] = (
            0.15 * base_vec[tracked] +
            0.70 * goal_vec[tracked] +
            1.10 * adapt_vec[tracked]
        )

        # Normalize direction.
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9
        target_theta2 = np.arctan2(direction[:, 1], direction[:, 0])

        # Smooth update.
        beta = np.where(tracked, 0.68, 0.88)
        blend_x = beta * np.cos(theta) + (1 - beta) * np.cos(target_theta2)
        blend_y = beta * np.sin(theta) + (1 - beta) * np.sin(target_theta2)
        theta = np.arctan2(blend_y, blend_x)

        target_speed = 0.018 + 0.060 * n_speed + 0.030 * tracked + 0.030 * event_strength * high_risk
        speed = 0.88 * speed + 0.12 * target_speed

        pos += np.column_stack([np.cos(theta), np.sin(theta)]) * speed[:, None] * dt
        # recycle exited agents to entrance side
        exited = pos[:, 0] > 1.02
        pos[exited, 0] = rng.uniform(0.0, 0.05, np.sum(exited))
        pos[exited, 1] = rng.uniform(0.15, 0.85, np.sum(exited))
        pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.02)
        pos[:, 1] = np.clip(pos[:, 1], 0.02, 0.98)

        histories.append({
            "pos": pos.copy(),
            "tracked": tracked.copy(),
            "high_risk": high_risk.copy(),
            "event_strength": event_strength,
            "congestion": congestion,
            "density": density.copy(),
        })

    # Quiver background for adaptive field.
    qn = 19
    xs = np.linspace(0.04, 0.96, qn)
    ys = np.linspace(0.05, 0.95, qn)
    X, Y = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=120)
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("virtual station x")
    ax.set_ylabel("virtual station y")

    # static map
    ax.add_patch(Rectangle((0, 0), 1.0, 1.0, fill=False, linewidth=1.2))
    ax.add_patch(Rectangle((-0.01, 0.15), 0.03, 0.70, alpha=0.20))
    ax.add_patch(Rectangle((1.0, 0.18), 0.03, 0.18, alpha=0.20))
    ax.add_patch(Rectangle((1.0, 0.64), 0.03, 0.18, alpha=0.20))
    for c, r in zip(sensor_centers, sensor_r):
        ax.add_patch(Circle(c, r, fill=False, linestyle="--", linewidth=1.3, alpha=0.75))
    danger_patch = Circle(tuple(danger_center), danger_r, fill=False, linewidth=2.0, alpha=0.8)
    ax.add_patch(danger_patch)

    TH0 = 2 * np.pi * fbm_noise(X, Y, 0, seed=501, octaves=4, base_size=6)
    quiv = ax.quiver(X, Y, np.cos(TH0), np.sin(TH0), alpha=0.45, scale=38)

    bg_sc = ax.scatter([], [], s=6, alpha=0.35, label="Perlin background agents")
    tr_sc = ax.scatter([], [], s=13, label="promoted PF-tracked agents")
    risk_sc = ax.scatter([], [], s=18, marker="x", label="high-risk / corrected agents")
    text = ax.text(0.02, 0.03, "", transform=ax.transAxes, fontsize=8,
                   bbox=dict(boxstyle="round", alpha=0.82))
    ax.text(0.02, 0.90, "entrance", fontsize=8)
    ax.text(0.88, 0.83, "exits", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.86)

    def update(f):
        h = histories[f]
        event_strength = h["event_strength"]
        congestion = h["congestion"]

        # Field shown is the Perlin substrate plus local adaptive bias near danger.
        TH = 2 * np.pi * fbm_noise(X, Y, f / frames, seed=501, octaves=4, base_size=6)
        U = np.cos(TH)
        V = np.sin(TH)
        grid_pos = np.stack([X, Y], axis=2)
        away = grid_pos - danger_center
        dist = np.linalg.norm(away, axis=2) + 1e-9
        away = away / dist[..., None]
        around = np.stack([-away[..., 1], away[..., 0]], axis=2)
        near = np.exp(-((dist / 0.30) ** 2))
        bias = (0.55 * away + 0.45 * around) * (event_strength * near * (0.5 + congestion))[..., None]
        U = U + bias[..., 0] * 2.0
        V = V + bias[..., 1] * 2.0
        norm = np.sqrt(U * U + V * V) + 1e-9
        quiv.set_UVC(U / norm, V / norm)

        p = h["pos"]
        tracked = h["tracked"]
        risk = h["high_risk"]
        bg_sc.set_offsets(p[~tracked])
        tr_sc.set_offsets(p[tracked & ~risk])
        risk_sc.set_offsets(p[risk])

        danger_patch.set_radius(danger_r + 0.10 * event_strength)
        text.set_text(
            "Hybrid proposal prototype\n"
            f"Background agents: {np.sum(~tracked)} | PF-tracked: {np.sum(tracked)}\n"
            f"Sensor congestion estimate: {congestion:.2f}\n"
            f"Dynamic field adaptation: {event_strength:.2f}"
        )
        ax.set_title(
            "Hybrid adaptive crowd simulation\n"
            "Perlin controls cheap background behavior; particle-filter zones correct critical agents"
        )
        return quiv, bg_sc, tr_sc, risk_sc, text, danger_patch

    anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    gif_path = OUT / "hybrid_adaptive_prototype.gif"
    anim.save(gif_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    return gif_path



if __name__ == "__main__":
    paths = []
    paths.append(simulate_perlin_crowd())
    pf_gif, rmse_png = simulate_particle_filter()
    paths.extend([pf_gif, rmse_png])
    paths.append(simulate_hybrid_adaptive())

    print("Created:")
    for p in paths:
        print(" -", p)

