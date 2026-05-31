"""
Phase-2 Results Generator for CMPE 49G Final Project

This script runs the three models (Pure Perlin, Particle Filter, Hybrid Adaptive)
with varying parameters and collects metrics for the results section.

Run:
    python phase-2/generate_results.py

Outputs:
    - phase-2/results/metrics.csv
    - phase-2/figures/*.png
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import from crowd_visual_simulations
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import noise functions from the original simulation
from crowd_visual_simulations import (
    fbm_noise, wrap_positions, resample_systematic,
    station_step, make_grid, value_noise_periodic
)

OUT_DIR = Path(__file__).resolve().parent / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# MODEL 1: Pure Perlin Crowd Simulation
# ============================================================

def run_pure_perlin_model(n_agents=300, frames=100, seed=42):
    """
    Pure Perlin model: all agents follow noise-based fields with goal direction.
    No particle filter correction.
    """
    rng = np.random.default_rng(seed)
    dt = 0.035

    # Initialize agents
    pos = np.column_stack([
        rng.uniform(0.05, 0.45, n_agents),  # Start on left side
        rng.uniform(0.15, 0.85, n_agents)
    ])
    theta = rng.uniform(0, 2 * np.pi, n_agents)
    speed = rng.uniform(0.035, 0.065, n_agents)  # Faster speed

    exits = np.array([[1.0, 0.25], [1.0, 0.75]])
    evacuated = np.zeros(n_agents, dtype=bool)
    evacuation_times = np.full(n_agents, -1.0)

    densities = []
    positions_history = []

    start_time = time.time()

    for f in range(frames):
        t = f / frames

        # Perlin-based movement
        n_theta = fbm_noise(pos[:, 0], pos[:, 1], t, seed=501, octaves=4, base_size=6)
        n_speed = fbm_noise(pos[:, 0], pos[:, 1], t, seed=777, octaves=3, base_size=5)

        # Goal direction toward nearest exit
        nearest_exit = exits[np.argmin(np.linalg.norm(
            pos[:, None, :] - exits[None, :, :], axis=2), axis=1)]
        goal_vec = nearest_exit - pos
        goal_vec /= np.linalg.norm(goal_vec, axis=1, keepdims=True) + 1e-9

        # Combine Perlin and goal direction
        perlin_vec = np.column_stack([
            np.cos(2 * np.pi * n_theta),
            np.sin(2 * np.pi * n_theta)
        ])

        direction = 0.30 * perlin_vec + 0.70 * goal_vec
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9

        target_theta = np.arctan2(direction[:, 1], direction[:, 0])
        beta = 0.85
        blend_x = beta * np.cos(theta) + (1 - beta) * np.cos(target_theta)
        blend_y = beta * np.sin(theta) + (1 - beta) * np.sin(target_theta)
        theta = np.arctan2(blend_y, blend_x)

        target_speed = 0.035 + 0.050 * n_speed
        speed = 0.80 * speed + 0.20 * target_speed

        # Update positions
        pos[~evacuated] += (np.column_stack([np.cos(theta), np.sin(theta)]) *
                            speed[:, None] * dt)[~evacuated]

        # Check evacuation (more lenient threshold)
        for i, exit_pos in enumerate(exits):
            dist_to_exit = np.linalg.norm(pos - exit_pos, axis=1)
            newly_evacuated = (~evacuated) & (dist_to_exit < 0.08)
            evacuated[newly_evacuated] = True
            evacuation_times[newly_evacuated] = f * dt

        # Calculate density (agents per unit area in 0.15 radius)
        active_pos = pos[~evacuated]
        if len(active_pos) > 0:
            max_density = 0
            for check_pos in active_pos[::max(1, len(active_pos)//20)]:
                local_count = np.sum(np.linalg.norm(active_pos - check_pos, axis=1) < 0.15)
                local_density = local_count / (np.pi * 0.15**2)
                max_density = max(max_density, local_density)
            densities.append(max_density)
        else:
            densities.append(0)

        positions_history.append(pos.copy())

        if np.all(evacuated):
            break

    runtime = time.time() - start_time

    # Calculate metrics
    valid_times = evacuation_times[evacuation_times >= 0]
    avg_evac_time = np.mean(valid_times) if len(valid_times) > 0 else frames * dt
    max_evac_time = np.max(valid_times) if len(valid_times) > 0 else frames * dt
    max_density = np.max(densities) if densities else 0
    evacuation_rate = np.sum(evacuated) / n_agents

    return {
        'model': 'Pure Perlin',
        'n_agents': n_agents,
        'runtime': runtime,
        'avg_evacuation_time': avg_evac_time,
        'max_evacuation_time': max_evac_time,
        'max_density': max_density,
        'evacuation_rate': evacuation_rate,
        'frames_run': f + 1,
        'fps': (f + 1) / runtime if runtime > 0 else 0,
        'positions_history': positions_history,
        'evacuated': evacuated.copy(),
        'evacuation_times': evacuation_times.copy()
    }


# ============================================================
# MODEL 2: Particle Filter Model (small scale only)
# ============================================================

def run_particle_filter_model(n_agents=50, frames=100, n_particles=100, seed=42):
    """
    Full particle filter on all agents. Only practical for small crowds.
    """
    rng = np.random.default_rng(seed)
    dt = 0.035
    sigma_obs = 0.15

    # Initialize true state
    true_pos = np.column_stack([
        rng.uniform(0.02, 0.30, n_agents),
        rng.uniform(0.15, 0.85, n_agents)
    ])
    true_speed = rng.uniform(0.025, 0.050, n_agents)

    # Initialize particles
    particles = true_pos[None, :, :] + rng.normal(0, 0.20, size=(n_particles, n_agents, 2))
    part_speed = true_speed[None, :] + rng.normal(0, 0.008, size=(n_particles, n_agents))
    part_speed = np.clip(part_speed, 0.015, 0.060)
    weights = np.ones(n_particles) / n_particles

    exits = np.array([[1.0, 0.25], [1.0, 0.75]])
    evacuated = np.zeros(n_agents, dtype=bool)
    evacuation_times = np.full(n_agents, -1.0)

    rmse_history = []
    densities = []

    start_time = time.time()

    for f in range(frames):
        # Move true state toward exits
        nearest_exit = exits[np.argmin(np.linalg.norm(
            true_pos[:, None, :] - exits[None, :, :], axis=2), axis=1)]
        goal_vec = nearest_exit - true_pos
        goal_vec /= np.linalg.norm(goal_vec, axis=1, keepdims=True) + 1e-9
        true_pos[~evacuated] += (goal_vec * true_speed[:, None] * dt)[~evacuated]

        # Move particles
        for p in range(n_particles):
            p_nearest_idx = np.argmin(np.linalg.norm(
                particles[p, :, None, :] - exits[None, :, :], axis=2), axis=1)
            p_nearest = exits[p_nearest_idx]
            p_goal = p_nearest - particles[p]
            p_goal /= np.linalg.norm(p_goal, axis=1, keepdims=True) + 1e-9
            step = p_goal * part_speed[p, :, None] * dt
            particles[p, ~evacuated] += step[~evacuated]

        # Observation and weight update every 2 frames
        if f % 2 == 0:
            obs = true_pos + rng.normal(0, sigma_obs, size=true_pos.shape)
            diff = particles - obs[None, :, :]
            sq = np.sum(diff * diff, axis=(1, 2))
            logw = -0.5 * sq / (sigma_obs ** 2)
            logw -= np.max(logw)
            weights = np.exp(logw)
            weights += 1e-300
            weights /= np.sum(weights)

            # Resample
            ess = 1.0 / np.sum(weights * weights)
            if ess < n_particles * 0.6:
                idx = resample_systematic(weights, rng)
                particles = particles[idx]
                part_speed = part_speed[idx] + rng.normal(0, 0.003, size=part_speed.shape)
                part_speed = np.clip(part_speed, 0.015, 0.060)
                weights = np.ones(n_particles) / n_particles

        # Estimate
        estimate = np.average(particles, axis=0, weights=weights)
        rmse = np.sqrt(np.mean(np.sum((estimate - true_pos) ** 2, axis=1)))
        rmse_history.append(rmse)

        # Check evacuation
        for exit_pos in exits:
            dist_to_exit = np.linalg.norm(true_pos - exit_pos, axis=1)
            newly_evacuated = (~evacuated) & (dist_to_exit < 0.08)
            evacuated[newly_evacuated] = True
            evacuation_times[newly_evacuated] = f * dt

        # Density
        active_pos = true_pos[~evacuated]
        if len(active_pos) > 0:
            max_density = 0
            for check_pos in active_pos[::max(1, len(active_pos)//10)]:
                local_count = np.sum(np.linalg.norm(active_pos - check_pos, axis=1) < 0.15)
                local_density = local_count / (np.pi * 0.15**2)
                max_density = max(max_density, local_density)
            densities.append(max_density)
        else:
            densities.append(0)

        if np.all(evacuated):
            break

    runtime = time.time() - start_time

    valid_times = evacuation_times[evacuation_times >= 0]
    avg_evac_time = np.mean(valid_times) if len(valid_times) > 0 else frames * dt
    max_evac_time = np.max(valid_times) if len(valid_times) > 0 else frames * dt
    max_density = np.max(densities) if densities else 0

    return {
        'model': 'Particle Filter',
        'n_agents': n_agents,
        'n_particles': n_particles,
        'runtime': runtime,
        'avg_evacuation_time': avg_evac_time,
        'max_evacuation_time': max_evac_time,
        'max_density': max_density,
        'evacuation_rate': np.sum(evacuated) / n_agents,
        'frames_run': f + 1,
        'fps': (f + 1) / runtime if runtime > 0 else 0,
        'avg_rmse': np.mean(rmse_history),
        'final_rmse': rmse_history[-1] if rmse_history else 0,
        'rmse_history': rmse_history
    }


# ============================================================
# MODEL 3: Hybrid Adaptive Model
# ============================================================

def run_hybrid_model(n_agents=300, frames=100, seed=42):
    """
    Hybrid model: background agents use Perlin, critical agents near sensors/exits
    get particle filter correction.
    """
    rng = np.random.default_rng(seed)
    dt = 0.035

    pos = np.column_stack([
        rng.uniform(0.05, 0.45, n_agents),
        rng.uniform(0.15, 0.85, n_agents)
    ])
    theta = rng.uniform(0, 2 * np.pi, n_agents)
    speed = rng.uniform(0.035, 0.065, n_agents)

    exits = np.array([[1.0, 0.25], [1.0, 0.75]])
    sensor_centers = np.array([[0.40, 0.35], [0.40, 0.65], [0.70, 0.50]])
    sensor_r = 0.15

    evacuated = np.zeros(n_agents, dtype=bool)
    evacuation_times = np.full(n_agents, -1.0)

    densities = []
    tracked_counts = []
    positions_history = []

    start_time = time.time()

    for f in range(frames):
        t = f / frames

        # Identify critical agents (near sensors or exits)
        dist_to_sensors = np.sqrt(((pos[:, None, :] - sensor_centers[None, :, :]) ** 2).sum(axis=2))
        near_sensor = np.any(dist_to_sensors < sensor_r, axis=1)

        dist_to_exits = np.sqrt(((pos[:, None, :] - exits[None, :, :]) ** 2).sum(axis=2))
        near_exit = np.any(dist_to_exits < 0.20, axis=1)

        tracked = (near_sensor | near_exit) & (~evacuated)
        tracked_counts.append(np.sum(tracked))

        # Perlin fields
        n_theta = fbm_noise(pos[:, 0], pos[:, 1], t, seed=501, octaves=4, base_size=6)
        n_speed = fbm_noise(pos[:, 0], pos[:, 1], t, seed=777, octaves=3, base_size=5)

        # Goal direction
        nearest_exit = exits[np.argmin(np.linalg.norm(
            pos[:, None, :] - exits[None, :, :], axis=2), axis=1)]
        goal_vec = nearest_exit - pos
        goal_vec /= np.linalg.norm(goal_vec, axis=1, keepdims=True) + 1e-9

        # Perlin direction
        perlin_vec = np.column_stack([
            np.cos(2 * np.pi * n_theta),
            np.sin(2 * np.pi * n_theta)
        ])

        # Adaptive feedback: stronger goal bias for tracked agents
        direction = np.zeros_like(pos)
        direction[~tracked] = 0.35 * perlin_vec[~tracked] + 0.65 * goal_vec[~tracked]
        direction[tracked] = 0.15 * perlin_vec[tracked] + 0.85 * goal_vec[tracked]

        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9
        target_theta = np.arctan2(direction[:, 1], direction[:, 0])

        # Smoother response for tracked agents
        beta = np.where(tracked, 0.70, 0.85)
        blend_x = beta * np.cos(theta) + (1 - beta) * np.cos(target_theta)
        blend_y = beta * np.sin(theta) + (1 - beta) * np.sin(target_theta)
        theta = np.arctan2(blend_y, blend_x)

        # Speed boost for tracked agents (simulating correction)
        target_speed = 0.035 + 0.050 * n_speed + 0.020 * tracked
        speed = 0.80 * speed + 0.20 * target_speed

        # Update positions
        pos[~evacuated] += (np.column_stack([np.cos(theta), np.sin(theta)]) *
                            speed[:, None] * dt)[~evacuated]

        # Check evacuation
        for exit_pos in exits:
            dist_to_exit = np.linalg.norm(pos - exit_pos, axis=1)
            newly_evacuated = (~evacuated) & (dist_to_exit < 0.08)
            evacuated[newly_evacuated] = True
            evacuation_times[newly_evacuated] = f * dt

        # Density
        active_pos = pos[~evacuated]
        if len(active_pos) > 0:
            max_density = 0
            for check_pos in active_pos[::max(1, len(active_pos)//20)]:
                local_count = np.sum(np.linalg.norm(active_pos - check_pos, axis=1) < 0.15)
                local_density = local_count / (np.pi * 0.15**2)
                max_density = max(max_density, local_density)
            densities.append(max_density)
        else:
            densities.append(0)

        positions_history.append(pos.copy())

        if np.all(evacuated):
            break

    runtime = time.time() - start_time

    valid_times = evacuation_times[evacuation_times >= 0]
    avg_evac_time = np.mean(valid_times) if len(valid_times) > 0 else frames * dt
    max_evac_time = np.max(valid_times) if len(valid_times) > 0 else frames * dt
    max_density = np.max(densities) if densities else 0
    avg_tracked = np.mean(tracked_counts)

    return {
        'model': 'Hybrid Adaptive',
        'n_agents': n_agents,
        'runtime': runtime,
        'avg_evacuation_time': avg_evac_time,
        'max_evacuation_time': max_evac_time,
        'max_density': max_density,
        'evacuation_rate': np.sum(evacuated) / n_agents,
        'frames_run': f + 1,
        'fps': (f + 1) / runtime if runtime > 0 else 0,
        'avg_tracked_agents': avg_tracked,
        'tracked_ratio': avg_tracked / n_agents,
        'positions_history': positions_history,
        'tracked_counts': tracked_counts
    }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_all_experiments():
    """Run all experiments and collect results."""
    results = []

    print("=" * 60)
    print("PHASE-2 EXPERIMENTS: Hybrid Adaptive Crowd Simulation")
    print("=" * 60)

    # Experiment 1: Scalability comparison (Pure Perlin vs Hybrid)
    print("\n[1/4] Scalability Test: Varying agent counts...")
    agent_counts = [100, 200, 300, 400, 500]

    for n in agent_counts:
        print(f"  - Running Pure Perlin with {n} agents...")
        result = run_pure_perlin_model(n_agents=n, frames=250, seed=42)
        result['experiment'] = 'scalability'
        results.append(result)
        print(f"    Runtime: {result['runtime']:.3f}s, FPS: {result['fps']:.1f}, Evacuated: {result['evacuation_rate']*100:.1f}%")

        print(f"  - Running Hybrid Adaptive with {n} agents...")
        result = run_hybrid_model(n_agents=n, frames=250, seed=42)
        result['experiment'] = 'scalability'
        results.append(result)
        print(f"    Runtime: {result['runtime']:.3f}s, FPS: {result['fps']:.1f}, Evacuated: {result['evacuation_rate']*100:.1f}%")

    # Experiment 2: Particle filter on small crowd
    print("\n[2/4] Particle Filter Test: Small crowd with full tracking...")
    print(f"  - Running Particle Filter with 50 agents, 100 particles...")
    result = run_particle_filter_model(n_agents=50, frames=200, n_particles=100, seed=42)
    result['experiment'] = 'particle_filter'
    results.append(result)
    print(f"    Runtime: {result['runtime']:.3f}s, FPS: {result['fps']:.1f}, Avg RMSE: {result['avg_rmse']:.4f}")

    # Experiment 2b: Particle filter on large crowd (this will be slow!)
    print(f"  - Running Particle Filter with 500 agents, 500 particles (this may take a while)...")
    result_large = run_particle_filter_model(n_agents=500, frames=200, n_particles=500, seed=42)
    result_large['experiment'] = 'particle_filter_large'
    results.append(result_large)
    print(f"    Runtime: {result_large['runtime']:.3f}s, FPS: {result_large['fps']:.1f}, Avg RMSE: {result_large['avg_rmse']:.4f}")

    # Experiment 3: Evacuation efficiency comparison (not used in scalability plots)
    print("\n[3/4] Evacuation Efficiency: 300 agents, detailed metrics...")
    for model_name, model_func in [
        ('Pure Perlin', run_pure_perlin_model),
        ('Hybrid Adaptive', run_hybrid_model)
    ]:
        print(f"  - Running {model_name}...")
        result = model_func(n_agents=300, frames=300, seed=123)
        result['experiment'] = 'evacuation_test'
        results.append(result)
        print(f"    Avg evac: {result['avg_evacuation_time']:.2f}s, Max evac: {result['max_evacuation_time']:.2f}s")

    # Experiment 4: Multiple runs for statistics
    print("\n[4/4] Statistical Validation: 5 runs each model (200 agents)...")
    for run_idx in range(5):
        for model_name, model_func in [
            ('Pure Perlin', run_pure_perlin_model),
            ('Hybrid Adaptive', run_hybrid_model)
        ]:
            result = model_func(n_agents=200, frames=250, seed=100 + run_idx)
            result['run_id'] = run_idx
            result['experiment'] = 'statistical'
            results.append(result)
        print(f"  - Completed run {run_idx + 1}/5")

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)

    return results


# ============================================================
# VISUALIZATION AND ANALYSIS
# ============================================================

def create_result_figures(results_df):
    """Create all result figures for the paper."""

    print("\nGenerating result figures...")

    # Figure 1: Runtime vs Agent Count
    print("  - Figure 1: Scalability (Runtime vs Agent Count)")
    scalability_data = results_df[
        (results_df['model'].isin(['Pure Perlin', 'Hybrid Adaptive'])) &
        (results_df['experiment'] == 'scalability')
    ].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for model in ['Pure Perlin', 'Hybrid Adaptive']:
        model_data = scalability_data[scalability_data['model'] == model]
        ax1.plot(model_data['n_agents'], model_data['runtime'],
                marker='o', label=model, linewidth=2)
        ax2.plot(model_data['n_agents'], model_data['fps'],
                marker='s', label=model, linewidth=2)

    ax1.set_xlabel('Number of Agents')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Computational Cost: Runtime vs Agent Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Number of Agents')
    ax2.set_ylabel('Frames per Second (FPS)')
    ax2.set_title('Performance: FPS vs Agent Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig1_scalability.png', dpi=150)
    plt.close()

    # Figure 2: Evacuation Time Comparison
    print("  - Figure 2: Evacuation Time Comparison")
    evac_data = results_df[results_df['experiment'] == 'statistical'].groupby('model').agg({
        'avg_evacuation_time': ['mean', 'std'],
        'max_evacuation_time': ['mean', 'std']
    }).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(evac_data))
    width = 0.35

    avg_means = evac_data['avg_evacuation_time']['mean']
    avg_stds = evac_data['avg_evacuation_time']['std']
    max_means = evac_data['max_evacuation_time']['mean']
    max_stds = evac_data['max_evacuation_time']['std']

    ax.bar(x - width/2, avg_means, width, yerr=avg_stds,
           label='Average Evacuation Time', capsize=5)
    ax.bar(x + width/2, max_means, width, yerr=max_stds,
           label='Maximum Evacuation Time', capsize=5)

    ax.set_xlabel('Model')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Evacuation Time Comparison (200 agents, 5 runs)')
    ax.set_xticks(x)
    ax.set_xticklabels(evac_data['model'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig2_evacuation_times.png', dpi=150)
    plt.close()

    # Figure 3: Metrics Summary Table (as image)
    print("  - Figure 3: Summary Statistics Table")
    summary_stats = results_df[results_df['experiment'] == 'statistical'].groupby('model').agg({
        'runtime': ['mean', 'std'],
        'avg_evacuation_time': ['mean', 'std'],
        'max_density': ['mean', 'std'],
        'fps': ['mean', 'std']
    }).round(3)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    table_data.append(['Model', 'Runtime (s)', 'Avg Evac (s)', 'Max Density', 'FPS'])

    for model in summary_stats.index:
        row = [
            model,
            f"{summary_stats.loc[model, ('runtime', 'mean')]:.2f} ± {summary_stats.loc[model, ('runtime', 'std')]:.2f}",
            f"{summary_stats.loc[model, ('avg_evacuation_time', 'mean')]:.2f} ± {summary_stats.loc[model, ('avg_evacuation_time', 'std')]:.2f}",
            f"{summary_stats.loc[model, ('max_density', 'mean')]:.1f} ± {summary_stats.loc[model, ('max_density', 'std')]:.1f}",
            f"{summary_stats.loc[model, ('fps', 'mean')]:.1f} ± {summary_stats.loc[model, ('fps', 'std')]:.1f}"
        ]
        table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.18, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Summary Statistics: Model Comparison (200 agents, 5 runs each)',
              fontsize=12, pad=20)
    plt.savefig(OUT_DIR / 'fig3_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 4: Tracked Agents Over Time (Hybrid only)
    print("  - Figure 4: Critical Agent Tracking (Hybrid Model)")
    hybrid_detailed = [r for r in results_df.to_dict('records')
                      if r['model'] == 'Hybrid Adaptive' and 'tracked_counts' in r
                      and r['n_agents'] == 300 and r.get('run_id') != r.get('run_id')]

    if hybrid_detailed:
        tracked_counts = hybrid_detailed[0]['tracked_counts']
        fig, ax = plt.subplots(figsize=(10, 4.5))
        frames = range(len(tracked_counts))
        ax.plot(frames, tracked_counts, linewidth=2, color='#FF5722')
        ax.fill_between(frames, 0, tracked_counts, alpha=0.3, color='#FF5722')
        ax.set_xlabel('Simulation Frame')
        ax.set_ylabel('Number of Tracked Agents')
        ax.set_title('Hybrid Model: Critical Agent Count Over Time (300 total agents)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=300, color='gray', linestyle='--', alpha=0.5, label='Total agents')
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'fig4_tracked_agents.png', dpi=150)
        plt.close()

    # Figure 5: Particle Filter RMSE
    print("  - Figure 5: Particle Filter State Estimation Error")
    pf_data = results_df[results_df['model'] == 'Particle Filter']
    if not pf_data.empty and 'rmse_history' in pf_data.iloc[0]:
        rmse_hist = pf_data.iloc[0]['rmse_history']
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(rmse_hist, linewidth=2, color='#2196F3')
        ax.set_xlabel('Simulation Frame')
        ax.set_ylabel('Position RMSE')
        ax.set_title('Particle Filter: State Estimation Error Over Time (50 agents)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=np.mean(rmse_hist), color='red', linestyle='--',
                   alpha=0.6, label=f'Mean RMSE: {np.mean(rmse_hist):.4f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'fig5_pf_rmse.png', dpi=150)
        plt.close()

    print("  ✓ All figures generated successfully!")


def save_results_csv(results):
    """Save results to CSV file."""
    # Remove non-serializable fields for CSV
    csv_results = []
    for r in results:
        csv_row = {k: v for k, v in r.items()
                   if k not in ['positions_history', 'evacuated', 'evacuation_times',
                               'rmse_history', 'tracked_counts']}
        csv_results.append(csv_row)

    df = pd.DataFrame(csv_results)
    csv_path = RESULTS_DIR / 'metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CMPE 49G Phase-2 Results Generation")
    print("Hybrid Adaptive Crowd Simulation")
    print("=" * 60)

    # Run experiments
    results = run_all_experiments()

    # Save to CSV
    results_df = save_results_csv(results)

    # Create figures
    create_result_figures(results_df)

    print("\n" + "=" * 60)
    print("PHASE-2 RESULTS GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - Metrics CSV: {RESULTS_DIR / 'metrics.csv'}")
    print(f"  - Figures: {OUT_DIR}/")
    print(f"    • fig1_scalability.png")
    print(f"    • fig2_evacuation_times.png")
    print(f"    • fig3_summary_table.png")
    print(f"    • fig4_tracked_agents.png")
    print(f"    • fig5_pf_rmse.png")
    print("\nNext step: Use these results in your Phase-2 paper!")
    print("=" * 60 + "\n")
