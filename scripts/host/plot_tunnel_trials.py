#!/usr/bin/env python3
"""
=================================================================================================
Script: plot_tunnel_trials.py
Plots trajectory and quantitative benchmark performance for 10-trial disaster recovery in Tunnel World.
=================================================================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_benchmark(json_path="/tmp/tunnel_trials_result.json", output_path="/home/lab/disaster_recovery_ws/docs/images/tunnel_disaster_benchmark_trajectories.png"):
    if not os.path.exists(json_path):
        print(f"[오류] 데이터 파일이 존재하지 않습니다: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    trials = data.get("trials", [])
    base_x = data.get("base_camp_x", 0.0)
    base_y = data.get("base_camp_y", 0.0)

    fig = plt.figure(figsize=(16, 8), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0])

    ax_traj = fig.add_subplot(gs[:, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_time = fig.add_subplot(gs[1, 1])

    # 1. Trajectory Map (2D)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(trials))))

    for idx, t in enumerate(trials):
        c = colors[idx]
        t_num = t["trial"]
        out_pts = np.array(t.get("outbound_path", []))
        ret_pts = np.array(t.get("return_path", []))

        if len(out_pts) > 1:
            ax_traj.plot(out_pts[:, 0], out_pts[:, 1], linestyle='--', color=c, alpha=0.6, linewidth=1.5, label=f"Trial #{t_num:02d} Out" if idx < 3 else None)
        if len(ret_pts) > 1:
            ax_traj.plot(ret_pts[:, 0], ret_pts[:, 1], linestyle='-', color=c, alpha=0.9, linewidth=2.0, label=f"Trial #{t_num:02d} Backtrack" if idx < 3 else None)

        if len(out_pts) > 0:
            ax_traj.scatter(out_pts[-1, 0], out_pts[-1, 1], marker='^', color='red', s=45, zorder=5)

        deb_pt = t.get("debris_pos")
        if deb_pt:
            circle_deb = patches.Circle((deb_pt[0], deb_pt[1]), 0.18, facecolor='crimson', edgecolor='black', alpha=0.75, zorder=6, label="Fallen Debris Obstacle" if idx == 0 else None)
            ax_traj.add_patch(circle_deb)

    # Base camp anchor and tolerance circles
    ax_traj.scatter(base_x, base_y, marker='*', s=260, color='gold', edgecolor='black', linewidth=1.5, zorder=10, label=f"Base Camp Anchor ({base_x:.2f}, {base_y:.2f})")
    circle_pass = patches.Circle((base_x, base_y), 0.25, fill=False, edgecolor='green', linestyle=':', linewidth=2.0, label="Pass Margin (< 25 cm)")
    ax_traj.add_patch(circle_pass)

    ax_traj.set_title(f"{len(trials)}-Trial Autonomous Deep Tunnel Trajectories (Dynamic Debris Drop & TD3 Bypass)\n(Dashed: Outbound Exploration | Solid: 4-Layer Backtracking Return | Crimson: Fallen Debris)", fontsize=12, fontweight='bold', pad=12)
    ax_traj.set_xlabel("X Position (meters)", fontsize=11)
    ax_traj.set_ylabel("Y Position (meters - Tunnel Depth)", fontsize=11)
    ax_traj.grid(True, linestyle='--', alpha=0.5)
    ax_traj.legend(loc='lower left', fontsize=9)
    ax_traj.axis('equal')

    # 2. Terminal Error Bar Chart
    trial_nums = [t["trial"] for t in trials]
    errs = [t["pos_err_cm"] for t in trials]
    times = [t["return_time_s"] for t in trials]
    successes = [t["success"] for t in trials]

    bar_colors = ['#2ecc71' if s else '#e74c3c' for s in successes]
    bars = ax_err.bar(trial_nums, errs, color=bar_colors, edgecolor='black', width=0.6)
    ax_err.axhline(25.0, color='red', linestyle='--', linewidth=1.8, label="Pass Threshold (25 cm)")

    succ_errs = [e for e, s in zip(errs, successes) if s]
    avg_err = np.mean(succ_errs) if succ_errs else 0.0
    ax_err.axhline(avg_err, color='blue', linestyle=':', linewidth=1.5, label=f"Mean Error: {avg_err:.2f} cm")

    for bar, err in zip(bars, errs):
        ax_err.text(bar.get_x() + bar.get_width()/2., err + 0.5, f"{err:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_err.set_title("Terminal Return Position Error (cm)", fontsize=12, fontweight='bold')
    ax_err.set_xlabel("Trial #", fontsize=10)
    ax_err.set_ylabel("Error from Base Camp (cm)", fontsize=10)
    ax_err.set_xticks(trial_nums)
    ax_err.set_ylim(0, max(30.0, max(errs) + 5.0 if errs else 30.0))
    ax_err.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax_err.legend(loc='upper right', fontsize=9)

    # 3. Return Elapsed Time Bar Chart
    time_bars = ax_time.bar(trial_nums, times, color='#3498db', edgecolor='black', width=0.6)
    succ_times = [tm for tm, s in zip(times, successes) if s]
    avg_time = np.mean(succ_times) if succ_times else 0.0
    ax_time.axhline(avg_time, color='navy', linestyle=':', linewidth=1.5, label=f"Mean Time: {avg_time:.1f} s")

    for bar, tm in zip(time_bars, times):
        ax_time.text(bar.get_x() + bar.get_width()/2., tm + 0.5, f"{tm:.1f}s", ha='center', va='bottom', fontsize=9)

    ax_time.set_title("Autonomous Backtracking Return Duration (seconds)", fontsize=12, fontweight='bold')
    ax_time.set_xlabel("Trial #", fontsize=10)
    ax_time.set_ylabel("Elapsed Time (s)", fontsize=10)
    ax_time.set_xticks(trial_nums)
    ax_time.set_ylim(0, max(35.0, max(times) + 6.0 if times else 35.0))
    ax_time.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax_time.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"[성공] 벤치마크 결과 그래프가 저장되었습니다: {output_path}")

    # Copy to Windows Desktop & OneDrive Desktop
    desktop_win = "/mnt/c/Users/lab/Desktop/tunnel_disaster_benchmark_trajectories.png"
    onedrive_win = "/mnt/c/Users/lab/OneDrive - 경북대학교/바탕 화면/tunnel_disaster_benchmark_trajectories.png"
    try:
        import shutil
        shutil.copyfile(output_path, desktop_win)
        print(f"[성공] 윈도우 바탕화면에 그래프 복사 완료: {desktop_win}")
    except Exception as e:
        print(f"[경고] 윈도우 바탕화면 복사 실패: {e}")

    try:
        import subprocess
        bat_content = (
            '@echo off\r\n'
            'copy /Y "C:\\Users\\lab\\Desktop\\tunnel_disaster_benchmark_trajectories.png" '
            '"C:\\Users\\lab\\OneDrive - 경북대학교\\바탕 화면\\tunnel_disaster_benchmark_trajectories.png" > nul\r\n'
        )
        bat_path = "/mnt/c/Users/lab/copy_plots.bat"
        with open(bat_path, "wb") as f:
            f.write(bat_content.encode("cp949"))
        subprocess.run(["cmd.exe", "/c", "C:\\Users\\lab\\copy_plots.bat"], stdin=subprocess.DEVNULL, timeout=5)
        if os.path.exists(bat_path):
            os.remove(bat_path)
        print(f"[성공] OneDrive 바탕화면에 그래프 복사 완료: {onedrive_win}")
    except Exception as e:
        print(f"[경고] OneDrive 바탕화면 복사 실패: {e}")

if __name__ == '__main__':
    plot_benchmark()
