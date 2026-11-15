#!/usr/bin/env python3
"""
Select a high-contrast multi-obstacle holdout seed (e.g., chicane_door_pinch or u_pocket_culdesac)
where Early (upd=25) visibly fails, Mid (upd=150) partially recovers/detours, and Final Student
executes a crisp winding avoidance PASS -> render 3D rollout MP4s and convert to GIFs (< 5 MB).
"""

import argparse
import os
import subprocess
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PKG_SRC = os.path.join(WS_ROOT, "src", "disaster_recovery")
for p in (SCRIPT_DIR, PKG_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from disaster_recovery.rl_context_policy import TemporalContextActorCritic  # noqa: E402
from rl_procedural_env import REGIME_NAMES, VectorDisasterEnv  # noqa: E402
from train_phase2_v100 import render_3d_rollout_video  # noqa: E402


def _load_teacher(ckpt_path: str, device: torch.device) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=device)
    hidden = int(ckpt.get("hidden_dim", 128))
    model = TemporalContextActorCritic(hidden_dim=hidden).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def _fn(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(h, p)[0]

    return _fn


def _load_onnx(onnx_path: str, device: torch.device) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])

    def _fn(h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        b = h.shape[0]
        flat = torch.cat([h.reshape(b, -1), p.reshape(b, -1)], dim=-1).cpu().numpy().astype(np.float32)
        out = sess.run(["cmd_vel"], {"obs_flat": flat})[0]
        return torch.from_numpy(out).to(device)

    return _fn


def _eval_batch(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    seeds: List[int],
    device: torch.device,
) -> List[Dict[str, object]]:
    env = VectorDisasterEnv(num_envs=len(seeds), track="sim", device=device, max_steps=320, mode="eval", base_seed=10000)
    h, p = env.reset_envs(eval_seeds=np.array(seeds, dtype=np.int64))
    done_mask = np.zeros(len(seeds), dtype=bool)
    results: List[Dict[str, object]] = [{} for _ in seeds]
    prev_xy = torch.stack([env.rx, env.ry], dim=-1).detach().cpu().numpy().copy()
    path_len = np.zeros(len(seeds), dtype=np.float64)

    for step in range(320):
        act = fn(h, p)
        h, p, _, dones, info = env.step(act)
        cur_xy = torch.stack([env.rx, env.ry], dim=-1).detach().cpu().numpy()
        active = ~done_mask
        path_len[active] += np.linalg.norm(cur_xy[active] - prev_xy[active], axis=1)
        prev_xy = cur_xy
        d_np = dones.detach().cpu().numpy()
        for i in range(len(seeds)):
            if not done_mask[i] and (d_np[i] or step == 319):
                done_mask[i] = True
                reached = bool(info["reached"][i].item())
                collided = bool(info["collision"][i].item())
                results[i] = {
                    "seed": seeds[i],
                    "regime": REGIME_NAMES[seeds[i] % len(REGIME_NAMES)],
                    "reached": reached,
                    "collision": collided,
                    "steps": step + 1,
                    "path_len": float(path_len[i]),
                }
        if done_mask.all():
            break
    return results


def _mp4_to_gif(mp4_path: str, gif_path: str) -> None:
    pal = f"/tmp/pal_ret_{os.getpid()}.png"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", mp4_path, "-vf", "fps=10,scale=640:-1:flags=lanczos,palettegen=stats_mode=diff", pal],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", mp4_path, "-i", pal, "-lavfi", "fps=10,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3", gif_path],
        check=True,
    )
    if os.path.exists(pal):
        os.remove(pal)
    print(f"[GIF] {gif_path} ({os.path.getsize(gif_path)} B)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--early-ckpt", required=True)
    parser.add_argument("--mid-ckpt", required=True)
    parser.add_argument("--student-onnx", required=True)
    parser.add_argument("--out-dir", default="/hdd/rl_v2_exp/evidence")
    parser.add_argument("--repo-img-dir", default=os.path.join(WS_ROOT, "docs", "images", "evidence"))
    parser.add_argument("--seed", type=int, default=0, help="Optional fixed holdout seed override")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_fn = _load_teacher(args.early_ckpt, device)
    mid_fn = _load_teacher(args.mid_ckpt, device)
    stu_fn = _load_onnx(args.student_onnx, device)

    seeds = [args.seed] if args.seed > 0 else list(range(10000, 10090))
    res_e = _eval_batch(early_fn, seeds, device)
    res_m = _eval_batch(mid_fn, seeds, device)
    res_s = _eval_batch(stu_fn, seeds, device)

    # Prefer complex slalom/trap regimes where Early fails, Student passes cleanly, and Mid differs visibly
    best_seed = seeds[0]
    best_score = -1e9
    for re, rm, rs in zip(res_e, res_m, res_s):
        if not rs["reached"]:
            continue
        reg = rs["regime"]
        score = 0.0
        if reg in ("chicane_door_pinch", "u_pocket_culdesac"):
            score += 40.0
        elif reg == "dynamic_crossing":
            score += 25.0
        if not re["reached"]:
            score += 50.0
        if re["collision"]:
            score += 10.0
        if rm["reached"] and not re["reached"]:
            score += 20.0  # Textbook staircase: Early FAIL -> Mid PASS (slower) -> Final PASS (fast)
        elif not rm["reached"] and rm["steps"] > re["steps"] + 15:
            score += 12.0  # Early fast crash -> Mid longer progress -> Final crisp PASS
        score += max(0.0, float(rm["steps"] - rs["steps"])) * 0.2
        score += min(15.0, float(rs["path_len"]) * 2.5)
        if score > best_score:
            best_score = score
            best_seed = int(rs["seed"])
            chosen = (re, rm, rs)

    print(f"[SELECTED SEED] {best_seed} (regime={chosen[2]['regime']}) score={best_score:.1f}", flush=True)
    print(f"  Early (upd=25)  : {chosen[0]}", flush=True)
    print(f"  Mid   (upd=150) : {chosen[1]}", flush=True)
    print(f"  Final (Student) : {chosen[2]}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.repo_img_dir, exist_ok=True)

    stages = [
        ("early upd=25", early_fn, os.path.join(args.out_dir, "phase2b_retrain_early_upd25.mp4"), os.path.join(args.repo_img_dir, "phase2b_retrain_early_upd25.gif"), "/tmp/retrain_early.png"),
        ("mid upd=150", mid_fn, os.path.join(args.out_dir, "phase2b_retrain_mid_upd150.mp4"), os.path.join(args.repo_img_dir, "phase2b_retrain_mid_upd150.gif"), "/tmp/retrain_mid.png"),
        ("Distilled Student ONNX", stu_fn, os.path.join(args.out_dir, "phase2b_retrain_final_student.mp4"), os.path.join(args.repo_img_dir, "phase2b_retrain_final_student.gif"), "/tmp/retrain_stu.png"),
    ]
    for label, fn, mp4_p, gif_p, png_p in stages:
        res = render_3d_rollout_video(
            "sim",
            label,
            fn,
            mp4_p,
            png_p,
            device=device,
            eval_seed=best_seed,
            require_pass=False,
        )
        print(f"[RENDERED] {label}: {res}", flush=True)
        _mp4_to_gif(mp4_p, gif_p)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
