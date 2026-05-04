"""Train the Phase 2 (insertion) SAC policy using the live Gazebo simulation.

Unlike train_insertion.py (MuJoCo), this script requires the eval container to be
running and trains at real-time simulation speed.  The SAC algorithm, network
architecture, and checkpoint format are identical; Gazebo-trained checkpoints are
fully compatible with LocalPrecisionGazeboPolicy for deployment.

Usage:
    # Terminal 1 — start eval container with ground truth
    ./scripts/start_eval.sh ground_truth:=true start_aic_engine:=false gazebo_gui:=false

    # Terminal 2 — run training (Phase 1 must be complete first)
    pixi run python -m aic_example_policies.training.train_insertion_gz \\
        --port_type sfp \\
        --port_frame task_board/nic_card_mount_0/sfp_port_0_link \\
        --save_dir checkpoints/gazebo/phase2

Note: Gazebo training is ~100× slower than MuJoCo (real-time vs. accelerated sim).
"""

import argparse
import os
import time

import numpy as np
import torch

from .gazebo_insertion_env import GazeboInsertionConfig, GazeboInsertionEnv
from .image_utils import IMG_SIZE
from .sac_trainer import ReplayBuffer, SACTrainer, DEVICE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port_type", default="sfp", choices=["sfp", "sc"])
    p.add_argument(
        "--port_frame",
        default="task_board/nic_card_mount_0/sfp_port_0_link",
        help="TF frame of the port (requires ground_truth:=true)",
    )
    p.add_argument("--total_steps", type=int, default=100_000,
                   help="Total env steps (default lower than MuJoCo due to real-time speed)")
    p.add_argument("--buffer_size", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--save_dir", default="checkpoints/gazebo/phase2")
    p.add_argument("--save_interval", type=int, default=10_000)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--det_loss_weight", type=float, default=0.1)
    p.add_argument("--load", default=None, help="Resume from checkpoint")
    return p.parse_args()


def _to_device_obs(obs: dict, device: torch.device) -> dict:
    out = {}
    for k, v in obs.items():
        if k == "proprio":
            out[k] = torch.from_numpy(v).unsqueeze(0).float().to(device)
        else:
            out[k] = (
                torch.from_numpy(v.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                / 255.0
            )
    return out


def main():
    args = parse_args()

    cfg = GazeboInsertionConfig(
        port_type=args.port_type,
        port_frame=args.port_frame,
    )
    env = GazeboInsertionEnv(cfg)

    buffer = ReplayBuffer(
        capacity=args.buffer_size,
        phase=2,
        img_shape=(IMG_SIZE, IMG_SIZE, 4),
        proprio_dim=13,
    )
    trainer = SACTrainer(
        phase=2,
        det_loss_weight=args.det_loss_weight,
        device=DEVICE,
    )
    if args.load:
        trainer.load(args.load)
        print(f"Resumed from {args.load} (step {trainer._train_steps})")

    os.makedirs(args.save_dir, exist_ok=True)

    obs, _ = env.reset()
    total_steps = 0
    episode_reward = 0.0
    episode_count = 0
    ep_rewards = []
    loss_log: dict = {}
    t0 = time.time()

    print(
        f"Training Phase 2 in Gazebo on {DEVICE}. "
        f"Total steps: {args.total_steps} (real-time)"
    )

    while total_steps < args.total_steps:
        if total_steps < args.warmup_steps:
            action = env.action_space.sample()
        else:
            obs_t = _to_device_obs(obs, DEVICE)
            action = trainer.actor.get_action(obs_t).squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        gt_uv = info.get("gt_uv")
        buffer.add(obs, action, reward, next_obs, float(done), gt_uv=gt_uv)
        obs = next_obs
        if done:
            obs, _ = env.reset()

        total_steps += 1
        episode_reward += reward
        if done:
            ep_rewards.append(episode_reward)
            episode_count += 1
            episode_reward = 0.0

        if total_steps >= args.warmup_steps and len(buffer) >= args.batch_size:
            loss_log = trainer.update(buffer, args.batch_size)

        if total_steps % args.log_interval == 0 and ep_rewards:
            elapsed = time.time() - t0
            mean_r = np.mean(ep_rewards[-20:])
            print(
                f"[{total_steps:>7d}] ep={episode_count:>5d} "
                f"mean_r={mean_r:+6.2f} "
                f"critic={loss_log.get('critic_loss', 0):.4f} "
                f"actor={loss_log.get('actor_loss', 0):.4f} "
                f"det={loss_log.get('det_loss', 0):.4f} "
                f"alpha={loss_log.get('alpha', 0):.3f} "
                f"fps={total_steps / elapsed:.1f}"
            )

        if total_steps % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"step_{total_steps}.pt")
            trainer.save(ckpt_path)
            print(f"  Saved: {ckpt_path}")

    trainer.save(os.path.join(args.save_dir, "final.pt"))
    print("Gazebo Phase 2 training complete.")
    env.close()


if __name__ == "__main__":
    main()
