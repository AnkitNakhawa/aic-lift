"""Train the Phase 2 (insertion) SAC policy.

Usage:
    python -m aic_example_policies.training.train_insertion \\
        --scene /path/to/scene.xml \\
        --port_type sfp \\
        --total_steps 500000 \\
        --save_dir checkpoints/phase2

No cameras. Observation is 86-D flat proprio.
Curriculum ramps XY offset from 0 → xy_offset_max over the first
curriculum_steps training steps (default 200k).
"""

import argparse
import os
import time

import numpy as np
import torch

from .insertion_env import InsertionConfig, InsertionEnv
from .networks import P2_OBS_DIM
from .sac_trainer import ReplayBuffer, SACTrainer, DEVICE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--port_type", default="sfp", choices=["sfp", "sc"])
    p.add_argument("--total_steps", type=int, default=500_000)
    p.add_argument("--buffer_size", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--warmup_steps", type=int, default=1_000)
    p.add_argument("--save_dir", default="checkpoints/phase2")
    p.add_argument("--save_interval", type=int, default=50_000)
    p.add_argument("--log_interval", type=int, default=1_000)
    p.add_argument("--xy_offset_max", type=float, default=0.005,
                   help="Max XY curriculum offset at full difficulty (m)")
    p.add_argument("--curriculum_steps", type=int, default=200_000,
                   help="Steps over which XY offset ramps from 0 to xy_offset_max")
    p.add_argument("--load", default=None, help="Resume from checkpoint")
    return p.parse_args()


def _to_device_obs(obs: np.ndarray, device: torch.device) -> dict:
    """Phase 2: obs is a flat (86,) ndarray → {'proprio': (1, 86) tensor}."""
    return {"proprio": torch.from_numpy(obs).unsqueeze(0).float().to(device)}


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — training on CPU will be slow.")
    else:
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    cfg = InsertionConfig(
        scene_path=args.scene,
        port_type=args.port_type,
        xy_offset_max=args.xy_offset_max,
        xy_offset_curriculum=0.0,  # starts at perfect alignment
    )
    env = InsertionEnv(cfg)

    buffer = ReplayBuffer(
        capacity=args.buffer_size,
        phase=2,
        proprio_dim=P2_OBS_DIM,
    )
    trainer = SACTrainer(phase=2, device=DEVICE)
    if args.load:
        trainer.load(args.load)
        print(f"Resumed from {args.load} (step {trainer._train_steps})")

    os.makedirs(args.save_dir, exist_ok=True)

    obs, _ = env.reset()
    total_steps = 0
    episode_reward = 0.0
    episode_count = 0
    ep_rewards: list = []
    loss_log: dict = {}
    t0 = time.time()

    print(f"Phase 2 training. Proprio dim={P2_OBS_DIM}. Device={DEVICE}. "
          f"Total steps={args.total_steps}")

    while total_steps < args.total_steps:
        # ── Curriculum update ──────────────────────────────────────────────
        curriculum_frac = min(1.0, total_steps / max(1, args.curriculum_steps))
        env.cfg.xy_offset_curriculum = curriculum_frac

        # ── Action selection ───────────────────────────────────────────────
        if total_steps < args.warmup_steps:
            action = env.action_space.sample()
        else:
            obs_t = _to_device_obs(obs, DEVICE)
            action = trainer.actor.get_action(obs_t).squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(obs, action, reward, next_obs, float(done))
        obs = next_obs
        episode_reward += reward
        total_steps += 1

        if done:
            ep_rewards.append(episode_reward)
            episode_count += 1
            episode_reward = 0.0
            obs, _ = env.reset()

        if total_steps >= args.warmup_steps and len(buffer) >= args.batch_size:
            loss_log = trainer.update(buffer, args.batch_size)

        if total_steps % args.log_interval == 0 and ep_rewards:
            elapsed = time.time() - t0
            mean_r = np.mean(ep_rewards[-20:])
            print(
                f"[{total_steps:>7d}] ep={episode_count:>5d} "
                f"mean_r={mean_r:+7.2f} "
                f"critic={loss_log.get('critic_loss', 0):.4f} "
                f"actor={loss_log.get('actor_loss', 0):.4f} "
                f"alpha={loss_log.get('alpha', 0):.3f} "
                f"curriculum={curriculum_frac:.2f} "
                f"fps={total_steps / elapsed:.0f}"
            )

        if total_steps % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"step_{total_steps}.pt")
            trainer.save(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    trainer.save(os.path.join(args.save_dir, "final.pt"))
    print("Training complete.")
    env.close()


if __name__ == "__main__":
    main()
