# LocalPrecisionPolicy — Script Workflow

## Stage 0 — Machine Setup *(run once, ever)*

**`setup_cloud.sh`**

- Installs Docker, NVIDIA Container Toolkit, Pixi
- Pulls the `aic_eval` Docker image
- Writes Zenoh env vars to `~/.bashrc`
- Runs `pixi install`

```bash
./scripts/setup_cloud.sh
source ~/.bashrc
```

> Use on a fresh machine, new cloud instance, or first time on this repo.

### ARM / linux-aarch64 machines

`pixi.toml` only declares `linux-64` and `osx-arm64` as supported platforms, so `pixi install` will fail on ARM Linux with an `unsupported-platform` error. Training and testing do not need ROS or pixi — bypass pixi entirely by installing the training deps directly and setting `USE_SYSTEM_PYTHON=1`.

**One-time setup — create an isolated venv outside the repo:**
```bash
python3 -m venv ~/.venvs/aic-training
source ~/.venvs/aic-training/bin/activate
pip install "mujoco==3.5.0" torch gymnasium opencv-python "numpy<2.3" transforms3d
```

**Each new terminal session — activate the venv, then train/test:**
```bash
source ~/.venvs/aic-training/bin/activate
export USE_SYSTEM_PYTHON=1
./scripts/run_local_precision.sh train-p1 --scene aic_utils/aic_mujoco/mjcf/scene.xml --save_dir checkpoints/phase1
```

The venv lives in `~/.venvs/` — nothing inside the repo is touched. The `deploy` subcommand still requires a `linux-64` machine since it needs the ROS/pixi environment.

---

## Stage 1 — Training *(no eval container needed)*

Both phases train entirely in MuJoCo headlessly — no Docker required.

### Phase 1 — XY Centering

**`run_local_precision.sh train-p1`**

| | |
|---|---|
| **Input** | `scene.xml` + random XY offsets (1–3 cm from port) |
| **Learns** | `(dx, dy)` to center plug over port using center wrist camera |
| **Output** | `checkpoints/phase1/final.pt` |
| **Physics** | Kinematic IK only (fast) |

```bash
./scripts/run_local_precision.sh train-p1 \
    --scene aic_utils/aic_mujoco/mjcf/scene.xml \
    --total_steps 500000 \
    --save_dir checkpoints/phase1
```

### Phase 2 — F/T-Guided Insertion

**`run_local_precision.sh train-p2`**

| | |
|---|---|
| **Input** | `scene.xml` + centered start pose (Phase 1 assumed complete) |
| **Learns** | `(dx, dy, dz)` descent guided by 3 cameras + 6-DOF F/T sensor |
| **Output** | `checkpoints/phase2/final.pt` |
| **Physics** | Full `mj_step` with 5 substeps — cable contact forces matter |

```bash
./scripts/run_local_precision.sh train-p2 \
    --scene aic_utils/aic_mujoco/mjcf/scene.xml \
    --total_steps 500000 \
    --save_dir checkpoints/phase2
```

> **Run Phase 1 before Phase 2.** Phase 2 assumes the plug starts centered; training it before Phase 1 converges produces poor results.

**To resume a crashed run:**
```bash
./scripts/run_local_precision.sh train-p1 \
    --scene aic_utils/aic_mujoco/mjcf/scene.xml \
    --load checkpoints/phase1/step_250000.pt
```

**To run overnight (both phases chained):**
```bash
mkdir -p logs
nohup bash -c '
  ./scripts/run_local_precision.sh train-p1 \
      --scene aic_utils/aic_mujoco/mjcf/scene.xml \
      --save_dir checkpoints/phase1 \
  && \
  ./scripts/run_local_precision.sh train-p2 \
      --scene aic_utils/aic_mujoco/mjcf/scene.xml \
      --save_dir checkpoints/phase2
' > logs/overnight_train.log 2>&1 &

echo "Training PID: $!"
tail -f logs/overnight_train.log   # monitor from another terminal
```

Checkpoints save every 50k steps to `checkpoints/phaseN/step_N.pt`.

---

## Stage 2 — Headless Testing *(no eval container needed)*

**`run_local_precision.sh test`**

Loads a checkpoint, runs N full episodes in MuJoCo sim, and prints per-episode stats. Use this to validate a checkpoint before deploying.

```bash
# Test Phase 1
./scripts/run_local_precision.sh test \
    --scene aic_utils/aic_mujoco/mjcf/scene.xml \
    --phase 1 \
    --ckpt checkpoints/phase1/final.pt

# Test Phase 2
./scripts/run_local_precision.sh test \
    --scene aic_utils/aic_mujoco/mjcf/scene.xml \
    --phase 2 \
    --ckpt checkpoints/phase2/final.pt \
    --episodes 5
```

**What good output looks like:**

| Phase | Signal | Target |
|---|---|---|
| Phase 1 | `xy_err` | < 0.004 m by step ~100 |
| Phase 1 | `reward` | Trending toward 0 |
| Phase 2 | `depth` | > 0.015 m |
| Phase 2 | `ft_mag` | < 10 N |
| Phase 2 | `reward` | > +40 |

---

## Stage 3 — Live Deployment *(eval container must be running)*

### Terminal 1 — Start the eval environment

**`start_eval.sh`**

```bash
# Default: headless, ground_truth=true, sfp+sc cable
./scripts/start_eval.sh

# With options
./scripts/start_eval.sh ground_truth:=false gazebo_gui:=true launch_rviz:=true
```

Leave this running — it is the simulator Docker container.

### Terminal 2 — Run the policy

**`run_local_precision.sh deploy`**

```bash
./scripts/run_local_precision.sh deploy \
    --p1-ckpt checkpoints/phase1/final.pt \
    --p2-ckpt checkpoints/phase2/final.pt
```

The policy runs three phases in sequence: **Orient → Center (P1) → Insert (P2)**.

**`run_policy.sh`** — generic runner for any policy class

```bash
# Smoke test with WaveArm to confirm the sim is up
./scripts/run_policy.sh aic_example_policies.ros.WaveArm

# Run your policy by class path
./scripts/run_policy.sh aic_example_policies.ros.LocalPrecisionPolicy
```

---

## Full Training Run — Recommended Order

```
1.  ./scripts/setup_cloud.sh               ← once per machine
2.  source ~/.bashrc
3.  run_local_precision.sh train-p1        ← overnight if needed
4.  run_local_precision.sh test --phase 1
        xy_err < 4 mm? ──► proceed
        still high?    ──► more steps or tune reward
5.  run_local_precision.sh train-p2        ← overnight
6.  run_local_precision.sh test --phase 2
        depth > 15 mm and ft_mag < 10 N? ──► proceed
7.  start_eval.sh                          ← Terminal 1
8.  run_local_precision.sh deploy          ← Terminal 2
```

---

## Gazebo-Based Training (No MuJoCo Required)

The `train-gz-p1` / `train-gz-p2` subcommands train the same SAC architecture
directly inside the running Gazebo simulation rather than in MuJoCo.  This is
useful when:

- You want to skip the MuJoCo dependency entirely
- You want to train on the exact same physics and sensors used at evaluation
- You are on a platform where MuJoCo is unavailable

**Trade-offs vs. MuJoCo training:**

| | MuJoCo | Gazebo |
|---|---|---|
| Sim speed | 50–200× real-time | 1× real-time |
| Docker needed during training | No | Yes |
| Fidelity to eval sensors | Approximate | Exact |
| Episode reset cost | Instant | ~4 s robot move |
| `scene.xml` required | Yes | No |

Because Gazebo runs at real-time speed, reduce `--total_steps` by roughly 10×
compared to the MuJoCo defaults, or run overnight with a larger budget.

### Stage 0 — Machine Setup

Same as the MuJoCo workflow (`setup_cloud.sh`).

### Stage 1 — Terminal 1: Start the eval container

```bash
# ground_truth:=true is required so TF port poses are available for reward
./scripts/start_eval.sh ground_truth:=true start_aic_engine:=false gazebo_gui:=false
```

Leave this running for the entire training session.

### Stage 2 — Terminal 2: Train Phase 1 (XY centering)

**`run_local_precision.sh train-gz-p1`**

```bash
./scripts/run_local_precision.sh train-gz-p1 \
    --port_type sfp \
    --port_frame task_board/nic_card_mount_0/sfp_port_0_link \
    --total_steps 100000 \
    --save_dir checkpoints/gazebo/phase1
```

> **`--port_frame`** must match the TF frame of the port you are training on.
> Inspect the available frames with `pixi run ros2 run tf2_tools view_frames` while
> the eval container is running with `ground_truth:=true`.

**What to watch:**

| Signal | Target |
|---|---|
| `xy_err` | < 0.004 m by step ~30 k |
| `reward` | Trending toward 0 |

### Stage 3 — Terminal 2: Train Phase 2 (F/T insertion)

**`run_local_precision.sh train-gz-p2`**

```bash
./scripts/run_local_precision.sh train-gz-p2 \
    --port_type sfp \
    --port_frame task_board/nic_card_mount_0/sfp_port_0_link \
    --total_steps 100000 \
    --save_dir checkpoints/gazebo/phase2
```

> Run Phase 1 first. Phase 2 assumes the plug starts centered above the port.

**What to watch:**

| Signal | Target |
|---|---|
| `depth` | > 0.015 m |
| `ft_mag` | < 10 N |
| `reward` | > +40 |

**To resume a crashed Gazebo run:**

```bash
./scripts/run_local_precision.sh train-gz-p1 \
    --port_frame task_board/nic_card_mount_0/sfp_port_0_link \
    --load checkpoints/gazebo/phase1/step_50000.pt
```

**To run both phases overnight:**

```bash
mkdir -p logs
nohup bash -c '
  ./scripts/run_local_precision.sh train-gz-p1 \
      --port_frame task_board/nic_card_mount_0/sfp_port_0_link \
      --save_dir checkpoints/gazebo/phase1 \
  && \
  ./scripts/run_local_precision.sh train-gz-p2 \
      --port_frame task_board/nic_card_mount_0/sfp_port_0_link \
      --save_dir checkpoints/gazebo/phase2
' > logs/overnight_gz_train.log 2>&1 &
echo "Training PID: $!"
tail -f logs/overnight_gz_train.log
```

### Stage 4 — Deploy with Gazebo-trained checkpoints

**Terminal 1** — keep the eval container running (or restart without ground_truth):

```bash
./scripts/start_eval.sh
```

**Terminal 2** — deploy `LocalPrecisionGazeboPolicy`:

```bash
./scripts/run_local_precision.sh deploy-gz \
    --p1-ckpt checkpoints/gazebo/phase1/final.pt \
    --p2-ckpt checkpoints/gazebo/phase2/final.pt
```

Or via `run_policy.sh` with env vars:

```bash
export AIC_GZ_PHASE1_CKPT=checkpoints/gazebo/phase1/final.pt
export AIC_GZ_PHASE2_CKPT=checkpoints/gazebo/phase2/final.pt
./scripts/run_policy.sh aic_example_policies.ros.LocalPrecisionGazeboPolicy
```

### Gazebo Full Run — Recommended Order

```
1.  ./scripts/setup_cloud.sh                   ← once per machine
2.  source ~/.bashrc
3.  start_eval.sh ground_truth:=true …         ← Terminal 1, keep running
4.  run_local_precision.sh train-gz-p1         ← Terminal 2, overnight if needed
5.  run_local_precision.sh train-gz-p2         ← Terminal 2, after Phase 1 converges
6.  start_eval.sh                              ← Terminal 1, restart without ground_truth
7.  run_local_precision.sh deploy-gz           ← Terminal 2
```

### Gazebo Quick Reference

| Subcommand | Eval container needed? | ground_truth? | Checkpoint needed? |
|---|---|---|---|
| `train-gz-p1` | Yes | Yes | No (or `--load`) |
| `train-gz-p2` | Yes | Yes | No (or `--load`) |
| `deploy-gz` | Yes | No | Yes |

---

## Quick Reference (MuJoCo)

| Script | Docker needed? | `scene.xml` needed? | Checkpoint needed? |
|---|---|---|---|
| `setup_cloud.sh` | No | No | No |
| `start_eval.sh` | Yes (pulls image) | No | No |
| `run_policy.sh` | Yes* | No | No |
| `train-p1` / `train-p2` | No | Yes | No (or `--load`) |
| `test` | No | Yes | Yes (or random) |
| `deploy` | Yes* | No | Yes |

\* Eval Docker container must already be running via `start_eval.sh`.
