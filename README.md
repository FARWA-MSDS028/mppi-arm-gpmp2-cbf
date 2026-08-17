# Feasibility-Coupled Layered Planning and Control
GPMP2 -> MPPI -> CBF-QP -> Feasibility -> Covariance Steering -> Conflict Factors -> iSAM2 -> GPMP2 (repeat)

## 0. What changed in this revision

- **Barrier function**: switched from a learned MLP to the analytic
  barrier you specified: **h(x) = d(x) - d_safe**, where d(x) is the
  minimum clearance (signed distance minus the robot's own sphere
  radius) between any robot collision sphere and the obstacle, computed
  directly via forward kinematics + the signed-distance field — no
  training data needed. Implemented in `cbf/barrier.py::DistanceBarrier`.
  It has the exact same `.forward(x)` / `.grad(x)` interface as the old
  MLP, so it drops in everywhere `lie_derivatives(barrier, x, f, g)` is
  called. The old learned-MLP code path (`MLPBarrier`) is still in the
  file if you want it later, but nothing uses it by default anymore.
- **DOF/actuator bug fixed**: the previous code assumed the arm's 7
  joints sit at `qpos[:7]` / `ctrl[:7]`. Real Panda MJCFs usually also
  define gripper joints/actuators, so `model.nu`/`model.nq` can be 8 or
  9, not 7 — meaning the old code was silently reading/writing the
  wrong joints. Fixed by resolving the 7 arm joints and 7 arm actuators
  **by name** (`robot/franka.py::resolve_arm_indices`, using MuJoCo's
  `model.joint(name)` / `model.actuator(name)`), used consistently in
  `robot/franka.py` and `robot/mujoco_env.py`.
- **Experiments are now actually wired together.** `main.py` only ever
  implemented the Stage-10 closed loop (Experiments 5-8) — it never
  called `run_baseline_mppi`, `run_gpmp2_mppi`, or `run_safe_mppi`, so
  running `main.py` alone could never reproduce Experiments 1-4. Added
  **`run_experiments.py`** at the project root: this is now the single
  command that runs all four required experiments (baseline MPPI,
  GPMP2+MPPI, MPPI+CBF, full adaptive loop) in sequence, or any one of
  them individually with `--only`.

## 1. Folder structure (what each file is for)

```
project/
  planner/
    factor_graph.py     # GP prior, obstacle hinge factor, start/goal factors (Step 1.1-1.3)
    gpmp2_planner.py     # LM optimization + marginal covariances (Step 1.4)
    conflict_factor.py   # Gaussian conflict factor + trigger condition (Block 5B)
    isam_update.py        # iSAM2 incremental updates (Step 5B.4)
  controller/
    sampling.py           # V_i = theta_GPMP2 + eps_i (Step 2.1)
    cost.py                 # running/terminal cost (Step 2.2)
    mppi.py                  # softmax weights + optimal control (Steps 2.3-2.4)
    covariance.py         # covariance steering update (Block 5A)
  cbf/
    barrier.py              # h(x) = d(x) - d_safe (DistanceBarrier) + Lie derivatives
    qp_solver.py           # CBF-QP via OSQP + unsafe detection (Step 3.3, Stage 4)
    feasibility.py         # extraction/storage of QP diagnostics (Block 4 / Stage 7)
  robot/
    franka.py                # FK/dynamics wrapper, arm-index resolution by name
    mujoco_env.py         # sim stepping + rendering (Stage 6), also name-resolved
  experiments/
    baseline_mppi.py    # Experiment 1
    gpmp2_mppi.py         # Experiment 2
    safe_mppi.py           # Experiments 3-4
    adaptive_loop.py    # Experiments 5-8 (wraps main.run_closed_loop)
  plots/
    main.py                   # all publication-quality figures
  main.py                       # Stage 10 closed loop ONLY (imported by adaptive_loop.py)
  run_experiments.py    # <-- ACTUAL entry point: runs all 4 experiments in sequence
  requirements.txt
```

## 2. Copy this project onto your Ubuntu VM

Pick ONE of these, whichever is easiest for you.

### Option A — you downloaded the .zip locally, upload it to the VM

From your **local machine's terminal** (not the VM):

```bash
scp gpmp2_mppi_cbf_project.zip your_username@your_vm_ip:/home/your_username/
```

Then, on the **VM**:

```bash
cd ~
unzip gpmp2_mppi_cbf_project.zip -d project
cd project
```

If your VM is on a cloud provider console (no direct `scp` access), upload
the zip through the provider's file-upload UI into the VM's home
directory, or drop it into a synced folder (Google Drive/Dropbox
desktop client mounted on the VM), then `unzip` it there.

### Option B — no file transfer available, paste file-by-file

If you truly can only copy-paste text into the VM's terminal (e.g. a
bare SSH session with no file transfer), recreate each file with a
heredoc. On the VM:

```bash
mkdir -p ~/project/planner ~/project/controller ~/project/cbf ~/project/robot ~/project/experiments ~/project/plots
cd ~/project
cat > planner/factor_graph.py << 'EOF'
<paste the exact contents of that file here>
EOF
```

Repeat `cat > <path> << 'EOF' ... EOF` for every file in the tree above
(the `'EOF'` with quotes stops the shell from expanding `$`, backticks,
etc. inside the pasted Python code — important, don't drop the quotes).
This works but is tedious; Option A is strongly preferred for ~26 files.

### Option C — push to GitHub/GitLab from local, clone on the VM

```bash
# locally, inside the unzipped project/
git init && git add -A && git commit -m "initial"
git remote add origin <your empty repo url>
git push -u origin main

# on the VM
git clone <your repo url> project
cd project
```

## 3. Install dependencies (on the VM, inside `project/`)

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev build-essential
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `pip install gtsam` fails for your Ubuntu/Python combo, build it
from source: https://github.com/borglab/gtsam with `-DGTSAM_BUILD_PYTHON=ON`.
`mujoco` and `osqp` install cleanly from PyPI in almost all cases.

## 4. Get the Franka Panda MuJoCo model

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie
mkdir -p assets
cp /tmp/menagerie/franka_emika_panda/panda.xml assets/panda.xml
cp -r /tmp/menagerie/franka_emika_panda/assets assets/assets
```

Then check the joint/actuator names actually match what
`robot/franka.py` expects (`joint1..joint7`, `actuator1..actuator7`):

```bash
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('assets/panda.xml')
print('joints:', [m.joint(i).name for i in range(m.njnt)])
print('actuators:', [m.actuator(i).name for i in range(m.nu)])
"
```

If the printed names differ (e.g. `panda_joint1` instead of `joint1`),
edit `ARM_JOINT_NAMES` / `ARM_ACTUATOR_NAMES` at the top of
`robot/franka.py` to match — this is a one-line fix per list, and it's
exactly why the resolution is name-based rather than positional.

## 5. Sequence of running files (do this in order)

Do **not** run `main.py` first — it only contains the closed-loop
piece (Experiments 5-8). Run `run_experiments.py`, which calls
everything in the right order:

```bash
cd ~/project
source venv/bin/activate

# Step 1 — sanity check imports before running anything numerically:
python -c "import gtsam, mujoco, osqp; print('deps OK')"
python -c "from robot.mujoco_env import MujocoFrankaEnv; MujocoFrankaEnv(mjcf_path='assets/panda.xml'); print('mjcf OK')"

# Step 2 — run everything, in the paper's required order:
#   (1) baseline MPPI -> (2) GPMP2+MPPI -> (3) MPPI+CBF -> (4) full closed loop
python run_experiments.py --mjcf assets/panda.xml --cycles 5

# Step 3 (optional) — rerun just one experiment while debugging, e.g.:
python run_experiments.py --mjcf assets/panda.xml --only baseline
python run_experiments.py --mjcf assets/panda.xml --only gpmp2_mppi
python run_experiments.py --mjcf assets/panda.xml --only safe_mppi
python run_experiments.py --mjcf assets/panda.xml --only full --cycles 10
```

Each run prints progress per experiment and saves figures
(`fig_exp1_trajectory.png`, `fig_exp2_cost_convergence.png`,
`fig_exp3_4_feasibility.png`, `fig_exp5_feasibility_full.png`,
`fig_exp6_covariance.png`) into the current directory.

Start with a **small** `--cycles` value (1-2) the very first time you
run the full loop — it's the slowest and most likely to surface a
model-specific bug (wrong joint names, wrong obstacle placement, etc.)
before you commit to a long run.

## 6. Reproducibility notes

- All stochastic components take an explicit `rng_seed` — no hidden
  global RNG state.
- `planner/gpmp2_planner.py` uses GTSAM's `LevenbergMarquardtParams`
  with `RelativeErrorTol=1e-6`; report `iterations`/`final_error` from
  `GPMP2Result` alongside any numbers you publish.
- `cbf/barrier.py::DistanceBarrier` needs NO offline training — it's
  the analytic h(x) = d(x) - d_safe you specified. `d_safe` (default
  0.05 m) is the one tunable parameter; raise it for a more
  conservative controller.
- Every module docstring states time/memory complexity per the spec's
  "Code Quality" requirement.

## 7. What's implemented vs. what's still a configuration/tuning surface

Implemented exactly per your equations: GP prior + obstacle hinge
factors, GPMP2 MAP inference, GPMP2-mean MPPI sampling, MPPI cost/
softmax/weighted-control, CBF-QP (OSQP), unsafe-controller pre-QP
detection, feasibility extraction, covariance steering (both the
windowed and single-sample rules), conflict factors (both
formulations), iSAM2 incremental insertion, and the analytic barrier
h(x) = d(x) - d_safe with correctly-name-resolved robot joints/actuators.

Left as configuration/tuning surface (not a math simplification):
obstacle placement/shape, collision-sphere layout/radii, `d_safe`, QP
joint-limit bounds, MPPI temperature `lambda` and sample count `N`,
covariance-steering `eta`/`beta`/window `W`, conflict thresholds
`tau_conflict`/`tau_safe`.
