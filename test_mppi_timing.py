"""
test_mppi_timing.py
Directly times a single mppi.step() call with the real settings used
in the demo, to check whether MPPI's own cost calculation is slow
enough to cause the jumpy motion -- the same kind of blocking problem
GPMP2 had, just smaller.
"""
import time
import numpy as np
import mujoco
from robot.franka import DOF, FrankaModel
from planner.factor_graph import SignedDistanceField
from cbf.barrier import DistanceBarrier
from controller.mppi import MPPIController

obstacle_center = (0.20, 0.09, 0.85)
obstacle_radius = 0.08
d_safe = 0.03

model = mujoco.MjModel.from_xml_path("assets/panda.xml")
franka = FrankaModel(model, mujoco.MjData(model))
sdf = SignedDistanceField(np.array([obstacle_center]), np.array([obstacle_radius]))
barrier = DistanceBarrier(fk_fn=franka.fk, sphere_radii=franka.sphere_radii, sdf=sdf, dof=DOF, d_safe=d_safe)

mppi = MPPIController(lam=1.0, dt=0.05, dof=DOF, sdf=sdf, eps_margin=0.15,
                       sigma_obs=0.02, lambda_cbf=0.3, fk_batch_fn=franka.fk_batch,
                       sphere_radii=franka.sphere_radii)

N_horizon = 30
theta_ref = np.tile(np.zeros(DOF), (N_horizon + 1, 1))
Sigma = 0.002 ** 2 * np.eye(DOF)
K_inv_diag = np.ones((N_horizon + 1, DOF))
rng = np.random.default_rng(0)

def barrier_batch_fn(V):
    N_, T_, _ = V.shape
    h = np.zeros((N_, T_))
    for i in range(N_):
        for t in range(T_):
            x = np.concatenate([V[i, t], np.zeros(DOF)])
            h[i, t] = barrier.forward(x)
    return h

times = []
for i in range(10):
    t0 = time.time()
    result = mppi.step(theta_ref, Sigma, 200, K_inv_diag, barrier_batch_fn, rng)
    times.append(time.time() - t0)

print(f"MPPI step times (seconds): {[round(t, 4) for t in times]}")
print(f"Average: {np.mean(times):.4f}s   Max: {np.max(times):.4f}s")
print(f"MPPI thread target rate: 10 Hz = 0.1s per step")
print(f"Is MPPI's OWN computation taking longer than its own target period? "
      f"{'YES -- this is the real remaining bottleneck' if np.mean(times) > 0.1 else 'NO'}")

print("\\nTesting with fewer samples...")
for n_samples in [100, 50, 25]:
    times2 = []
    for i in range(5):
        t0 = time.time()
        result = mppi.step(theta_ref, Sigma, n_samples, K_inv_diag, barrier_batch_fn, rng)
        times2.append(time.time() - t0)
    print(f"n_samples={n_samples}: average={np.mean(times2):.4f}s")
