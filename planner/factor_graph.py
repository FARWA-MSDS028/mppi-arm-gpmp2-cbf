"""
planner/factor_graph.py
========================
Implements Block 1 (Steps 1.1-1.3) of the specification:

  - Constant-velocity GP prior (Section IV-A, GPMP2):  xi_ddot = w(t)
  - State transition matrix  Phi(t,s)
  - Noise covariance  Q_i
  - GP prior factor            f_gp(theta_{i-1}, theta_i)
  - Obstacle hinge-loss factor f_obs(theta_i)
  - Start/prior factor         f_0(theta_0)

All factors are implemented as gtsam.CustomFactor objects so that the
resulting nonlinear factor graph is solved with GTSAM's Gauss-Newton /
Levenberg-Marquardt optimizer exactly as in Step 1.4, and so the SAME
graph object can later be extended incrementally with iSAM2
(planner/isam_update.py) when Conflict Factors are inserted (Block 5B).

State convention
-----------------
Each pose in the trajectory is theta_i = [q_i; qdot_i] in R^{2*dof}
(GPMP2's "augmented state" -- constant-velocity GP prior is defined on
this augmented state, matching Section IV-A exactly).  For the Franka
Panda dof = 7, so theta_i in R^14.
"""

from __future__ import annotations
import numpy as np
import gtsam
from gtsam import CustomFactor, noiseModel
from gtsam.symbol_shorthand import X

# --------------------------------------------------------------------------
# Step 1.2: constant-velocity GP prior system matrices
# --------------------------------------------------------------------------

def transition_matrix(dt: float, dof: int) -> np.ndarray:
    """
    Phi(t_i, t_{i-1}) for the white-noise-on-acceleration model.

        Phi(t,s) = [[I, (t-s) I],
                    [0,     I  ]]

    Inputs
    ------
    dt  : t_i - t_{i-1}
    dof : number of joints (7 for Panda)

    Returns
    -------
    (2*dof, 2*dof) ndarray
    """
    I = np.eye(dof)
    Z = np.zeros((dof, dof))
    top = np.hstack([I, dt * I])
    bot = np.hstack([Z, I])
    return np.vstack([top, bot])


def noise_covariance_Qi(dt: float, dof: int, Qc: np.ndarray) -> np.ndarray:
    """
    Q_i = [[ 1/3 dt^3 Qc , 1/2 dt^2 Qc ],
           [ 1/2 dt^2 Qc ,     dt   Qc ]]

    Qc : (dof,dof) power spectral density of the white-noise acceleration.

    Returns (2*dof, 2*dof) ndarray. Time complexity O(dof^2) to assemble.
    """
    top = np.hstack([(dt**3 / 3.0) * Qc, (dt**2 / 2.0) * Qc])
    bot = np.hstack([(dt**2 / 2.0) * Qc, dt * Qc])
    Q = np.vstack([top, bot])
    return Q


# --------------------------------------------------------------------------
# Step 1.3, Factor Type 1: Prior / start factor  f_0(theta_0)
#   e_0 = mu_0 - theta_0
# --------------------------------------------------------------------------

def make_start_factor(key0, theta0_mean: np.ndarray, K0: np.ndarray):
    """
    Gaussian prior factor pinning the first pose to theta0_mean with
    covariance K0.  Returns a gtsam.PriorFactorVector (native GTSAM
    type -- exact for the linear-Gaussian case, no custom code needed).
    """
    noise = noiseModel.Gaussian.Covariance(K0)
    return gtsam.PriorFactorVector(key0, theta0_mean, noise)


def make_goal_factor(keyN, thetaN_goal: np.ndarray, KN: np.ndarray):
    """Goal factor: same functional form as the start factor, applied
    to the final pose theta_N (Step 1.3 lists Start/Goal as boundary
    priors of the same type)."""
    noise = noiseModel.Gaussian.Covariance(KN)
    return gtsam.PriorFactorVector(keyN, thetaN_goal, noise)


# --------------------------------------------------------------------------
# Step 1.3, Factor Type 2: GP prior (smoothness) factor
#   f_gp(theta_{i-1}, theta_i) = exp{ -1/2 || theta_i - Phi theta_{i-1} ||^2_{Q_i} }
#   (u_i, the exogenous mean-drift integral, is 0 for the white-noise
#    acceleration model with zero-mean control input, per Sec. IV-A.)
# --------------------------------------------------------------------------

def _gp_error_and_jacobians(Phi: np.ndarray):
    """
    Builds the CustomFactor error function for the GP prior factor.

    error(theta_{i-1}, theta_i) = theta_i - Phi @ theta_{i-1}

    Jacobians:
        d(error)/d(theta_{i-1}) = -Phi
        d(error)/d(theta_i)     =  I
    """
    dim = Phi.shape[0]

    def error_func(this: CustomFactor, values: gtsam.Values, jacobians):
        key_im1 = this.keys()[0]
        key_i = this.keys()[1]
        th_im1 = values.atVector(key_im1)
        th_i = values.atVector(key_i)
        err = th_i - Phi @ th_im1
        if jacobians is not None:
            jacobians[0] = -Phi
            jacobians[1] = np.eye(dim)
        return err

    return error_func


def make_gp_prior_factor(key_im1, key_i, dt: float, dof: int, Qc: np.ndarray):
    """
    Constructs the GP smoothness factor f_gp(theta_{i-1}, theta_i).

    Time complexity: O(dof^3) once (Cholesky of Q_i for the noise model).
    Memory: O(dof^2).
    """
    Phi = transition_matrix(dt, dof)
    Qi = noise_covariance_Qi(dt, dof, Qc)
    noise = noiseModel.Gaussian.Covariance(Qi)
    err_fn = _gp_error_and_jacobians(Phi)
    return CustomFactor(noise, [key_im1, key_i], err_fn)


# --------------------------------------------------------------------------
# Step 1.3, Factor Type 3: Obstacle factor (hinge loss)
#   c(z) = eps - d(z)  if d(z) <= eps   else 0
#   h(theta_i) = [ c(x(theta_i, s_j)) ]_{j=1..M}
# --------------------------------------------------------------------------

class SignedDistanceField:
    """
    Minimal SDF wrapper. d(z) = signed distance from workspace point z
    to the nearest obstacle surface (positive outside, negative inside).

    In production this samples a precomputed voxel grid (as in GPMP2);
    here we support either (a) a precomputed grid with trilinear
    interpolation, or (b) analytic spheres for fast synthetic testing.
    """

    def __init__(self, obstacle_centers: np.ndarray, obstacle_radii: np.ndarray):
        self.centers = np.asarray(obstacle_centers, dtype=float)  # (M_obs, 3)
        self.radii = np.asarray(obstacle_radii, dtype=float)      # (M_obs,)

    def distance(self, p: np.ndarray, robot_radius: float = 0.0) -> float:
        """Signed CLEARANCE from workspace point p (center of a robot
        collision sphere of radius `robot_radius`) to the nearest sphere
        obstacle surface: ||p - c|| - r_obstacle - r_robot. Subtracting
        the robot sphere's own radius (not just the obstacle's) is
        required for a correct swept-sphere collision check -- a point
        exactly on the obstacle surface with robot_radius=0.05 is
        already 5cm into collision, not at zero clearance."""
        diffs = p[None, :] - self.centers
        dists = np.linalg.norm(diffs, axis=1) - self.radii - robot_radius
        return float(np.min(dists))

    def distance_and_grad(self, p: np.ndarray, robot_radius: float = 0.0):
        diffs = p[None, :] - self.centers
        norms = np.linalg.norm(diffs, axis=1)
        dists = norms - self.radii - robot_radius
        j = int(np.argmin(dists))
        grad = diffs[j] / max(norms[j], 1e-9)  # d(dist)/dp
        return float(dists[j]), grad


def hinge_cost(d: float, eps: float):
    """
    c(z) = eps - d(z)  if d(z) <= eps
           0            otherwise

    Returns (cost, dcost_dd) where dcost_dd = -1 in the active region,
    0 otherwise (needed for the chain rule in the Jacobian).
    """
    if d <= eps:
        return eps - d, -1.0
    return 0.0, 0.0


def make_obstacle_factor(key_i, dof: int, fk_fn, sphere_offsets, sdf: SignedDistanceField,
                          eps: float, sigma_obs: float):
    """
    f_obs(theta_i) = exp{ -1/2 || h(theta_i) ||^2_{Sigma_obs} }

    Inputs
    ------
    fk_fn          : function q (dof,) -> list of body-sphere centers in
                      world frame AND their Jacobians d(center)/dq,
                      i.e. fk_fn(q) -> (centers: (M,3), Jlin: (M,3,dof))
                      (forward kinematics, robot/franka.py)
    sphere_offsets : per-sphere ROBOT RADII r_j (robot/franka.py's
                      `sphere_radii`), i.e. s_j's own swept-sphere size --
                      subtracted from the SDF query so clearance reflects
                      actual robot geometry, not just a point approximation.
    sdf            : SignedDistanceField
    eps            : safety margin epsilon
    sigma_obs      : scalar std used to build isotropic Sigma_obs = sigma_obs^2 I_M

    Time complexity per evaluation: O(M * dof) for the FK Jacobian chain
    rule (M = number of collision spheres).
    """
    M = len(sphere_offsets)

    def error_func(this: CustomFactor, values: gtsam.Values, jacobians):
        theta_i = values.atVector(this.keys()[0])
        q = theta_i[:dof]
        centers, Jlin = fk_fn(q)  # centers: (M,3), Jlin: (M,3,dof)
        h = np.zeros(M)
        H = np.zeros((M, 2 * dof))
        for j in range(M):
            d, grad_p = sdf.distance_and_grad(centers[j], robot_radius=sphere_offsets[j])
            c, dc_dd = hinge_cost(d, eps)
            h[j] = c
            # dc/dq = dc/dd * dd/dp * dp/dq = dc_dd * grad_p @ Jlin[j]
            dc_dq = dc_dd * (grad_p @ Jlin[j])
            H[j, :dof] = dc_dq  # zero wrt velocity block
        if jacobians is not None:
            jacobians[0] = H
        return h

    noise = noiseModel.Isotropic.Sigma(M, sigma_obs)
    return CustomFactor(noise, [key_i], error_func)


# --------------------------------------------------------------------------
# Assemble full factor graph  f(Theta) = f_0 * prod f_gp * prod f_obs
# --------------------------------------------------------------------------

def build_gpmp2_graph(N: int, dt: float, dof: int, Qc: np.ndarray,
                       theta0: np.ndarray, theta_goal: np.ndarray,
                       K0: np.ndarray, KN: np.ndarray,
                       fk_fn, sphere_offsets, sdf: SignedDistanceField,
                       eps: float, sigma_obs: float, init_trajectory: np.ndarray = None):
    """
    Builds the complete GPMP2 nonlinear factor graph (Step 1.3) with
    N+1 states theta_0 ... theta_N, keyed by symbol X(0..N).

    init_trajectory : optional (N+1, 2*dof) warm-start for the LM
                       initial values -- see GPMP2Planner.plan()'s
                       docstring for why this matters (prevents cycle-
                       to-cycle route-flipping around obstacles). Falls
                       back to straight-line interpolation if None.

    Returns (graph, keys, init_values) ready for Gauss-Newton/LM
    optimization (planner/gpmp2_planner.py) or for iSAM2 (isam_update.py).
    """
    graph = gtsam.NonlinearFactorGraph()
    keys = [X(i) for i in range(N + 1)]

    graph.add(make_start_factor(keys[0], theta0, K0))
    graph.add(make_goal_factor(keys[-1], theta_goal, KN))

    for i in range(1, N + 1):
        graph.add(make_gp_prior_factor(keys[i - 1], keys[i], dt, dof, Qc))

    for i in range(N + 1):
        graph.add(make_obstacle_factor(keys[i], dof, fk_fn, sphere_offsets,
                                        sdf, eps, sigma_obs))

    init = gtsam.Values()
    if init_trajectory is not None and init_trajectory.shape[0] == N + 1:
        for i in range(N + 1):
            init.insert(keys[i], init_trajectory[i])
    else:
        # Straight-line initialization between theta0 and theta_goal
        for i in range(N + 1):
            alpha = i / N
            init.insert(keys[i], (1 - alpha) * theta0 + alpha * theta_goal)

    return graph, keys, init
