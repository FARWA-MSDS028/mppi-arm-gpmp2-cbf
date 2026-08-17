"""
cbf/barrier.py
================
Implements Step 3.1 (learned barrier h_theta) and Step 3.2 (control-
affine dynamics + Lie derivatives), plus the analytic distance-based
barrier h(x) = d(x) - d_safe and its High-Order CBF (HOCBF) terms.

h_theta(x) is a small MLP: R^n -> R (n=14 for Franka: 7 pos + 7 vel).
Trained offline with the 3-term hinge loss (Eq. in Step 3.1):

    L(theta) = ||theta||^2
             + lambda_s * sum_{safe}   [gamma_safe   - h_theta(x)]_+
             + lambda_u * sum_{unsafe} [h_theta(x)    + gamma_unsafe]_+
             + lambda_d * sum_{dyn}    [gamma_dyn - <grad h_theta(x), f(x,u)> - alpha(h_theta(x))]_+

implemented with plain NumPy (forward + manual backward pass) so the
whole pipeline has no autodiff-framework dependency beyond NumPy/SciPy,
matching the requested software stack.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


def closest_clearance(fk_fn, sphere_radii, sdf, q) -> float:
    """
    Minimum clearance (signed distance minus both the obstacle's and the
    robot sphere's own radius) over ALL of the robot's collision spheres
    at configuration q -- the SAME quantity DistanceBarrier.forward
    computes internally (before subtracting d_safe: h(x) = this - d_safe).

    Exposed standalone (not just inside DistanceBarrier) so every
    experiment script can log an honest "closest approach" distance --
    fixes a real bug where every experiment previously logged
    sdf.distance(env.ee_position(), ...), i.e. END-EFFECTOR-ONLY
    clearance. That silently under-reported risk whenever a different
    link (e.g. the forearm/wrist) was actually the closest point to the
    obstacle.
    """
    centers, _ = fk_fn(q)
    best = np.inf
    for j in range(centers.shape[0]):
        d, _ = sdf.distance_and_grad(centers[j], robot_radius=sphere_radii[j])
        best = min(best, d)
    return float(best)


# --------------------------------------------------------------------------
# Analytic distance-based barrier:  h(x) = d(x) - d_safe
# --------------------------------------------------------------------------

@dataclass
class DistanceBarrier:
    """
    h(x) = d(x) - d_safe

    where d(x) is the minimum clearance (signed distance minus both the
    obstacle's radius and the robot sphere's own radius, per
    planner/factor_graph.py's SignedDistanceField.distance) over ALL of
    the robot's collision spheres at configuration q = x[:dof], and
    d_safe is a fixed safety margin (h(x) > 0 means "more than d_safe
    clearance"; h(x) = 0 is exactly the safety boundary; h(x) < 0 means
    the margin is violated).

    IMPORTANT: h(x) depends only on q, not qdot -- under torque control
    this makes h RELATIVE DEGREE 2 (see hocbf_lie_derivatives below).
    Do NOT feed lie_derivatives(barrier, x, f, g)'s (Lf_h, Lg_h, h)
    directly into cbf.qp_solver.CBFQPSolver -- Lg_h is identically zero
    and the QP could never correct anything. Use hocbf_lie_derivatives
    instead for the actual CBF-QP constraint.

    Inputs
    ------
    fk_fn        : q (dof,) -> (centers: (M,3), Jlin: (M,3,dof))
                   (robot/franka.py FrankaModel.fk)
    sphere_radii : (M,) robot sphere radii (FrankaModel.sphere_radii)
    sdf          : planner.factor_graph.SignedDistanceField
    dof          : number of joints (7 for Panda)
    d_safe       : desired safety margin (m). h(x)=0 at clearance=d_safe.

    Time complexity: O(M*dof) per forward+grad call (M = # collision
    spheres), dominated by the FK Jacobian evaluation.
    """
    fk_fn: object
    sphere_radii: np.ndarray
    sdf: object
    dof: int
    d_safe: float = 0.05

    def _closest(self, q: np.ndarray):
        """Returns (d_min, grad_q) = clearance of the closest robot sphere
        to any obstacle, and its gradient wrt q (dof,)."""
        centers, Jlin = self.fk_fn(q)  # (M,3), (M,3,dof)
        M = centers.shape[0]
        best_d = np.inf
        best_grad_q = np.zeros(self.dof)
        for j in range(M):
            d, grad_p = self.sdf.distance_and_grad(
                centers[j], robot_radius=self.sphere_radii[j])
            if d < best_d:
                best_d = d
                # d(clearance)/dq = grad_p . d(center)/dq = grad_p @ Jlin[j]
                best_grad_q = grad_p @ Jlin[j]
        return best_d, best_grad_q

    def forward(self, x: np.ndarray) -> float:
        q = x[:self.dof]
        d_min, _ = self._closest(q)
        return float(d_min - self.d_safe)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """grad_x h(x) in R^{2*dof}. h depends only on q (not qdot), so
        the velocity block of the gradient is exactly zero."""
        q = x[:self.dof]
        _, grad_q = self._closest(q)
        grad_full = np.zeros(2 * self.dof)
        grad_full[:self.dof] = grad_q
        return grad_full


# --------------------------------------------------------------------------
# Step 3.2: control-affine Franka dynamics f(x), g(x)
# --------------------------------------------------------------------------
def franka_dynamics(x, dof, gravity_fn, coriolis_fn, M_fn):
    """
    Control-affine dynamics xdot = f(x) + g(x)*u, where u is a DESIRED
    POSITION (not torque) -- confirmed by direct testing that the Panda
    actuators in this MJCF are position servos with fixed internal gains
    (kp=4500/3500/2000, kv=450/350/200 per joint group), NOT torque
    motors. Every earlier version of this function assumed direct torque
    input (g(x) = [0; M^-1]) -- that assumption was wrong, confirmed by
    sending a torque-scale ctrl value and observing the joint clamp to
    its own position limit instead of behaving as a torque.
    """
    q, qdot = x[:dof], x[dof:]
    Kp_actuator = np.diag([4500., 4500., 3500., 3500., 2000., 2000., 2000.])
    Kv_actuator = np.diag([450., 450., 350., 350., 200., 200., 200.])

    M = M_fn(q)
    Minv = np.linalg.inv(M)
    C = coriolis_fn(q, qdot)
    G = gravity_fn(q)

    f_qddot = Minv @ (-Kp_actuator @ q - Kv_actuator @ qdot - C - G)
    f_x = np.concatenate([qdot, f_qddot])

    g_qddot = Minv @ Kp_actuator
    g_x = np.vstack([np.zeros((dof, dof)), g_qddot])

    return f_x, g_x

# --------------------------------------------------------------------------
# Step 3.1: learned barrier MLP  h_theta(x)
# --------------------------------------------------------------------------

def _relu(z):
    return np.maximum(0, z)


def _relu_grad(z):
    return (z > 0).astype(float)


@dataclass
class MLPBarrier:
    """
    Two-layer MLP:  h_theta(x) = w2^T relu(W1 x + b1) + b2

    Params stored flat for easy SGD; forward() and grad() both O(n*H)
    (H = hidden width), giving O(n*H) Lie-derivative evaluation, cheap
    enough for the CBF-QP's per-control-cycle linear-constraint build.
    """
    n: int
    H: int = 64
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(scale=1.0 / np.sqrt(self.n), size=(self.H, self.n))
        self.b1 = np.zeros(self.H)
        self.w2 = rng.normal(scale=1.0 / np.sqrt(self.H), size=self.H)
        self.b2 = 0.0

    def forward(self, x: np.ndarray) -> float:
        z1 = self.W1 @ x + self.b1
        a1 = _relu(z1)
        return float(self.w2 @ a1 + self.b2)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """grad_x h_theta(x) in R^n, via manual backprop."""
        z1 = self.W1 @ x + self.b1
        d1 = _relu_grad(z1)
        # dh/dx = w2^T * diag(d1) * W1
        return (self.w2 * d1) @ self.W1

    def params(self):
        return [self.W1, self.b1, self.w2, self.b2]

    # ---- Offline training: 3-term hinge loss (Step 3.1) -----------------
    def train_step(self, X_safe, X_unsafe, XU_dyn, f_fn, alpha_fn, alpha_grad_fn,
                    lam_s=1.0, lam_u=1.0, lam_d=1.0,
                    gamma_safe=0.1, gamma_unsafe=0.1, gamma_dyn=0.0,
                    weight_decay=1e-4, lr=1e-3):
        """
        One SGD step over a minibatch.

        X_safe   : (Ns, n) states known safe
        X_unsafe : (Nu, n) states known unsafe
        XU_dyn   : list of (x, u) pairs with known transitions
        f_fn     : x -> f(x) (drift dynamics, for the CBF-derivative term)
        alpha_fn, alpha_grad_fn : class-K function alpha(h) and its derivative

        Returns scalar loss (for logging / convergence plots).
        """
        gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
        gw2 = np.zeros_like(self.w2); gb2 = 0.0
        loss = weight_decay * (np.sum(self.W1**2) + np.sum(self.w2**2))

        def backprop_scalar(x, dL_dh):
            """Accumulate gradients of loss w.r.t. params given dL/dh at x."""
            nonlocal gW1, gb1, gw2, gb2
            z1 = self.W1 @ x + self.b1
            a1 = _relu(z1)
            d1 = _relu_grad(z1)
            gw2 += dL_dh * a1
            gb2 += dL_dh
            dz1 = dL_dh * self.w2 * d1
            gW1 += np.outer(dz1, x)
            gb1 += dz1

        # Condition 1: safe loss  lam_s * [gamma_safe - h(x)]_+
        for x in X_safe:
            h = self.forward(x)
            hinge = gamma_safe - h
            if hinge > 0:
                loss += lam_s * hinge
                backprop_scalar(x, -lam_s)

        # Condition 2: unsafe loss  lam_u * [h(x) + gamma_unsafe]_+
        for x in X_unsafe:
            h = self.forward(x)
            hinge = h + gamma_unsafe
            if hinge > 0:
                loss += lam_u * hinge
                backprop_scalar(x, lam_u)

        # Condition 3: CBF-derivative loss
        # lam_d * [gamma_dyn - <grad h(x), f(x,u)> - alpha(h(x))]_+
        for (x, u) in XU_dyn:
            h = self.forward(x)
            gh = self.grad(x)
            fx = f_fn(x, u)
            lie = gh @ fx
            a = alpha_fn(h)
            hinge = gamma_dyn - lie - a
            if hinge > 0:
                loss += lam_d * hinge
                # d(hinge)/dh = -alpha'(h);  d(hinge)/d(grad h . f) = -1
                # Approximate grad-of-lie-term wrt params via h's dependency only
                # (grad_x h treated as locally frozen for this simple 2-layer
                #  update -- adequate for a shallow barrier network).
                backprop_scalar(x, -lam_d * alpha_grad_fn(h))

        # weight decay grads
        gW1 += 2 * weight_decay * self.W1
        gw2 += 2 * weight_decay * self.w2

        self.W1 -= lr * gW1; self.b1 -= lr * gb1
        self.w2 -= lr * gw2; self.b2 -= lr * gb2
        return loss


def alpha_linear(h: float, gamma: float = 1.0) -> float:
    """Class-K function alpha(h) = gamma * h (Step 3.3 choice)."""
    return gamma * h


def alpha_linear_grad(h: float, gamma: float = 1.0) -> float:
    return gamma


def hocbf_lie_derivatives(barrier, x: np.ndarray, f_fn, g_fn,
                           alpha0: float = 1.0, dt_fd: float = 1e-3):
    """
    High-Order Control Barrier Function terms (Xiao & Belta, 2019),
    needed because h0(x) = d(q) - d_safe depends only on POSITION, while
    torque u only affects ACCELERATION -- h0 has relative degree 2 wrt u,
    so the plain first-order constraint Lg_h0 @ u >= ... is IDENTICALLY
    VACUOUS (Lg_h0 = grad_h0 @ g(x) = 0 always, since grad_h0's velocity
    block is zero and g(x)'s position block is zero -- see franka_dynamics).
    Using the plain (Lf_h0, Lg_h0, h0) triple in the CBF-QP silently
    produces a constraint that can never change u, regardless of how
    unsafe the state is -- this is exactly the "intervention always
    0.0000, even while h0(x) < 0" bug.

    Fix: define the first-order surrogate
        psi1(x) = Lf_h0(x) + alpha0 * h0(x)
    and constrain psi1's OWN derivative instead:
        psi1_dot(x) = Lf_psi1(x) + Lg_psi1(x) . u  >=  -alpha1(psi1(x))
    which has the same (Lf, Lg, h)-shape cbf.qp_solver.CBFQPSolver.solve
    already expects -- just feed it (Lf_psi1, Lg_psi1, psi1) in place of
    (Lf_h0, Lg_h0, h0).

    Derivation (h0 depends only on q, so grad_h0 = [grad_q_h0; 0]):
        psi1 = grad_q_h0 . qdot + alpha0 * h0
        d(psi1)/dt = qdot^T Hess_q(h0) qdot + grad_q_h0 . qddot + alpha0 * Lf_h0
        qddot = f_qddot(x) + M(q)^-1 u        (manipulator dynamics)
      =>  Lg_psi1 = grad_q_h0 @ M(q)^-1                        [EXACT, no Hessian]
          Lf_psi1 = [qdot^T Hess_q(h0) qdot + grad_q_h0 . f_qddot(x)] + alpha0*Lf_h0

    M(q)^-1 is read directly off g(x)'s velocity block (g_fn(x)[dof:,:],
    per franka_dynamics' g(x) = [0; M(q)^-1]) rather than recomputing it,
    so no separate M_fn is needed at the call site.

    The bracketed Hessian term is exactly d/dt[Lf_h0(x(t))] evaluated
    along DRIFT-ONLY dynamics (u=0) -- rather than an explicit second-
    derivative of the forward-kinematics chain, it is estimated with a
    single forward-Euler finite difference:
        x_next = x + dt_fd * f(x)
        d/dt[Lf_h0] ~= (Lf_h0(x_next) - Lf_h0(x)) / dt_fd

    Inputs
    ------
    barrier : DistanceBarrier (uses barrier.dof; any object with the
              same .forward/.grad/.dof interface works)
    x       : (2*dof,) current state [q; qdot]
    f_fn    : callable, x -> f(x) drift dynamics (2*dof,)
    g_fn    : callable, x -> g(x) control matrix (2*dof, dof)
    alpha0  : inner class-K gain (linear: alpha0(h0) = alpha0 * h0)
    dt_fd   : finite-difference step for the drift-derivative estimate

    Returns
    -------
    (psi1, Lf_psi1, Lg_psi1, h0) -- feed (Lf_psi1, Lg_psi1, psi1) to
    qp.solve()/detect_unsafe() as the constraint's (Lf_h, Lg_h, h); h0
    is the raw physical barrier value, for honest reporting/logging
    (pass as h0_physical) since psi1 is a derived quantity, not itself
    a distance.

    Time complexity: O(M*dof) for two barrier evaluations (M = collision
    spheres), same order as the relative-degree-1 path it replaces.
    """
    dof = barrier.dof
    q = x[:dof]
    h0 = barrier.forward(x)
    grad_h0 = barrier.grad(x)
    grad_q_h0 = grad_h0[:dof]

    f_x = f_fn(x)
    Lf_h0 = float(grad_h0 @ f_x)

    g_x = g_fn(x)               # (2*dof, dof); bottom block = M(q)^-1
    Minv = g_x[dof:, :]
    Lg_psi1 = grad_q_h0 @ Minv  # (dof,) -- exact, closed form

    x_next = x + dt_fd * f_x
    f_next = f_fn(x_next)
    grad_h0_next = barrier.grad(x_next)
    Lf_h0_next = float(grad_h0_next @ f_next)
    d_Lf_h0_dt = (Lf_h0_next - Lf_h0) / dt_fd

    psi1 = Lf_h0 + alpha0 * h0
    Lf_psi1 = d_Lf_h0_dt + alpha0 * Lf_h0

    return psi1, Lf_psi1, Lg_psi1, h0


def lie_derivatives(barrier, x: np.ndarray, f: np.ndarray, g: np.ndarray):
    """
    L_f h(x) = grad_x h(x) . f(x)
    L_g h(x) = grad_x h(x) . g(x)     (row vector in R^{1 x m})

    Returns (Lf: float, Lg: (m,) ndarray, h: float, grad_h: (n,) ndarray)

    NOTE: for DistanceBarrier under second-order (torque-controlled)
    dynamics, Lg here is identically zero (relative degree 2) -- use
    hocbf_lie_derivatives for the actual CBF-QP constraint instead.
    This function is kept for barriers that genuinely have relative
    degree 1 (e.g. a learned MLPBarrier trained directly against
    <grad h, f(x,u)>, per Step 3.1's training loss).
    """
    h = barrier.forward(x)
    grad_h = barrier.grad(x)
    Lf = float(grad_h @ f)
    Lg = grad_h @ g  # (m,)
    return Lf, Lg, h, grad_h
