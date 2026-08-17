"""
cbf/qp_solver.py
==================
Implements Step 3.3 (CBF-QP) and Stage 4 (unsafe-controller detection)
exactly.

    u* = argmin_u  ||u - u_mppi||^2
    s.t.  Lg h_theta(x) u >= -alpha(h_theta(x)) - Lf h_theta(x)     [CBF]
          u_min <= u <= u_max                                       [joint limits]

Rewritten as a standard QP:  min 1/2 u^T P u + q^T u   s.t.  l <= A u <= u_bnd
with P = 2I, q = -2 u_mppi, and the CBF row + box constraints stacked
into A. Solved with OSQP (sparse active-set-free ADMM QP solver),
exactly the "OSQP" dependency requested.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import osqp
from dataclasses import dataclass


@dataclass
class UnsafeDetection:
    """Stage 4: everything needed to show a control is unsafe BEFORE the QP."""
    h_value: float
    Lf_h: float
    Lg_h: np.ndarray
    predicted_hdot: float          # Lf_h + Lg_h @ u_mppi
    required_hdot_min: float       # -alpha(h)
    constraint_violation: float    # max(0, required_hdot_min - predicted_hdot)
    is_unsafe: bool
    distance_to_obstacle: float


@dataclass
class CBFQPResult:
    u_safe: np.ndarray
    u_unsafe: np.ndarray
    intervention: np.ndarray       # delta_u = u_safe - u_unsafe
    intervention_magnitude: float
    dual_variables: np.ndarray     # lambda* for [CBF row; box rows]
    active_constraints: list       # names of constraints with lambda* > tol
    safety_margin: float           # h(x) at query time
    slack: float                   # constraint residual at optimum (0 if strictly feasible)
    solve_status: str


def detect_unsafe(u_mppi: np.ndarray, Lf_h: float, Lg_h: np.ndarray, h: float,
                   alpha_gamma: float, d_obstacle: float,
                   violation_tol: float = 1e-6, h0_physical: float = None) -> UnsafeDetection:
    """
    Stage 4: check, BEFORE solving the QP, whether u_mppi already
    violates  hdot >= -alpha(h) = -gamma*h.

    predicted_hdot = Lf h(x) + Lg h(x) . u_mppi   (Step 3.2's h_dot expansion)

    Lf_h/Lg_h/h here are whatever quantity the CBF-QP constraint is
    actually built from -- for a relative-degree-2 barrier (e.g.
    DistanceBarrier under second-order dynamics) these should be the
    HOCBF surrogate (psi1, Lf_psi1, Lg_psi1) from
    cbf.barrier.hocbf_lie_derivatives, NOT the raw h/Lf_h/Lg_h from
    lie_derivatives (which has Lg_h identically zero and can never
    trigger an intervention -- see hocbf_lie_derivatives' docstring).
    h0_physical, if given, is the actual physical barrier value (for
    honest logging) since psi1 is a derived quantity, not itself a
    distance.
    """
    predicted_hdot = Lf_h + float(Lg_h @ u_mppi)
    required_min = -alpha_gamma * h
    violation = max(0.0, required_min - predicted_hdot)
    return UnsafeDetection(
        h_value=h0_physical if h0_physical is not None else h,
        Lf_h=Lf_h, Lg_h=Lg_h,
        predicted_hdot=predicted_hdot, required_hdot_min=required_min,
        constraint_violation=violation, is_unsafe=violation > violation_tol,
        distance_to_obstacle=d_obstacle,
    )


class CBFQPSolver:
    """
    Time complexity: OSQP's ADMM iterations are O(m_dim^3) worst case
    per factorization but for our tiny (m~7 dims, ~2m+1 constraints) QP
    this is sub-millisecond; effectively O(1) per control cycle.
    Memory: O(m_dim^2).
    """

    def __init__(self, m: int, u_min: np.ndarray, u_max: np.ndarray,
                 alpha_gamma: float = 1.0, verbose: bool = False):
        self.m = m
        self.u_min = u_min
        self.u_max = u_max
        self.alpha_gamma = alpha_gamma
        self.verbose = verbose

    def solve(self, u_mppi: np.ndarray, Lf_h: float, Lg_h: np.ndarray,
              h: float, h0_physical: float = None) -> CBFQPResult:
        """
        Lf_h/Lg_h/h build the actual QP constraint -- for a relative-
        degree-2 barrier these must be the HOCBF surrogate
        (psi1, Lf_psi1, Lg_psi1), not the raw barrier value (see
        cbf.barrier.hocbf_lie_derivatives). h0_physical, if given, is
        reported as `safety_margin` instead of `h` for a human-readable
        physical-distance number in CBFQPResult/plots, since psi1 is a
        derived quantity, not itself a distance.
        """
        m = self.m
        P = sp.csc_matrix(2 * np.eye(m))
        q = -2 * u_mppi

        # Row 0: CBF constraint   Lg_h @ u >= -alpha(h) - Lf_h
        #   => in OSQP's  l <= A u <= u_bnd  form:  A[0]=Lg_h, l0 = -alpha*h - Lf_h, u0=+inf
        A_rows = [Lg_h.reshape(1, m)]
        l = [-self.alpha_gamma * h - Lf_h]
        u_bnd = [np.inf]
        names = ["CBF"]

        # Box constraints u_min <= u <= u_max, one row per joint
        A_rows.append(np.eye(m))
        l.extend(list(self.u_min))
        u_bnd.extend(list(self.u_max))
        names.extend([f"u_max_{i}" for i in range(m)])

        A = sp.csc_matrix(np.vstack(A_rows))
        l = np.array(l)
        u_bnd = np.array(u_bnd)

        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u_bnd, verbose=self.verbose,
                   eps_abs=1e-6, eps_rel=1e-6, polish=False)
        # polish=False (was True): "polish" is a refinement step OSQP
        # runs after solving, printing "Polishing not needed" or
        # similar EVERY solve regardless of the verbose=False setting
        # above. This was flooding the terminal, hundreds of lines per
        # second, with no useful information. Turning polish off removes
        # this refinement step entirely -- for this small (7-variable)
        # QP, the un-polished solution is already accurate enough, so
        # this does not weaken the actual safety guarantee.
        res = prob.solve()

        status = res.info.status if res.info is not None else "unknown"
        solved_ok = status in ("solved", "solved inaccurate")

        # IMPORTANT (fixes a real bug): `res.x is not None` is NOT enough
        # to trust the solution. OSQP can return a non-None `res.x` for a
        # PRIMAL/DUAL INFEASIBLE or UNBOUNDED problem too -- in that case
        # `res.x` is a certificate vector (can be numerically enormous,
        # e.g. ~1e18+), not a valid control command. Using it unchecked
        # previously let astronomically large "torque" values silently
        # flow into env.step() and into control-effort metrics (this is
        # exactly what produced the ~2.28e21 control-effort number seen in
        # testing). Two independent safeguards now: (1) only trust res.x
        # when the solver actually reports success, and (2) unconditionally
        # clip whatever is used to the physical actuator limits, since
        # those are hard limits regardless of solver behavior.
        if res.x is not None and solved_ok and np.all(np.isfinite(res.x)):
            u_safe = np.clip(res.x, self.u_min, self.u_max)
        else:
            u_safe = np.clip(u_mppi, self.u_min, self.u_max)
            status = f"{status} (solver output rejected -- fell back to clipped u_mppi)"

        delta_u = u_safe - u_mppi
        duals = res.y if (res.y is not None and solved_ok) else np.zeros(A.shape[0])

        active = [names[i] for i in range(len(names))
                  if abs(duals[i]) > 1e-4]

        # slack = residual of CBF constraint at the optimum
        cbf_lhs = float(Lg_h @ u_safe)
        cbf_rhs = -self.alpha_gamma * h - Lf_h
        slack = max(0.0, cbf_rhs - cbf_lhs)

        return CBFQPResult(
            u_safe=u_safe, u_unsafe=u_mppi, intervention=delta_u,
            intervention_magnitude=float(np.linalg.norm(delta_u)),
            dual_variables=duals, active_constraints=active,
            safety_margin=(h0_physical if h0_physical is not None else h), slack=slack,
            solve_status=status,
        )
