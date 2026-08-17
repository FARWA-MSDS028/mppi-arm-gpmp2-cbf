"""
planner/isam_update.py
========================
Implements Step 5B.4 (iSAM2 incremental update) exactly as specified:

    Input: f_conflict (new conflict factor), f_remove = empty
    1. f_add = {f_conflict}
    2. iSAM2.updateBayesTree(f_add, f_remove)
    3. theta*_new = iSAM2.getCurrentEstimation()

GTSAM's `gtsam.ISAM2` class *is* the Bayes-Tree-based incremental
smoother described in GPMP2 Section III-C; `.update(newFactors,
newValues)` performs exactly the "only touch affected nodes" re-
linearization the spec calls for -- it does NOT rebuild the whole
factor graph, it re-eliminates only the cliques on the path from the
changed variable to the root of the Bayes Tree.

IMPORTANT -- fixed a real bug from the previous version: within ONE
planning cycle, conflict-factor insertions are genuinely incremental
(that's the whole point of iSAM2). But at the START of each NEW
planning cycle, Stage 1 re-solves GPMP2 from scratch with a new start
state and a shifted horizon -- that fresh batch solution must REPLACE
the iSAM2 tree, not be layered on top of it. The previous version kept
one `gtsam.ISAM2` object for the entire run and called `.update()` on
it every time `initialize_from_batch` ran, which added each new
cycle's factors on top of the previous cycle's stale ones sharing the
same variable keys X(0..N) -- duplicating/conflicting factors on those
keys instead of starting a clean problem. `initialize_from_batch` now
constructs a genuinely NEW `gtsam.ISAM2` instance each time it's
called, so it starts truly "from scratch" per cycle; only
`add_factors_incremental` (conflict factors, within a cycle) reuses
the existing tree incrementally.

Also fixed a GTSAM Python-binding API mismatch: `ISAM2Params.factorization`
is a plain attribute, `relinearizeSkip` is a plain attribute, but
`relinearizeThreshold` must be set via the `setRelinearizeThreshold()`
method (not attribute assignment) in the standard GTSAM Python bindings
-- calling it the wrong way for either raises AttributeError depending
on your installed gtsam build. The setup below tries the method form
first and falls back to attribute assignment (and vice versa) so it
works across binding versions without you having to guess which one
your `pip install gtsam` gave you.
"""

from __future__ import annotations
import numpy as np
import gtsam
from dataclasses import dataclass


def _build_isam2_params(relinearize_threshold: float, relinearize_skip: int) -> "gtsam.ISAM2Params":
    """
    Builds ISAM2Params robustly across GTSAM Python-binding versions --
    some expose relinearizeThreshold/relinearizeSkip/factorization as
    plain attributes, others as setter methods (setRelinearizeThreshold,
    setRelinearizeSkip, setFactorization). Try attribute first, then the
    setter method, for each field independently; never let one field's
    API mismatch crash setup of the others.
    """
    params = gtsam.ISAM2Params()

    def _set(field_name: str, setter_name: str, value):
        try:
            setattr(params, field_name, value)
            return
        except (AttributeError, TypeError):
            pass
        setter = getattr(params, setter_name, None)
        if setter is not None:
            try:
                setter(value)
                return
            except TypeError:
                pass
        # Neither worked -- leave GTSAM's default for this field rather
        # than crashing; log so it's visible instead of silently wrong.
        print(f"[isam_update] warning: could not set ISAM2Params.{field_name} "
              f"via attribute or {setter_name}() on this GTSAM build; using default.")

    _set("relinearizeThreshold", "setRelinearizeThreshold", relinearize_threshold)
    _set("relinearizeSkip", "setRelinearizeSkip", relinearize_skip)
    _set("factorization", "setFactorization", "CHOLESKY")

    return params


@dataclass
class ISAM2UpdateResult:
    theta_new: np.ndarray      # (N+1, 2*dof) updated MAP trajectory
    n_relinearized: int        # number of variables ISAM2 actually touched
    keys_updated: list


class ISAM2Manager:
    """
    Time complexity of `add_factors_incremental`: O(c^3) where c is the
    size of the largest affected clique (typically small and local for
    a single conflict factor -- this is the entire point of using
    iSAM2 instead of batch LM, per Step 5B.4).
    Memory complexity: O(N * dof^2) for the retained Bayes Tree.

    Lifecycle: call `initialize_from_batch` once per planning cycle
    (it rebuilds the ISAM2 tree fresh from that cycle's GPMP2 batch
    solve), then call `add_factors_incremental` zero or more times
    WITHIN that same cycle for conflict-factor insertions -- those
    calls are genuinely incremental and do not rebuild anything.
    """

    def __init__(self, relinearize_threshold: float = 0.05,
                 relinearize_skip: int = 1):
        self._relinearize_threshold = relinearize_threshold
        self._relinearize_skip = relinearize_skip
        self.isam = None
        self._initialized = False
        self.keys = None
        self.dof2 = None

    def initialize_from_batch(self, graph: gtsam.NonlinearFactorGraph,
                               values: gtsam.Values, keys: list, dof2: int):
        """
        (Re)seeds iSAM2 with a fresh batch GPMP2 solution. Builds a NEW
        gtsam.ISAM2 instance every call -- this is the "from scratch"
        reset each planning cycle needs; do not call this mid-cycle for
        conflict factors, use add_factors_incremental for those.
        """
        params = _build_isam2_params(self._relinearize_threshold, self._relinearize_skip)
        self.isam = gtsam.ISAM2(params)
        self.isam.update(graph, values)
        self.keys = keys
        self.dof2 = dof2
        self._initialized = True

    def add_factors_incremental(self, new_factors: list,
                                 new_values: gtsam.Values = None) -> ISAM2UpdateResult:
        """
        Inserts new factors (e.g. a single conflict factor, Step 5B.3)
        WITHOUT rebuilding the graph. f_remove is empty in our use case
        (conflict factors are additive penalties, never retracted),
        matching the spec's f_remove = ∅. This is the genuinely
        incremental path -- it reuses self.isam as-is.
        """
        assert self._initialized, "Call initialize_from_batch() first."
        fg = gtsam.NonlinearFactorGraph()
        for f in new_factors:
            fg.add(f)
        vals = new_values if new_values is not None else gtsam.Values()

        result = self.isam.update(fg, vals)
        current = self.isam.calculateEstimate()

        theta_new = np.stack([current.atVector(k) for k in self.keys], axis=0)
        return ISAM2UpdateResult(
            theta_new=theta_new,
            n_relinearized=result.getVariablesRelinearized() if hasattr(
                result, "getVariablesRelinearized") else -1,
            keys_updated=list(self.keys),
        )

    def current_trajectory(self) -> np.ndarray:
        current = self.isam.calculateEstimate()
        return np.stack([current.atVector(k) for k in self.keys], axis=0)
