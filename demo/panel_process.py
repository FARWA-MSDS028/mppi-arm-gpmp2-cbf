"""
demo/panel_process.py
========================
Runs the matplotlib info panel in a completely SEPARATE OS PROCESS from
the MuJoCo viewer -- not just a separate thread. This fully decouples
the two GUI toolkits (GLFW for MuJoCo, Tk/Qt for matplotlib) so a stall
in either one's event loop can never freeze the other. Communication is
one-way, non-blocking: the main process pushes state snapshots (plain
picklable dicts) into a multiprocessing.Queue; this process reads and
redraws whenever one is available, dropping stale ones if it falls
behind.
"""
from __future__ import annotations
import queue as queue_module

PIPELINE_STAGES = [
    "GPMP2", "MPPI", "CBF-QP", "Robot Execution",
    "Feasibility Extraction", "Conflict Factor",
    "Covariance Steering", "Update GPMP2",
]


class InfoPanel:
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig, self.axes = plt.subplots(4, 1, figsize=(4.5, 8),
                                            gridspec_kw={"height_ratios": [3, 2, 1, 1]})
        self.fig.canvas.manager.set_window_title("Pipeline Dashboard")
        self.ax_pipeline, self.ax_metrics, self.ax_feas, self.ax_conflict = self.axes
        plt.ion()
        self.fig.show()

    def render(self, snap: dict):
        self._draw_pipeline(snap)
        self._draw_metrics(snap)
        self._draw_feasibility(snap)
        self._draw_conflict(snap)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def _draw_pipeline(self, snap):
        ax = self.ax_pipeline
        ax.clear()
        ax.set_title(f"Pipeline  (cycle {snap.get('cycle')}, step {snap.get('control_step')})",
                     fontsize=10)
        ax.axis("off")
        current = snap.get("current_stage")
        try:
            current_idx = PIPELINE_STAGES.index(current) if current else -1
        except ValueError:
            current_idx = -1
        for i, name in enumerate(PIPELINE_STAGES):
            y = 1.0 - i / (len(PIPELINE_STAGES) - 1)
            if i == current_idx:
                color, weight = "#2ea043", "bold"
            elif i < current_idx:
                color, weight = "#8b8b8b", "normal"
            else:
                color, weight = "#c9d1d9", "normal"
            ax.text(0.1, y, name, fontsize=11, color=color, fontweight=weight,
                     transform=ax.transAxes, va="center")

    def _draw_metrics(self, snap):
        ax = self.ax_metrics
        ax.clear()
        ax.axis("off")
        ge = snap.get("goal_error")
        st = snap.get("sim_time")
        cf = snap.get("control_freq_hz")
        mh = snap.get("qp_min_h_seen")
        ss = snap.get("sigma_scale")
        lines = [
            f"Target error:      {ge:.4f}" if ge is not None else "Target error:      --",
            f"Sim time:          {st:.2f} s" if st is not None else "Sim time:          --",
            f"Ctrl freq:         {cf:.1f} Hz" if cf else "Ctrl freq:         --",
            f"MPPI samples:      {snap.get('mppi_n_samples')}" if snap.get('mppi_n_samples') else "MPPI samples:      --",
            f"QP unsafe (total): {snap.get('qp_n_unsafe_total', 0)}",
            f"QP corrected:      {snap.get('qp_n_corrected_total', 0)}",
            f"Min h(x) seen:     {mh:.4f}" if mh is not None else "Min h(x) seen:     --",
            f"Conflicts total:   {snap.get('n_conflicts_total', 0)}",
            f"Sigma scale:       {ss:.5f}" if ss is not None else "Sigma scale:       --",
            f"GPMP2 accepted:    {snap.get('gpmp2_accepted')}",
            f"GPMP2 rolled back: {snap.get('gpmp2_rolled_back')}",
        ]
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                 fontsize=9, va="top", family="monospace")

    def _draw_feasibility(self, snap):
        ax = self.ax_feas
        ax.clear()
        val = snap.get("feasibility_bar") or 0.0
        color = "#2ea043" if val < 0.5 else ("#d29922" if val < 0.85 else "#da3633")
        ax.barh([0], [val], color=color)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        sm = snap.get("safety_margin")
        ax.set_title(f"Feasibility bar  (h={sm:.4f})" if sm is not None else "Feasibility bar",
                     fontsize=9)

    def _draw_conflict(self, snap):
        ax = self.ax_conflict
        ax.clear()
        val = snap.get("conflict_score") or 0.0
        cmap = self.plt.get_cmap("RdYlGn_r")
        ax.barh([0], [val], color=cmap(val))
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        active = "ACTIVE" if snap.get("conflict_active") else "inactive"
        ax.set_title(f"Conflict score: {val:.2f}  ({active})", fontsize=9)


def run_panel_process(q):
    """Entry point run in the separate process. Blocks reading from q,
    redraws whenever a fresh snapshot arrives; times out periodically so
    the matplotlib window stays responsive (drag/resize/close) even
    between snapshots."""
    panel = InfoPanel()
    latest = None
    while True:
        try:
            latest = q.get(timeout=0.15)
            while True:
                try:
                    latest = q.get_nowait()
                except queue_module.Empty:
                    break
        except queue_module.Empty:
            pass
        if latest is not None:
            try:
                panel.render(latest)
            except Exception:
                pass
        else:
            panel.plt.pause(0.05)
