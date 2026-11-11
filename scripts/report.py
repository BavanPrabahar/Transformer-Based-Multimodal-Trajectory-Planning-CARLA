"""Aggregate evaluation episodes into tables (markdown/json) and README figures.

Usage: python scripts/report.py --results results/town02 --history checkpoints/planner_history.json
"""
import argparse
import json
import math
import os
from collections import OrderedDict

import numpy as np

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference palette (dataviz skill): categorical slots in fixed order, status palette, chrome
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = OrderedDict(success=("#0ca30c", "#ffffff", "Success"), freeze=("#fab219", "#0b0b0b", "Freeze / timeout"),
                     offroad=("#ec835a", "#0b0b0b", "Off-road"), collision=("#d03b3b", "#ffffff", "Collision"))
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORDER = ["lab_mpc", "capsule_nmpc", "hybrid", "expert", "hybrid_soft", "hybrid_noselector", "hybrid_nosafety",
         "hybrid_top1"]
LABEL = {"lab_mpc": "Lab MPC stack (MPC_capsule)", "capsule_nmpc": "Capsule-constraint NMPC",
         "hybrid": "Transformer + selector + LTV-MPC (as designed)", "expert": "Privileged lattice expert (oracle)",
         "hybrid_soft": "Hybrid, soft selector (revised)",
         "hybrid_noselector": "Transformer top-1 + safety LTV-MPC (recommended)", "hybrid_nosafety": "Hybrid w/o safety half-planes",
         "hybrid_top1": "Transformer top-1 + plain tracking"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.facecolor": SURFACE,
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
                     "axes.spines.right": False})


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def outcome(r):
    if r["success"]:
        return "success"
    if r["collision"]:
        return "collision"
    if r["offroad"]:
        return "offroad"
    return "freeze"


def load(paths):
    rows = []
    for p in paths:
        f = os.path.join(p, "episodes.jsonl") if os.path.isdir(p) else p
        if os.path.exists(f):
            rows += [json.loads(l) for l in open(f)]
    return rows


def common_scenarios(rows, ctrls):
    """Only scenarios that every listed controller has finished (paired comparison)."""
    by = {}
    for r in rows:
        by.setdefault((r["town"], r["template"], r["seed"]), {})[r["controller"]] = r
    keep = {k for k, v in by.items() if all(c in v for c in ctrls)}
    return [r for r in rows if (r["town"], r["template"], r["seed"]) in keep and r["controller"] in ctrls], len(keep)


def summarize(rows, ctrls):
    out = OrderedDict()
    for c in ctrls:
        rs = [r for r in rows if r["controller"] == c]
        n = len(rs)
        if not n:
            continue
        oc = [outcome(r) for r in rs]
        succ = [r for r in rs if r["success"]]
        k = oc.count("success")
        out[c] = dict(
            n=n, success=k / n, success_ci=wilson(k, n), collision=oc.count("collision") / n,
            freeze=oc.count("freeze") / n, offroad=oc.count("offroad") / n,
            progress=float(np.mean([r["progress"] for r in rs])),
            time_to_goal=float(np.median([r["time"] for r in succ])) if succ else float("nan"),
            min_dist=float(np.median([r["min_dist"] for r in rs if r["min_dist"] < 90])) if any(
                r["min_dist"] < 90 for r in rs) else float("nan"),
            rms_jerk=float(np.median([r["rms_jerk"] for r in succ])) if succ else float("nan"),
            rms_lat_acc=float(np.median([r["rms_lat_acc"] for r in succ])) if succ else float("nan"),
            solve_mean_ms=float(np.mean([r["solve_mean_ms"] for r in rs])),
            solve_p99_ms=float(np.percentile([r["solve_p99_ms"] for r in rs], 50)),
            solve_max_ms=float(np.max([r["solve_max_ms"] for r in rs])),
            solver_fail=float(np.mean([r["solver_fail_rate"] for r in rs])),
            collided_with=sorted({r.get("collided_with", "") for r in rs if r["collision"]} - {""}),
        )
    return out


def per_template(rows, ctrls):
    tpls = sorted({r["template"] for r in rows})
    M = np.full((len(tpls), len(ctrls)), np.nan)
    N = np.zeros_like(M)
    for i, t in enumerate(tpls):
        for j, c in enumerate(ctrls):
            rs = [r for r in rows if r["template"] == t and r["controller"] == c]
            if rs:
                M[i, j] = np.mean([r["success"] for r in rs])
                N[i, j] = len(rs)
    return tpls, M, N


def fig_outcomes(summ, path):
    ctrls = list(summ)
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(ctrls) + 1.4))
    y = np.arange(len(ctrls))[::-1]
    left = np.zeros(len(ctrls))
    for key, (col, txt, lab) in STATUS.items():
        vals = np.array([summ[c][key] for c in ctrls]) * 100
        ax.barh(y, vals, left=left, color=col, height=0.56, edgecolor=SURFACE, linewidth=2, label=lab)
        for yi, l, v in zip(y, left, vals):
            if v >= 7:
                ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", color=txt, fontsize=9)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([f"{LABEL.get(c, c)}  (n={summ[c]['n']})" for c in ctrls], color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of episodes (%)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.4, -0.22), frameon=False, fontsize=9)
    ax.set_title("Closed-loop outcomes on held-out CARLA Town02 scenarios", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_heatmap(tpls, M, N, ctrls, path):
    fig, ax = plt.subplots(figsize=(1.55 * len(ctrls) + 2.6, 0.48 * len(tpls) + 1.6))
    cmap = matplotlib.colors.ListedColormap(SEQ)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(tpls)):
        for j in range(len(ctrls)):
            if np.isnan(M[i, j]):
                continue
            v = M[i, j]
            ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=9,
                    color="#ffffff" if v > 0.55 else INK)
    ax.set_xticks(range(len(ctrls)))
    ax.set_xticklabels([LABEL.get(c, c).replace(" (", "\n(").replace(" + ", "\n+ ") for c in ctrls], fontsize=8,
                       color=INK)
    ax.set_yticks(range(len(tpls)))
    ax.set_yticklabels(tpls, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(ctrls)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tpls)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("success rate")
    cb.outline.set_visible(False)
    ax.set_title("Success rate by scenario template", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_latency(rows, ctrls, path):
    fig, ax = plt.subplots(figsize=(8.6, 0.6 * len(ctrls) + 1.4))
    y = np.arange(len(ctrls))[::-1]
    rng = np.random.default_rng(0)
    for yi, c, col in zip(y, ctrls, SERIES):
        p99 = np.array([r["solve_p99_ms"] for r in rows if r["controller"] == c])
        mean = np.array([r["solve_mean_ms"] for r in rows if r["controller"] == c])
        ax.scatter(p99, yi + rng.uniform(-0.16, 0.16, len(p99)), s=14, color=col, alpha=0.55, linewidths=0,
                   label=None)
        ax.plot([np.median(mean)] * 2, [yi - 0.3, yi + 0.3], color=INK, linewidth=2)
        ax.text(1.01, yi, f"median p99 {np.median(p99):.0f} ms", va="center", fontsize=8, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.axvline(100, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(100, y.max() + 0.55, "100 ms = one control period", fontsize=8, color=MUTED, ha="center")
    ax.set_xscale("log")
    ticks = [5, 10, 20, 50, 100, 200, 500]
    lo, hi = ax.get_xlim()
    ax.set_xticks([t for t in ticks if lo <= t <= hi * 1.5])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL.get(c, c) for c in ctrls], color=INK)
    ax.set_xlabel("p99 compute time per control step, one dot per episode (ms, log scale)\nblack tick = median of episode means")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, which="both")
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, y.max() + 0.9)
    ax.set_title("Compute latency (CPU, Python)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(right=0.84)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_comfort(rows, ctrls, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 0.55 * len(ctrls) + 1.6), sharey=True)
    y = np.arange(len(ctrls))[::-1]
    for ax, key, lab in zip(axes, ["rms_jerk", "rms_lat_acc"],
                            ["RMS longitudinal jerk (m/s³)", "RMS lateral acceleration (m/s²)"]):
        for yi, c, col in zip(y, ctrls, SERIES):
            v = np.array([r[key] for r in rows if r["controller"] == c and r["success"]])
            if not len(v):
                ax.text(0.02, yi, "no successful episodes", va="center", fontsize=8, color=MUTED,
                        transform=ax.get_yaxis_transform())
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([q1, q3], [yi, yi], color=col, linewidth=6, solid_capstyle="round", alpha=0.45)
            ax.scatter([med], [yi], s=60, color=col, edgecolors=SURFACE, linewidths=2, zorder=3)
            ax.text(med, yi + 0.3, f"{med:.2f}", ha="center", fontsize=8, color=INK2)
        ax.set_xlabel(lab)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(left=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL.get(c, c) for c in ctrls], color=INK)
    fig.suptitle("Ride comfort on successful episodes (dot = median, bar = interquartile range)", x=0.01,
                 ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_training(hist, path):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for key, lab, col in [("val_minADE", "minADE over 6 modes", SERIES[0]),
                          ("val_ADE1", "ADE of the top-scored mode", SERIES[1])]:
        v = [h[key] for h in hist]
        ax.plot(ep, v, color=col, linewidth=2)
        ax.text(ep[-1] + 0.4, v[-1], f"{lab}  {v[-1]:.2f} m", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation displacement error (m)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlim(1, ep[-1] * 1.45)
    ax.set_xticks([t for t in range(0, ep[-1] + 1, 10) if t >= 1] + ([ep[-1]] if ep[-1] % 10 else []))
    ax.set_ylim(bottom=0)
    ax.set_title("Trajectory transformer training (held-out episodes, 3 s horizon)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def md_table(summ):
    head = ("| Controller | n | Success (95% CI) | Collision | Freeze/timeout | Off-road | Route progress | "
            "Time to goal (s, median) | Min gap (m, median) | RMS jerk | RMS lat. acc. | Compute mean / p99 / max (ms) |")
    lines = [head, "|" + "---|" * 12]
    for c, s in summ.items():
        lo, hi = s["success_ci"]
        lines.append(
            f"| {LABEL.get(c, c)} | {s['n']} | **{100 * s['success']:.0f}%** ({100 * lo:.0f}–{100 * hi:.0f}) | "
            f"{100 * s['collision']:.0f}% | {100 * s['freeze']:.0f}% | {100 * s['offroad']:.0f}% | "
            f"{100 * s['progress']:.0f}% | {s['time_to_goal']:.1f} | {s['min_dist']:.2f} | {s['rms_jerk']:.2f} | "
            f"{s['rms_lat_acc']:.2f} | {s['solve_mean_ms']:.1f} / {s['solve_p99_ms']:.1f} / {s['solve_max_ms']:.0f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["results/town02"])
    ap.add_argument("--history", default="checkpoints/planner_history.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--controllers", nargs="*", default=None)
    args = ap.parse_args()
    rows = load(args.results)
    present = [c for c in ORDER if any(r["controller"] == c for r in rows)]
    ctrls = args.controllers or present
    main_ctrls = [c for c in ["lab_mpc", "capsule_nmpc", "hybrid", "hybrid_noselector", "expert"] if c in ctrls]
    paired, n_scen = common_scenarios(rows, main_ctrls)
    figdir = os.path.join(args.out, "figures")
    os.makedirs(figdir, exist_ok=True)
    summ = summarize(paired, main_ctrls)
    out = dict(n_scenarios=n_scen, main=summ)
    md = [f"### Main comparison ({n_scen} paired held-out scenarios, CARLA Town02)\n", md_table(summ)]
    abl = [c for c in ctrls if c.startswith("hybrid_") and c != "hybrid_noselector"]
    if abl:
        base = ["hybrid", "hybrid_noselector"]
        ab_rows, n_ab = common_scenarios(rows, base + abl)
        s_ab = summarize(ab_rows, base + abl)
        out["ablation"] = s_ab
        md += [f"\n### Ablations ({n_ab} paired scenarios)\n", md_table(s_ab)]
    tpls, M, N = per_template(paired, main_ctrls)
    out["per_template"] = {t: {c: (None if np.isnan(M[i, j]) else float(M[i, j])) for j, c in enumerate(main_ctrls)}
                           for i, t in enumerate(tpls)}
    md += ["\n### Success rate per scenario template\n",
           "| Template | " + " | ".join(LABEL.get(c, c) for c in main_ctrls) + " |",
           "|---|" + "---|" * len(main_ctrls)]
    for i, t in enumerate(tpls):
        md.append(f"| {t} | " + " | ".join("–" if np.isnan(M[i, j]) else f"{100 * M[i, j]:.0f}% ({int(N[i, j])})"
                                           for j in range(len(main_ctrls))) + " |")
    if summ:
        fig_outcomes(summ, os.path.join(figdir, "outcomes.png"))
        fig_heatmap(tpls, M, N, main_ctrls, os.path.join(figdir, "success_by_template.png"))
        fig_latency(paired, main_ctrls, os.path.join(figdir, "latency.png"))
        fig_comfort(paired, main_ctrls, os.path.join(figdir, "comfort.png"))
    if os.path.exists(args.history):
        hist = json.load(open(args.history))
        fig_training(hist, os.path.join(figdir, "training.png"))
        out["training_final"] = hist[-1]
    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(md) + "\n")
    json.dump(out, open(os.path.join(args.out, "summary.json"), "w"), indent=2, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
