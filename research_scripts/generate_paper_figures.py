#!/usr/bin/env python3
"""
generate_paper_figures.py
=========================
Generate publication-quality figures (PNG + PDF, 300 dpi) for the paper
from all collected JSON data.

Outputs to results/paper/fig*.{pdf,png}.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex":         True,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb,bm}",
    "font.family":         "serif",
    "font.serif":          ["Computer Modern Roman"],
    "mathtext.fontset":    "cm",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "paper")
os.makedirs(OUT, exist_ok=True)


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        print(f"  WARN: missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  saved fig {name}.{{pdf,png}}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — γ-sweep: fidelity and CQEC gain (main physics result)
# ══════════════════════════════════════════════════════════════════════════════

def fig1_gamma_sweep():
    data = load("organic_benchmarks_extended.json")
    if not data:
        return
    sweep = data["gamma_sweep_with_ci"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    colors = {"QKAN": "#1f77b4", "qDRIFT": "#ff7f0e",
              "QPE": "#2ca02c", "Shor_Regev": "#d62728"}
    for name, rows in sweep.items():
        g = [r["gamma"] for r in rows]
        fn = [r["fid_noisy_mean"] for r in rows]
        fc = [r["fid_cqec_mean"]  for r in rows]
        ci_n = [r["fid_noisy_ci"] for r in rows]
        ci_c = [r["fid_cqec_ci"]  for r in rows]
        color = colors.get(name, "black")
        axs[0].errorbar(g, fn, yerr=ci_n, fmt="o--", ms=3, lw=0.8,
                         capsize=2, color=color, alpha=0.6,
                         label=f"{name} (noisy)")
        axs[0].errorbar(g, fc, yerr=ci_c, fmt="s-", ms=3, lw=1.2,
                         capsize=2, color=color,
                         label=f"{name} (CQEC)")
        df = np.array(fc) - np.array(fn)
        axs[1].plot(g, df, "o-", ms=4, lw=1.2, color=color, label=name)
    for ax in axs:
        ax.axvline(0.3, color="gray", ls=":", lw=1, label=r"$\gamma_c = 0.3$")
        ax.set_xscale("log")
        ax.set_xlabel(r"Effective dephasing rate $\gamma$")
    axs[0].set_ylabel(r"Fidelity $F$")
    axs[0].set_title(r"(a) Fidelity vs $\gamma$")
    axs[0].legend(fontsize=6, ncol=2, loc="lower left")
    axs[1].set_ylabel(r"CQEC gain $\Delta F$")
    axs[1].set_title("(b) CQEC recovery $F_{\\mathrm{cqec}} - F_{\\mathrm{noisy}}$")
    axs[1].legend(fontsize=7)
    plt.tight_layout()
    save(fig, "fig1_gamma_sweep")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Algorithm × Path fidelity heatmap
# ══════════════════════════════════════════════════════════════════════════════

def fig2_algorithm_path_heatmap():
    data = load("organic_benchmarks_extended.json")
    if not data:
        return
    rows = data["algorithm_benchmarks_all_paths"]
    algs   = sorted(set(r["algorithm"] for r in rows))
    paths  = sorted(set(r["profile"]   for r in rows),
                     key=lambda s: int(s.split("_")[0].replace("Path", "")))
    M_noisy = np.zeros((len(algs), len(paths)))
    M_cqec  = np.zeros((len(algs), len(paths)))
    for r in rows:
        i = algs.index(r["algorithm"])
        j = paths.index(r["profile"])
        M_noisy[i, j] = r["fid_noisy_mean"]
        M_cqec [i, j] = r["fid_cqec_mean"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    im0 = axs[0].imshow(M_noisy, vmin=0.8, vmax=1.0, aspect="auto",
                         cmap="viridis")
    im1 = axs[1].imshow(M_cqec,  vmin=0.8, vmax=1.0, aspect="auto",
                         cmap="viridis")
    for ax, M, ttl in [(axs[0], M_noisy, "(a) Noisy"),
                       (axs[1], M_cqec , "(b) CQEC")]:
        ax.set_xticks(range(len(paths)))
        ax.set_xticklabels([p.replace("Path", "P") for p in paths], rotation=45,
                            ha="right", fontsize=7)
        ax.set_yticks(range(len(algs)))
        ax.set_yticklabels(algs, fontsize=8)
        for i in range(len(algs)):
            for j in range(len(paths)):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                         color="white" if M[i, j] < 0.9 else "black",
                         fontsize=6)
        ax.set_title(ttl)
    fig.colorbar(im1, ax=axs, fraction=0.03, pad=0.02, label="Fidelity")
    save(fig, "fig2_alg_path_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — MNIST 5-fold CV accuracy
# ══════════════════════════════════════════════════════════════════════════════

def fig3_mnist():
    data = load("organic_benchmarks_extended.json")
    if not data:
        return
    r = data["mnist_full"]
    labels  = ["Classical", "Ideal QM", "P1 noisy", "P1 CQEC",
               "P2 noisy", "P2 CQEC", "P3 noisy", "P3 CQEC",
               "P4 noisy", "P4 CQEC"]
    keys    = ["classical", "ideal_quantum",
               "noisy_Path1_RadicalPairRes",  "cqec_Path1_RadicalPairRes",
               "noisy_Path2_PTMRadical",       "cqec_Path2_PTMRadical",
               "noisy_Path3_OrganicSC_SVILC",  "cqec_Path3_OrganicSC_SVILC",
               "noisy_Path4_SSHSoliton",       "cqec_Path4_SSHSoliton"]
    accs = [r[k]["acc_mean"] for k in keys]
    cis  = [r[k]["acc_ci"]   for k in keys]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    colors  = ["#555"] + ["#aaa"] + ["#4c72b0", "#1f5fa6"] * 4
    xs = np.arange(len(labels))
    bars = ax.bar(xs, accs, yerr=cis, color=colors, capsize=3, alpha=0.85)
    ax.set_ylim(0.93, 1.0)
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title("MNIST 10-class classification (sklearn digits, 1797 samples)")
    ax.axhline(accs[0], color="gray", ls=":", lw=0.7, label="classical baseline")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, "fig3_mnist")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — κ-BEDT-TTF lattice physics
# ══════════════════════════════════════════════════════════════════════════════

def fig4_kbedt():
    data = load("svilc_kbedt_lattice.json")
    if not data:
        return
    cpl  = data["coupling_vs_distance"]
    fc   = data["feed_current_activation"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    xs = [e["distance"] for e in cpl]
    ys = [e["V_coupling"] for e in cpl]
    axs[0].plot(xs, ys, "o-", color="#2ca02c")
    axs[0].axhline(0, color="gray", lw=0.4)
    axs[0].set_xlabel("SVQ separation r_x / a")
    axs[0].set_ylabel(r"$V_{\alpha\Upsilon}$  (phase frustration)")
    axs[0].set_title("(a) Two-SVQ coupling vs distance")
    axs[0].grid(alpha=0.3)

    amps = [e["feed_amp"]    for e in fc]
    vs   = [abs(e["V_coupling"]) for e in fc]
    axs[1].plot(amps, vs, "s-", color="#d62728")
    axs[1].set_xlabel("External feed current amplitude J_ext")
    axs[1].set_ylabel(r"$|V_{\alpha\Upsilon}|$  (activated coupling)")
    axs[1].set_title("(b) Feed-current activation at r = 10 a")
    axs[1].set_yscale("log")
    axs[1].grid(alpha=0.3, which="both")
    save(fig, "fig4_kbedt_lattice")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Regev vs Shor gate-count scaling
# ══════════════════════════════════════════════════════════════════════════════

def fig5_regev_scaling():
    data = load("regev_classical_postprocess.json")
    if not data:
        return
    sc = data["gate_count_scaling"]
    ns = [g["n_bits"] for g in sc]
    shor = [g["gate_count_shor"]  for g in sc]
    regev= [g["gate_count_regev"] for g in sc]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.loglog(ns, shor,  "o-", label="Shor (classical baseline)", color="#1f77b4")
    ax.loglog(ns, regev, "s-", label="Regev (2024)",              color="#d62728")
    # Annotate speedup at RSA-2048
    idx = ns.index(2048)
    sp = shor[idx] / regev[idx]
    ax.annotate(rf"{sp:.0f}$\times$ speedup"+"\nat RSA-2048",
                 xy=(2048, regev[idx]),
                 xytext=(400, regev[-1] * 3), fontsize=8,
                 arrowprops=dict(arrowstyle="->", lw=0.5))
    ax.set_xlabel("n (bits of integer to factor)")
    ax.set_ylabel("Quantum-circuit gate count")
    ax.set_title("Shor vs Regev scaling (Regev eq. 6)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    save(fig, "fig5_regev_scaling")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Photoswitch CZ gate fidelity
# ══════════════════════════════════════════════════════════════════════════════

def fig6_photoswitch():
    data = load("organic_benchmarks_extended.json")
    if not data:
        return
    p = data["photoswitch_gate"]
    # The JSON may contain the initial (bad) or re-run result;
    # re-run data lives separately:
    # we prefer the explicit re-run saved result if present
    t = p["gate_time_ns"]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for k, v in p.items():
        if isinstance(v, dict) and "fid" in v:
            ax.plot(t, v["fid"], "o-", label=k.replace("_", " "), ms=3)
    ax.set_xlabel("Gate time (ns)")
    ax.set_ylabel("CZ-gate fidelity F")
    ax.set_title("Diarylethene photoswitch 2-qubit CZ gate")
    ax.axvline(5.82, color="gray", ls=":", lw=0.7, label="t_opt = 5.82 ns")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    save(fig, "fig6_photoswitch")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Hybrid nonlinear denoising
# ══════════════════════════════════════════════════════════════════════════════

def fig7_hybrid_nonlinear():
    data = load("hybrid_nonlinear_denoising.json")
    if not data:
        print("  fig7 skipped (no hybrid_nonlinear_denoising.json yet)")
        return
    res = data["results"]
    noises = list(res.keys())
    pipe_names = ["A_classical", "B_path2", "C_path1_res", "D_hybrid"]
    labels = ["Classical", "Path 2", "Path 1 reservoir", r"Hybrid P1$\to$P2"]
    accs = {p: [] for p in pipe_names}
    cis  = {p: [] for p in pipe_names}
    for nn in noises:
        for p in pipe_names:
            accs[p].append(res[nn]["pipelines"][p]["acc_mean"])
            cis [p].append(res[nn]["pipelines"][p]["acc_ci"])
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    x = np.arange(len(noises))
    width = 0.2
    colors = ["#555", "#4c72b0", "#55a868", "#c44e52"]
    for i, p in enumerate(pipe_names):
        ax.bar(x + (i - 1.5) * width, accs[p], width,
                yerr=cis[p], capsize=2, label=labels[i],
                color=colors[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(noises, rotation=20)
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("MNIST denoising: pipeline comparison under corruption")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    save(fig, "fig7_hybrid_nonlinear")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Generating paper figures …")
    fig1_gamma_sweep()
    fig2_algorithm_path_heatmap()
    fig3_mnist()
    fig4_kbedt()
    fig5_regev_scaling()
    fig6_photoswitch()
    fig7_hybrid_nonlinear()
    print(f"\n  All figures → {OUT}")


if __name__ == "__main__":
    main()
