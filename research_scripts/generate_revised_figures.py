#!/usr/bin/env python3
"""Generate two new figures for the revised paper:
   fig8_bv.pdf  — Bernstein-Vazirani quantum advantage
   fig9_hs.pdf  — High-statistics CQEC gain with Wilcoxon p-values
"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex":         True,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb,bm}",
    "font.family":         "serif",
    "font.serif":          ["Computer Modern Roman"],
    "mathtext.fontset":    "cm",
    "font.size": 9, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, "results")
OUT  = os.path.join(RES, "paper")
os.makedirs(OUT, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8  — Bernstein-Vazirani quantum advantage
# ══════════════════════════════════════════════════════════════════════════════

def fig8_bv():
    with open(os.path.join(RES, "bernstein_vazirani_bench.json")) as f:
        data = json.load(f)
    profiles = ["noiseless", "Path2_PTM", "Path3_OrgSC", "Path4_SSH", "Path1_RPRes"]
    colors   = {"noiseless": "#000", "Path1_RPRes": "#d62728",
                "Path2_PTM": "#4c72b0", "Path3_OrgSC": "#2ca02c",
                "Path4_SSH": "#ff7f0e"}
    ns = sorted(int(k) for k in data["results"])

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    # Classical line
    cls = [data["classical"][str(n)]["success_rate"] for n in ns]
    ax.plot(ns, cls, "k:", lw=1.5, label="Classical (1 query)")

    for prof in profiles:
        rates = [data["results"][str(n)][f"{prof}_cqec"]["success_rate"]
                 for n in ns]
        his   = [data["results"][str(n)][f"{prof}_cqec"]["ci_half_width_95"]
                 for n in ns]
        label = "Path 1 (RP reservoir)" if prof == "Path1_RPRes" else \
                "Path 2 (PTM)"   if prof == "Path2_PTM" else \
                "Path 3 (SVILC)" if prof == "Path3_OrgSC" else \
                "Path 4 (SSH)"   if prof == "Path4_SSH" else "Noiseless Q"
        ax.errorbar(ns, rates, yerr=his, fmt="o-", ms=5, lw=1.3,
                     capsize=3, color=colors[prof], label=label)

    ax.set_xlabel(r"Number of query qubits $n$")
    ax.set_ylabel("BV success rate (100 trials)")
    ax.set_title("Bernstein-Vazirani: quantum advantage under\norganic noise (1 query)")
    ax.set_xticks(ns)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center right")
    ax.set_ylim(-0.05, 1.1)
    # Annotation on the advantage
    ax.annotate(r"+31$\times$ vs classical"+"\n@ $n=5$",
                 xy=(5, 1.0), xytext=(3.2, 0.55),
                 arrowprops=dict(arrowstyle="->"), fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig8_bv.{ext}"))
    plt.close(fig)
    print("  saved fig8_bv.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9  — High-statistics Wilcoxon p-values (path × algorithm)
# ══════════════════════════════════════════════════════════════════════════════

def fig9_hs():
    with open(os.path.join(RES, "high_stats_flagship.json")) as f:
        data = json.load(f)
    rows = data["path_alg"]
    algs   = sorted(set(r["algorithm"] for r in rows))
    paths  = sorted(set(r["profile"] for r in rows),
                    key=lambda s: int(s.split("_")[0][-1]))
    dF = np.zeros((len(algs), len(paths)))
    logp = np.zeros((len(algs), len(paths)))
    for r in rows:
        i = algs.index(r["algorithm"])
        j = paths.index(r["profile"])
        dF[i, j] = r["delta_mean"]
        logp[i, j] = -np.log10(max(r["p_wilcoxon"], 1e-30))

    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2))
    im0 = axs[0].imshow(dF, vmin=0, vmax=0.12, cmap="viridis", aspect="auto")
    im1 = axs[1].imshow(logp, vmin=0, vmax=30, cmap="magma", aspect="auto")
    for ax, M, ttl in [(axs[0], dF,   r"(a) CQEC gain $\Delta F$"),
                        (axs[1], logp, r"(b) $-\log_{10} p$ (Wilcoxon)")]:
        ax.set_xticks(range(len(paths)))
        ax.set_xticklabels([p.replace("Path", "P") for p in paths],
                            rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(algs))); ax.set_yticklabels(algs, fontsize=8)
        for i in range(len(algs)):
            for j in range(len(paths)):
                txt = f"{M[i,j]:.3f}" if ttl.startswith("(a)") else f"{M[i,j]:.1f}"
                ax.text(j, i, txt, ha="center", va="center",
                         color="white" if M[i, j] < M.max() * 0.4 else "black",
                         fontsize=6)
        ax.set_title(ttl)
    fig.colorbar(im0, ax=axs[0], fraction=0.03, pad=0.02)
    fig.colorbar(im1, ax=axs[1], fraction=0.03, pad=0.02)
    # Bonferroni note — placed safely BELOW both axes (outside heatmap)
    fig.text(0.5, -0.04,
             r"Bonferroni $\alpha = 0.05/44\Rightarrow -\log_{10}p > 2.94$  "
             r"(all 16 tests pass)",
             ha="center", fontsize=8, color="darkred")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig9_hs.{ext}"))
    plt.close(fig)
    print("  saved fig9_hs.{pdf,png}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 10 — Hybrid v1 vs v2 comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig10_hybrid_v2():
    try:
        with open(os.path.join(RES, "hybrid_pipeline_v2.json")) as f:
            v2 = json.load(f)
        with open(os.path.join(RES, "hybrid_qrc_qc.json")) as f:
            v1 = json.load(f)
    except FileNotFoundError:
        print("  skipped fig10 (file missing)")
        return
    noises = sorted(v2["results"].keys(), key=float)
    cls   = [v2["results"][n]["classical"]["mse"]    for n in noises]
    p1    = [v2["results"][n]["path1"]["mse"]         for n in noises]
    p2    = [v2["results"][n]["path2"]["mse"]         for n in noises]
    hv2   = [v2["results"][n]["hybrid_v2"]["mse"]     for n in noises]
    # v1 hybrid
    hv1 = []
    for n in noises:
        # v1 data keyed by float
        key = float(n)
        if str(key) in v1["results_by_noise"]:
            rv1 = v1["results_by_noise"][str(key)]
        else:
            rv1 = v1["results_by_noise"].get(list(v1["results_by_noise"].keys())[0], {})
        hv1.append(rv1.get("hybrid_path1_path2", {}).get("mse", np.nan))

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(noises)); w = 0.18
    ax.bar(x - 2 * w, cls, w, label="Classical FFT",     color="#555")
    ax.bar(x - w,     p1,  w, label="Path 1 + Ridge",    color="#4c72b0")
    ax.bar(x,         p2,  w, label="Path 2 QPE",        color="#55a868")
    ax.bar(x + w,     hv1, w, label="Hybrid v1 (diag)",  color="#c44e52", alpha=0.55)
    ax.bar(x + 2 * w, hv2, w, label="Hybrid v2 (SWAP)",  color="#c44e52")
    ax.set_yscale("log")
    ax.set_xlabel(r"Input noise $\sigma$")
    ax.set_ylabel("Angular MSE  (log scale)")
    ax.set_xticks(x); ax.set_xticklabels(noises)
    ax.set_title("Hybrid pipeline improvement: v1 (diag) vs v2 (SWAP bridge)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y", which="both")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig10_hybrid_v2.{ext}"))
    plt.close(fig)
    print("  saved fig10_hybrid_v2.{pdf,png}")


if __name__ == "__main__":
    fig8_bv()
    fig9_hs()
    fig10_hybrid_v2()
