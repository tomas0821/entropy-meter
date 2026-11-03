"""make_figures.py: the paper's three figures and Table 1 from results/ (draft styling).

    python make_figures.py          # -> figures/fig{1,2,3}.{pdf,png}, figures/table1.md (or set FIGDIR)

Fig. 1  e(beta) - e0 against beta: MC (L = 96) on the exact infinite-lattice curves; the shaded
        area is ln 2 - S0 (AFM) and ln 2 (FM).
Fig. 2  S(T) from the same data, S = ln 2 + beta e - int_0^beta e dbeta', against the exact curve.
Fig. 3  ln(Omega)/N against 1/N: exact ground-state counts and MC means (SEM over 8 seeds); inset: the
        antiferromagnet near 1/N -> 0 with the MC-only extrapolation S0(L) = S0_inf + b/L^2.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.environ.get("FIGDIR", os.path.join(HERE, "figures"))
EX = json.load(open(os.path.join(HERE, "results", "exact", "exact_refs.json")))
S0_INF = EX["S0_inf"]; S0_L = {int(k): v for k, v in EX["S0_L"].items()}
BF = np.array(EX["beta_fine"])

# palette: dataviz reference slots 1-2 (validated light-mode pair), text inks, recessive grid
AFM, FM = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
                     "axes.labelcolor": INK, "legend.frameon": False, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4})
MK = dict(ms=4.2, mec=SURF, mew=0.8, ls="none", zorder=3)

def runs(L, J, db, h=0.0, sub="production"):
    rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "results", sub, f"L{L}_J{J:g}_h{h:g}_db{db:g}_s*.json")))]
    assert rs, (L, J, db, h)
    return rs

def simpson(y, dx):
    return dx / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())

def entropy_curve(beta, e):
    """S at every even grid index: ln 2 + beta e - int_0^beta e (Simpson on [0, beta_2m])."""
    db = beta[1] - beta[0]
    idx = np.arange(2, len(beta), 2)
    integ = np.array([simpson(e[:i + 1], db) for i in idx])
    return beta[idx], np.log(2) + beta[idx] * e[idx] - integ

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

import exact_refs as XR                      # exact infinite-lattice FM free energy (for the FM S(T) curve)
from analyze import fit_inv_N                # the students' MC-only extrapolation S0(L) = S0_inf + b/L^2
from matplotlib.lines import Line2D

def pick_T(bb, targets):
    """Indices of the grid points whose T = 1/beta is nearest each target T (unique, sorted)."""
    T = 1 / bb
    return np.array(sorted({int(np.argmin(abs(T - t))) for t in targets}))

def fm_s_exact(T):
    return np.array([XR.fm_lnZ(1 / t) + XR.fm_e(1 / t) / t for t in T])

def fig1():
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    m = BF <= 3.0
    ea, ef = np.array(EX["afm_e_inf"])[m] + 1, np.array(EX["fm_e_inf"])[m] + 3
    plt.rcParams["hatch.linewidth"] = 0.6
    ax.fill_between(BF[m], 0, ef, facecolor="none", edgecolor=FM, hatch="////", lw=0, alpha=0.55, zorder=1)
    ax.fill_between(BF[m], 0, ea, color=AFM, alpha=0.22, lw=0, zorder=1)
    ax.plot(BF[m], ef, color=FM, lw=1.2, ls=(0, (4, 2)), zorder=2); ax.plot(BF[m], ea, color=AFM, lw=1.2, zorder=2)
    ra = runs(96, 1, 0.1); b = np.array(ra[0]["beta"]); E = np.array([r["e"] for r in ra]).mean(0) + 1
    k = (b > 0) & (b <= 3.0)                 # beta = 0 is set by hand (e(0) = 0), not simulated
    ax.plot(b[k], E[k], "o", color=AFM, **MK)
    rf = runs(96, -1, 0.0125); bf = np.array(rf[0]["beta"]); Ef = np.array([r["e"] for r in rf]).mean(0) + 3
    on01 = np.isclose(np.mod(bf + 1e-9, 0.1), 0, atol=1e-6)
    near_c = (bf > 0.2) & (bf < 0.4) & np.isclose(np.mod(bf + 1e-9, 0.025), 0, atol=1e-6)
    k = (bf > 0) & (bf <= 3.0) & (on01 | near_c)
    ax.plot(bf[k], Ef[k], "s", color=FM, **MK)
    arrow = dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=0)
    ax.annotate("ferromagnet: area $=\\ln 2$", xy=(0.12, 1.15), xytext=(0.62, 2.15), color=INK, arrowprops=arrow)
    ax.annotate("antiferromagnet:\narea $=\\ln 2 - S_0/N = 0.370$", xy=(0.80, 0.07), xytext=(1.25, 0.85), color=INK, arrowprops=arrow)
    ax.set_xlim(0, 3); ax.set_ylim(0, 3.05)
    ax.set_xlabel(r"$\beta |J|$"); ax.set_ylabel(r"$[e(\beta)-e_0]/|J|$")
    h = [Line2D([], [], color=AFM, marker="o", mec=SURF, ms=4.2, lw=1.2, label="antiferromagnet ($J>0$)"),
         Line2D([], [], color=FM, marker="s", mec=SURF, ms=4.2, lw=1.2, ls=(0, (4, 2)), label="ferromagnet ($J<0$)")]
    ax.legend(handles=h, loc="upper right", handletextpad=0.4, handlelength=2.4)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig1_energy_area")

def fig2():
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.axhline(np.log(2), color=GRID, lw=0.8, zorder=0); ax.text(5.95, np.log(2) + 0.012, r"$\ln 2$", ha="right", color=INK2)
    ax.axhline(S0_INF, color=INK2, lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.text(6.1, S0_INF - 0.012, r"$S_0/N = 0.3231$ (Wannier)", ha="right", va="top", color=INK2)
    T = 1 / BF[1:]; k = T <= 7.0
    ax.plot(T[k], np.array(EX["afm_s_inf"][1:])[k], color=AFM, lw=1.2, zorder=2)
    Tf = np.r_[np.linspace(0.08, 3.5, 120), np.linspace(3.5, 3.8, 60), np.linspace(3.8, 6.2, 60)]
    ax.plot(Tf, fm_s_exact(Tf), color=FM, lw=1.2, ls=(0, (4, 2)), zorder=2)
    ra = runs(96, 1, 0.05); b = np.array(ra[0]["beta"])
    bb, _ = entropy_curve(b, np.array(ra[0]["e"])); S = np.array([entropy_curve(b, np.array(r["e"]))[1] for r in ra]).mean(0)
    i = pick_T(bb, np.arange(0.25, 6.01, 0.25)); i = i[1 / bb[i] <= 6.05]
    ax.plot(1 / bb[i], S[i], "o", color=AFM, **MK)
    rf = runs(96, -1, 0.0125); bf = np.array(rf[0]["beta"])
    bbf, _ = entropy_curve(bf, np.array(rf[0]["e"])); Sf = np.array([entropy_curve(bf, np.array(r["e"]))[1] for r in rf]).mean(0)
    j = pick_T(bbf, np.r_[np.arange(0.5, 6.01, 0.5), np.arange(3.3, 4.01, 0.1)]); j = j[1 / bbf[j] <= 6.05]
    ax.plot(1 / bbf[j], Sf[j], "s", color=FM, **MK)
    ax.set_xlim(0, 6.2); ax.set_ylim(0, 0.75)
    ax.set_xlabel(r"$k_BT/|J|$"); ax.set_ylabel(r"$S/(Nk_B)$")
    h = [Line2D([], [], color=AFM, marker="o", mec=SURF, ms=4.2, lw=1.2, label="antiferromagnet"),
         Line2D([], [], color=FM, marker="s", mec=SURF, ms=4.2, lw=1.2, ls=(0, (4, 2)), label="ferromagnet"),
         Line2D([], [], color=INK2, marker="o", mfc=INK2, mec=SURF, ms=4.2, ls="none", label="Monte Carlo, $L=96$"),
         Line2D([], [], color=INK2, lw=1.2, label="exact, $L\\to\\infty$")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0.01, 0.07), handletextpad=0.4, handlelength=2.4)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig2_entropy_T")

def mc_means(J, db, Ls=(12, 24, 48, 96)):
    out = []
    for L in Ls:
        s = np.array([r["S0"] for r in runs(L, J, db)])
        out.append((L, s.mean(), s.std(ddof=1) / np.sqrt(len(s))))
    return np.array(out)

def fig3():
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    Ls = sorted(L for L in S0_L if L <= 96)
    x = np.array([1 / L**2 for L in Ls]); xx = np.linspace(0, 1 / 36 * 1.06, 50)
    ax.plot(xx, S0_INF + 0.8676 * xx, color=AFM, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.plot(xx, np.log(2) * xx, color=FM, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.plot(x, [S0_L[L] for L in Ls], "o", ms=4.6, mfc="white", mec=AFM, mew=1.0, ls="none", zorder=2)
    ax.plot(x, np.log(2) * x, "s", ms=4.6, mfc="white", mec=FM, mew=1.0, ls="none", zorder=2)
    A, F = mc_means(1, 0.05), mc_means(-1, 0.0125)
    for M, col, mk in ((A, AFM, "o"), (F, FM, "s")):
        ax.errorbar(1 / M[:, 0]**2, M[:, 1], M[:, 2], fmt=mk, color=col, ms=3.4, mec=col, elinewidth=0.8, capsize=0, zorder=4)
    for L in (6, 12):
        ax.text(1 / L**2, S0_L[L] + 0.010, f"$L={L}$", ha="center", fontsize=7.5, color=INK2)
    ax.text(0.0285, S0_INF + 0.8676 * 0.0285 - 0.036, "antiferromagnet\n$\\ln\\Omega/N \\to 0.3231$", ha="right", va="top", color=INK, fontsize=8)
    ax.text(0.0285, np.log(2) * 0.0285 + 0.022, "ferromagnet ($\\Omega=2$): $\\ln 2/N \\to 0$", ha="right", va="bottom", color=INK, fontsize=8)
    ax.set_xlim(0, 1 / 36 * 1.08); ax.set_ylim(-0.01, 0.37)
    ax.set_xlabel(r"$1/N = 1/L^2$"); ax.set_ylabel(r"$\ln\Omega/N = S_0/(Nk_B)$")
    h = [Line2D([], [], color=INK2, marker="o", mfc="white", mec=INK2, ms=4.6, ls="none", label="exact ground-state count"),
         Line2D([], [], color=INK2, marker="o", mfc=INK2, mec=INK2, ms=3.4, ls="none", label="Monte Carlo (8 seeds)")]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.24), fontsize=7.5, handletextpad=0.3)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    # inset: the antiferromagnet near 1/N -> 0, where the error bars and the extrapolation are visible
    ai = ax.inset_axes([0.065, 0.22, 0.40, 0.44])
    X1 = 0.00185; xs = np.linspace(0, X1, 20)
    ai.plot(xs, S0_INF + 0.8676 * xs, color=AFM, lw=0.6, ls=(0, (3, 2)), zorder=1)
    Le = [L for L in (24, 30, 36, 48, 60, 72, 96, 192, 384)]
    ai.plot([1 / L**2 for L in Le], [S0_L[L] for L in Le], "o", ms=3.8, mfc="white", mec=AFM, mew=0.9, ls="none", zorder=2)
    f = fit_inv_N(A[:, 0], A[:, 1], A[:, 2])
    ai.plot(xs, f["S0_inf"] + f["b"] * xs, color=INK2, lw=0.6, zorder=3)
    ai.errorbar(1 / A[1:, 0]**2, A[1:, 1], A[1:, 2], fmt="o", color=AFM, ms=3.0, mec=AFM, elinewidth=0.8, capsize=1.5, zorder=4)
    ai.errorbar([0], [f["S0_inf"]], [f["S0_inf_err_scaled"]], fmt="D", color=INK, ms=3.2, elinewidth=0.8, capsize=1.5, zorder=5, clip_on=False)
    ai.text(0.00004, 0.32488, f"\u25c6 MC fit: {f['S0_inf']:.5f}({round(f['S0_inf_err_scaled'] * 1e5):d})\nexact (Wannier): 0.3230659", fontsize=6.3, color=INK, va="top")
    for L, ha, dx in ((24, "right", -0.00003), (48, "left", 0.00004), (96, "left", 0.00004)):
        ai.text(1 / L**2 + dx, S0_L[L] - 0.00012, f"$L={L}$", ha=ha, va="top", fontsize=6.3, color=INK2)
    ai.set_xlim(-0.00004, X1); ai.set_ylim(0.32275, 0.32495)
    ai.set_xticks([0, 0.0005, 0.0010, 0.0015]); ai.set_xticklabels(["0", "0.5", "1.0", "1.5"], fontsize=6.3)
    ai.set_yticks([0.3230, 0.3235, 0.3240, 0.3245]); ai.tick_params(labelsize=6.3, length=2); ai.yaxis.tick_right()
    ai.set_xlabel(r"$1/N\ (10^{-3})$", fontsize=6.5, labelpad=1)
    for s in ("top", "right"):
        ai.spines[s].set_visible(True)
    ai.set_facecolor("white")
    ax.indicate_inset_zoom(ai, edgecolor=INK2, lw=0.5, alpha=0.8)
    save(fig, "fig3_one_over_N")

def table1():
    rows = ["| L | MC, dβ = 0.1 | MC, dβ = 0.05 | exact finite-L | exact L → ∞ |", "|---|---|---|---|---|"]
    M = {db: mc_means(1, db) for db in (0.1, 0.05)}
    for i, L in enumerate((12, 24, 48, 96)):
        cells = [f"{M[db][i, 1]:.5f} ± {M[db][i, 2]:.5f}" for db in (0.1, 0.05)]
        rows.append(f"| {L} | {cells[0]} | {cells[1]} | {S0_L[L]:.5f} | {S0_INF:.5f} |")
    fits = {db: fit_inv_N(M[db][:, 0], M[db][:, 1], M[db][:, 2]) for db in (0.1, 0.05)}
    rows.append("| fit S0∞ + b/L² (L = 12–96) | " + " | ".join(f"{fits[db]['S0_inf']:.5f} ± {fits[db]['S0_inf_err_scaled']:.5f} (b = {fits[db]['b']:.3f} ± {fits[db]['b_err_scaled']:.3f}; χ²/dof = {fits[db]['chi2']:.1f}/{fits[db]['dof']}; errors × {fits[db]['err_scale']:.2f})" for db in (0.1, 0.05))
                + f" | b = 0.8676 (exact counts) | {S0_INF:.7f} |")
    g = EX["grid_error_inf"]
    rows += ["", "Error budget (per L, dβ = 0.1 / 0.05):",
             f"- grid (Simpson, exact L = ∞ curve): {g['0.1']['afm_simpson']:+.1e} / {g['0.05']['afm_simpson']:+.1e}; the trapezoid rule would give {g['0.1']['afm_trapezoid']:+.1e} / {g['0.05']['afm_trapezoid']:+.1e}",
             f"- truncation at β = 6: e − e0 = {EX['tail_afm_beta6']:.0e} (the integrated tail is smaller still)",
             "- statistics: SEM over 8 seeds (table entries)",
             "- finite size: exact S0(L) − S0(∞) = 0.8676/L² (L = 96: 9.4e-5)"]
    open(os.path.join(FIG, "table1.md"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))

if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    for f in (fig1, fig2, fig3, table1):
        try:
            f(); print("ok", f.__name__)
        except AssertionError as e:
            print("skip", f.__name__, "(missing runs:", e, ")")
