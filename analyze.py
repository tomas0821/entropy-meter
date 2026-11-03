"""analyze.py: compare the per-seed results of entropy_meter.py with the exact references.

    python analyze.py stage1          # validation gate (Stage 1): writes results/stage1_validation.md
    python analyze.py summary         # every group of runs: S0 by L, grid checks, field checks -> results/summary.json

Errors are the standard error of the mean over independent seeds (SEM, ddof = 1).
Needs results/exact/exact_refs.json (python exact_refs.py).
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EX = json.load(open(os.path.join(HERE, "results", "exact", "exact_refs.json")))
S0_INF = EX["S0_inf"]
S0_L = {int(k): v for k, v in EX["S0_L"].items()}
BF = np.array(EX["beta_fine"])

def simpson(y, dx):
    assert len(y) % 2 == 1
    return dx / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())

def exact_inf(beta, J):
    """Infinite-lattice exact e at the given betas (all on the 0.025 grid of exact_refs)."""
    idx = np.rint(np.asarray(beta) / 0.025).astype(int)
    assert np.allclose(BF[idx], beta)
    return np.array(EX["afm_e_inf" if J > 0 else "fm_e_inf"])[idx]

def grid_error_inf(J, db):
    g = EX["grid_error_inf"].get(str(db)) or EX["grid_error_inf"].get(repr(db))
    return None if g is None else g["afm_simpson" if J > 0 else "fm_simpson"]

def load(pattern):
    groups = defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f))
        groups[(r["L"], r["J"], r["h"], r["dbeta"], r["bmax"])].append(r)
    return groups

def mean_sem(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)), len(x)

def s0_subsampled(runs, k):
    """S0 per seed from every k-th beta point (Simpson needs an odd number of points)."""
    out = []
    for r in runs:
        e, b = np.array(r["e"])[::k], np.array(r["beta"])[::k]
        if len(e) % 2 == 0 or not np.isclose(b[-1], r["bmax"]):
            return None
        out.append(np.log(2) - simpson(e - r["e0"], r["dbeta"] * k))
    return np.array(out)

def exact_target(L, J, h):
    if h != 0:                                           # field, 0 < |h| < 6J: up-up-down state, 3-fold degenerate on the
        return np.log(3) / L**2                          # torus (which sublattice is down), so S0(L) = ln 3 / N -> 0
    return S0_L[L] if J > 0 else np.log(2) / L**2

def stage1():
    lines = ["# Stage 1 validation (entropy_meter.py v0.1)", ""]
    G = load(os.path.join(HERE, "results", "production", "L*_J1_h0_db0.1_s*.json"))
    ok = True
    bias = grid_error_inf(1.0, 0.1)
    lines += [f"Simpson grid error at dβ = 0.1 (exact, L = ∞): {bias:+.2e}. Target = exact finite-L S0 + this error.", "",
              "## S0 by L (dβ = 0.1, β ≤ 6, 8 seeds; SEM over seeds)", "",
              "| L | seeds | S0 (MC) | SEM | exact S0(L) | target | z | tail e(6)−e0 (max) |", "|---|---|---|---|---|---|---|---|"]
    for key in sorted(G):
        L, J, h, db, bmax = key; runs = G[key]
        m, se, n = mean_sem([r["S0"] for r in runs])
        tgt = S0_L[L] + bias
        z = (m - tgt) / se
        ok &= abs(z) < 2
        lines.append(f"| {L} | {n} | {m:.6f} | {se:.6f} | {S0_L[L]:.6f} | {tgt:.6f} | {z:+.2f} | {max(r['tail'] for r in runs):.1e} |")
    lines += ["", "Gate: |z| < 2 at every L → " + ("**PASS**" if ok else "**FAIL**"), ""]
    # L = 48 energy against the infinite-lattice exact curve
    k48 = (48, 1.0, 0.0, 0.1, 6.0)
    if k48 in G:
        runs = G[k48]; b = np.array(runs[0]["beta"])
        E = np.array([r["e"] for r in runs]); m = E.mean(0); se = E.std(0, ddof=1) / np.sqrt(len(runs))
        d = m - exact_inf(b, 1.0)
        integ = simpson(d, 0.1)
        integ_se = np.std([simpson(np.array(r["e"]) - exact_inf(b, 1.0), 0.1) for r in runs], ddof=1) / np.sqrt(len(runs))
        expect = S0_INF - S0_L[48]
        lines += ["## L = 48: e(β) against the exact L = ∞ curve (Houtappel)", "",
                  f"- max |e_MC − e_∞| = {np.abs(d).max():.2e} at β = {b[np.abs(d).argmax()]:.1f} (pilot: ≤ 1.3e-4). "
                  f"Per-β z-scores are not used: the finite-size part is not noise.",
                  f"- ∫(e_MC − e_∞) dβ = {integ:+.3e} ± {integ_se:.1e}; the TI identity at L = 48 and L = ∞ predicts "
                  f"S0_∞ − S0(48) = {expect:+.3e} (z = {(integ - expect) / integ_se:+.2f}).", "",
                  "| β | e_MC | SEM | e_∞ exact | e_MC − e_∞ |", "|---|---|---|---|---|"]
        for i in list(range(1, 11)) + list(range(12, 31, 2)) + [40, 50, 60]:
            lines.append(f"| {b[i]:.1f} | {m[i]:.6f} | {se[i]:.1e} | {exact_inf(b[i:i+1], 1.0)[0]:.6f} | {d[i]:+.1e} |")
        lines.append("")
    # L = 12 energy against the exact finite-torus transfer matrix
    k12 = (12, 1.0, 0.0, 0.1, 6.0)
    if k12 in G and "tm12_afm_e" in EX:
        runs = G[k12]; b = np.array(runs[0]["beta"])
        btm = np.array(EX["tm12_afm_beta"]); etm = np.array(EX["tm12_afm_e"])
        idx = [int(np.argmin(abs(btm - x))) for x in b]; assert np.allclose(btm[idx], b)
        E = np.array([r["e"] for r in runs]); m = E.mean(0); se = E.std(0, ddof=1) / np.sqrt(len(runs))
        with np.errstate(divide="ignore", invalid="ignore"):   # SEM = 0 where every seed sits in a ground state
            t = (m - etm[idx])[1:] / se[1:]
        good = (etm[idx][1:] + 1) > 1e-3          # below this, per-seed means are counts of a few excitations
        lines += ["## L = 12: e(β) against the exact 12 × 12 torus (row transfer matrix)", "",
                  f"- per-β t = (e_MC − e_exact)/SEM over β with e − e0 > 1e-3 ({good.sum()} points): rms t = {np.sqrt(np.mean(t[good]**2)):.2f} "
                  f"(expected ≈ 1.18 for 7 d.o.f.), max |t| = {np.abs(t[good]).max():.2f} at β = {b[1:][good][np.abs(t[good]).argmax()]:.1f}",
                  f"- all points: max |e_MC − e_exact| = {np.abs(m - etm[idx]).max():.1e}", ""]
    out = "\n".join(lines)
    open(os.path.join(HERE, "results", "stage1_validation.md"), "w").write(out + "\n")
    print(out)

def summary():
    S = {}
    for sub in ("production", "field"):
        G = load(os.path.join(HERE, "results", sub, "*.json"))
        for key in sorted(G):
            L, J, h, db, bmax = key; runs = G[key]
            tgt = exact_target(L, J, h)
            row = dict(L=L, J=J, h=h, dbeta=db, bmax=bmax, seeds=len(runs), nmeas=runs[0]["nmeas"], neq=runs[0]["neq"],
                       exact=tgt, tail_max=max(r["tail"] for r in runs))
            for k in (1, 2, 4, 8):
                s = s0_subsampled(runs, k)
                if s is not None:
                    m, se, n = mean_sem(s)
                    g = grid_error_inf(J, round(db * k, 6)) if h == 0 else None
                    row[f"S0_db{db * k:g}"] = dict(mean=m, sem=se, grid_error_inf=g,
                                                    z_vs_exact_plus_grid=None if g is None else (m - tgt - g) / se,
                                                    z_vs_exact=(m - tgt) / se)
            if h != 0:
                E = np.array([r["e"] for r in runs]); M = np.array([r["m"] for r in runs])
                row["e_end"], row["e0"] = mean_sem(E[:, -1])[:2], runs[0]["e0"]
                row["m_end"] = mean_sem(M[:, -1])[:2]
            S[f"{sub}/L{L}_J{J:g}_h{h:g}_db{db:g}"] = row
            print(f"{sub:10s} L={L:3d} J={J:+g} h={h:g} dβ={db:g} n={len(runs)} exact={tgt:.6f} | " +
                  " | ".join(f"dβ={kk[5:]}: {v['mean']:.6f}±{v['sem']:.6f} (z={v['z_vs_exact']:+.1f}"
                             + (f", z_grid={v['z_vs_exact_plus_grid']:+.1f})" if v['z_vs_exact_plus_grid'] is not None else ")")
                             for kk, v in row.items() if kk.startswith("S0_db")))
    S["extrapolation"] = extrapolate()
    json.dump(S, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1)

def fit_inv_N(Ls, y, se):
    """Weighted least squares S0(L) = S0_inf + b / L^2 (what a student would do with the MC means alone)."""
    Ls, y, se = (np.asarray(v, float) for v in (Ls, y, se))
    X = np.c_[np.ones_like(Ls), 1 / Ls**2]; W = 1 / se**2
    C = np.linalg.inv(X.T @ (X * W[:, None])); p = C @ X.T @ (W * y)
    chi2, dof = float(np.sum(W * (y - X @ p)**2)), len(Ls) - 2
    k = np.sqrt(max(1.0, chi2 / dof)) if dof > 0 else 1.0     # inflate the errors when chi2/dof > 1
    return dict(S0_inf=p[0], S0_inf_err=np.sqrt(C[0, 0]), b=p[1], b_err=np.sqrt(C[1, 1]),
                chi2=chi2, dof=dof, err_scale=k, S0_inf_err_scaled=k * np.sqrt(C[0, 0]), b_err_scaled=k * np.sqrt(C[1, 1]),
                Ls=Ls.astype(int).tolist(), z_vs_exact=(p[0] - S0_INF) / (k * np.sqrt(C[0, 0])))

def extrapolate():
    """AFM, h = 0: fit the MC means at each grid, and on the per-L inverse-variance pool of both grids."""
    out, pool = {}, defaultdict(list)
    for f in sorted(glob.glob(os.path.join(HERE, "results", "production", "L*_J1_h0_db*_s*.json"))):
        r = json.load(open(f)); pool[(r["dbeta"], r["L"])].append(r["S0"])
    for db in sorted({k[0] for k in pool}):
        Ls = sorted(L for d, L in pool if d == db and L >= 12)
        if len(Ls) < 3:
            continue
        ms = [mean_sem(pool[(db, L)])[:2] for L in Ls]
        out[f"db{db:g}"] = fit_inv_N(Ls, [m for m, _ in ms], [s for _, s in ms])
        out[f"db{db:g}"]["grid_error_inf"] = grid_error_inf(1.0, db)
    Ls = sorted({L for _, L in pool if all((d, L) in pool for d in {k[0] for k in pool})})
    if len(Ls) >= 3 and len({k[0] for k in pool}) > 1:
        y, se = [], []
        for L in Ls:
            ms = [mean_sem(pool[(d, L)])[:2] for d in sorted({k[0] for k in pool})]
            w = np.array([1 / s**2 for _, s in ms]); y.append(sum(wi * m for wi, (m, _) in zip(w, ms)) / w.sum())
            se.append(1 / np.sqrt(w.sum()))
        out["pooled"] = fit_inv_N(Ls, y, se)
    for k, v in out.items():
        print(f"extrapolation {k:8s}: S0_inf = {v['S0_inf']:.6f} ± {v['S0_inf_err_scaled']:.6f} (exact {S0_INF:.7f}, z = {v['z_vs_exact']:+.2f}), "
              f"b = {v['b']:.3f} ± {v['b_err_scaled']:.3f} (exact 0.8676), chi2/dof = {v['chi2']:.2f}/{v['dof']} (errors x {v['err_scale']:.2f})")
    return out

if __name__ == "__main__":
    {"stage1": stage1, "summary": summary}[sys.argv[1] if len(sys.argv) > 1 else "summary"]()
