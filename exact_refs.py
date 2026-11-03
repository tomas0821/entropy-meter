"""exact_refs.py: exact reference values for checking entropy_meter.py (NumPy + SciPy).

  * S0(L) = ln(Omega_L)/N for the L x L AFM torus: Kasteleyn dimer count (any even L), and a row
    transfer-matrix count for small L of either parity (L <= 12; used for L = 3, 9 and as a check).
  * e(beta), s(beta) for the infinite lattice, AFM and FM: Houtappel's free energy reduced to a
    1-D integral (never Wannier 1950, whose e(T) has a sign error).
  * e(beta) for the finite L x L torus (L <= 12) from a row transfer matrix, AFM or FM: an exact
    point-by-point check of the Monte Carlo energies at a production size.
  * The grid error of Simpson's rule and of the trapezoid rule on the TI integral.

    python exact_refs.py            # writes results/exact/exact_refs.json (takes ~1 h: the L=12 TM)
"""
import json, os, sys, time
import numpy as np
from scipy.integrate import quad

S0_INF = 0.3230659472            # Wannier 1950 / Houtappel: (2/pi) int_0^{pi/3} ln(2 cos w) dw

# ------------------------------------------------------------------ exact finite-L ground-state counts
def lnOmega(L1, L2):
    """ln of the number of AFM ground states on an L1 x L2 torus (both even), honeycomb-dimer mapping:
    Omega = 2 N00, N00 = (1/4) sum_{theta,phi in {0,pi}} +-|det K_{theta,phi}|, det = prod (1 + e^{-ik} + e^{-iq}).
    Checked against the independent transfer-matrix count tm_lnOmega() in tests/test_exact_refs.py."""
    terms = []
    for th in (0.0, np.pi):
        for ph in (0.0, np.pi):
            k = (2 * np.pi * np.arange(L1) + th) / L1
            q = (2 * np.pi * np.arange(L2) + ph) / L2
            f = 1 + np.exp(-1j * k)[:, None] + np.exp(-1j * q)[None, :]
            a = np.abs(f).ravel()
            if np.any(a < 1e-12):
                continue
            terms.append((float(np.sign(np.cos(np.angle(f).sum()))), float(np.log(a).sum())))
    lmax = max(l for _, l in terms)
    return lmax + np.log(2 * sum(s * np.exp(l - lmax) for s, l in terms) / 4)

def tm_lnOmega(L):
    """ln of the number of AFM ground states on the L x L torus (any L <= 12), by a row transfer matrix
    restricted to zero-energy row pairs: Omega = Tr T0^L, with T0_ab = 1 if rows a, b add no unhappy bond
    beyond one per triangle. Same geometry as entropy_meter.py. Omega(3) = 42."""
    st = ((np.arange(1 << L)[:, None] >> np.arange(L)) & 1) * 2 - 1
    Erow = (st * np.roll(st, -1, axis=1)).sum(1)
    E = Erow[:, None] + st @ st.T + st @ np.roll(st, 1, axis=1).T
    T0 = (E == -L).astype(np.float64)              # each row pair is a strip of L triangles: E >= -L
    lam = np.linalg.eigvals(T0)
    lmax = np.abs(lam).max()
    return L * np.log(lmax) + np.log(np.sum((lam / lmax) ** L).real)

# ------------------------------------------------------------------ infinite lattice, Houtappel
def _afm_ab(w, K):               # overflow-safe AFM form (divided by e^{2K}) of Houtappel's integrand
    u = np.exp(-4 * K)
    at = 0.75 + 0.25 * u * u + 0.5 * (1 - u) * np.cos(w)
    bt = (1 - u) * np.cos(w / 2)
    dat = -4 * u * (0.5 * u - 0.5 * np.cos(w))
    dbt = 4 * u * np.cos(w / 2)
    return at, bt, dat, dbt

def afm_lnZ(K):
    def f(w):
        at, bt, _, _ = _afm_ab(w, K)
        return np.log((at + np.sqrt(max(at * at - bt * bt, 0.0))) / 2)
    return np.log(2) + K + quad(f, 0, np.pi, points=[2 * np.pi / 3], limit=400, epsabs=1e-14, epsrel=1e-13)[0] / (2 * np.pi)

def afm_e(K):
    """Energy per spin of the infinite AFM (J = 1) at beta = K; analytic K-derivative of the 1-D integral."""
    if K == 0:
        return 0.0
    def f(w):
        at, bt, dat, dbt = _afm_ab(w, K)
        sD = np.sqrt(at * at - bt * bt)
        return (dat + (at * dat - bt * dbt) / sD) / (at + sD)
    return -(1 + quad(f, 0, np.pi, points=[2 * np.pi / 3], limit=400, epsabs=1e-14, epsrel=1e-13)[0] / (2 * np.pi))

def afm_s(K):
    return np.log(2) if K == 0 else afm_lnZ(K) + K * afm_e(K)

def fm_lnZ(K):
    """Infinite triangular FERROmagnet (J = -1 in H = J sum s s): ln Z/N = ln2 + (1/2pi) int_0^pi ln[(a+sqrt(a^2-b^2))/2],
    a = C^3 + S^3 - S cos w, b = 2 S cos(w/2), C = cosh 2K, S = sinh 2K. Critical at a = b, w = 0: K_c = ln(3)/4."""
    C, S = np.cosh(2 * K), np.sinh(2 * K)
    def f(w):
        a, b = C**3 + S**3 - S * np.cos(w), 2 * S * np.cos(w / 2)
        return np.log((a + np.sqrt(max(a * a - b * b, 0.0))) / 2)
    return np.log(2) + quad(f, 0, np.pi, points=[0.0], limit=400, epsabs=1e-14, epsrel=1e-13)[0] / (2 * np.pi)

def fm_e(K, d=1e-5):
    if K == 0:
        return 0.0
    return -(fm_lnZ(K + d) - fm_lnZ(K - d)) / (2 * d)

# ------------------------------------------------------------------ finite torus, row transfer matrix
def tm_energy(L, betas, J=1.0, threads_note=""):
    """Exact <E>/N on the L x L torus (same lattice as entropy_meter.py; geometry of tests/), h = 0.
    Z = Tr T^L with T_ab = exp(-beta W_ab), W = J(E_row + E_inter) - E_min >= 0 (E_min = -L for the AFM,
    -3L for the FM). d ln Z / d beta = -L Tr(T^{L-1} (W o T)) / Tr(T^L), so no finite differences."""
    st = ((np.arange(1 << L)[:, None] >> np.arange(L)) & 1) * 2 - 1
    Erow = (st * np.roll(st, -1, axis=1)).sum(1)
    E = (Erow[:, None] + st @ st.T + st @ np.roll(st, 1, axis=1).T).astype(np.float64)
    Emin = -L if J > 0 else -3 * L
    W = J * E - Emin
    assert W.min() >= -1e-9
    del E, st
    out = []
    for b in betas:
        t = time.time()
        T = np.exp(-b * W)
        P, Q, n = None, T, L - 1                 # P = T^(L-1) by binary powering
        while n:
            if n & 1:
                P = Q.copy() if P is None else P @ Q
            n >>= 1
            if n:
                Q = Q @ Q
        del Q
        Z = np.sum(P * T.T)
        dZ = np.sum(P * (W * T).T)
        out.append(L * dZ / Z / (L * L) + Emin / L)      # <E>/N = (L <W_row> + L Emin)/L^2
        print(f"  TM L={L} J={J:g} beta={b:.2f}: e={out[-1]:.12f}  ({time.time() - t:.0f} s)", flush=True)
    return np.array(out)

# ------------------------------------------------------------------ quadrature
def simpson(y, dx):
    assert len(y) % 2 == 1
    return dx / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())

def trapezoid(y, dx):
    return dx * (y.sum() - 0.5 * (y[0] + y[-1]))

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "results", "exact"), exist_ok=True)
    R = {"S0_inf": S0_INF}
    # finite-L counts
    R["S0_L"] = {L: lnOmega(L, L) / L**2 for L in (6, 12, 18, 24, 30, 36, 48, 60, 72, 96, 192, 384)}
    for L, v in R["S0_L"].items():
        print(f"S0({L}) = {v:.8f}   (S0 - S0_inf) L^2 = {(v - S0_INF) * L * L:.4f}")
    R["S0_L_tm"] = {L: tm_lnOmega(L) / L**2 for L in (3, 6, 9, 12)}      # odd L too (the paper quotes L = 9)
    print("transfer matrix:", {L: round(v, 6) for L, v in R["S0_L_tm"].items()})
    # infinite-lattice curves on the fine grid (contains the 0.1 and 0.05 grids), to beta = 15 for the field runs' axis
    bf = np.round(np.arange(0, 15.0 + 1e-9, 0.025), 10)
    R["beta_fine"] = bf.tolist()
    R["afm_e_inf"] = [afm_e(b) for b in bf]
    R["afm_s_inf"] = [afm_s(b) for b in bf]
    R["fm_e_inf"] = [fm_e(b) for b in bf]
    # grid errors of the TI integral, infinite lattice, truncated at beta = 6
    ea, ef = np.array(R["afm_e_inf"]), np.array(R["fm_e_inf"])
    grid = {}
    for db in (0.2, 0.1, 0.05, 0.025):
        k = int(round(db / 0.025)); n = int(round(6 / 0.025))
        ya, yf = ea[:n + 1:k] + 1, ef[:n + 1:k] + 3
        grid[db] = dict(afm_simpson=np.log(2) - simpson(ya, db) - S0_INF, afm_trapezoid=np.log(2) - trapezoid(ya, db) - S0_INF,
                        fm_simpson=np.log(2) - simpson(yf, db), fm_trapezoid=np.log(2) - trapezoid(yf, db))
        print(f"grid dbeta={db}: " + "  ".join(f"{k}={v:+.2e}" for k, v in grid[db].items()))
    R["grid_error_inf"] = grid
    R["tail_afm_beta6"] = ea[240] + 1
    json.dump(R, open(os.path.join(here, "results", "exact", "exact_refs.json"), "w"), indent=1)
    # finite torus L = 12, AFM and FM, on the 0.05 grid to beta = 6 (the slow part)
    if "--no-tm" not in sys.argv:
        b05 = np.round(np.arange(0, 6.0 + 1e-9, 0.05), 10)
        for J, key in ((1.0, "afm"), (-1.0, "fm")):
            e12 = tm_energy(12, b05[1:], J)
            R[f"tm12_{key}_beta"] = b05.tolist()
            R[f"tm12_{key}_e"] = [0.0] + e12.tolist()
            e0 = -1.0 if J > 0 else -3.0
            y = np.array(R[f"tm12_{key}_e"]) - e0
            exact = R["S0_L"][12] if J > 0 else np.log(2) / 144
            R[f"tm12_{key}_grid"] = {db: dict(simpson=np.log(2) - simpson(y[::k], db) - exact,
                                              trapezoid=np.log(2) - trapezoid(y[::k], db) - exact)
                                     for db, k in ((0.1, 2), (0.05, 1))}
            print(key, "L=12 grid errors:", R[f"tm12_{key}_grid"], flush=True)
            json.dump(R, open(os.path.join(here, "results", "exact", "exact_refs.json"), "w"), indent=1)
