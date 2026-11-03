"""Checks of entropy_meter.py against exact results on small tori.

Run:  python tests/test_entropy_meter.py          (about 3-4 min on one core)
      python tests/test_entropy_meter.py --quick  (skips the full thermodynamic integration)

L = 3 and L = 6 are used here only because they can be solved exactly. Physics results are
quoted for L >= 12 only.
"""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from entropy_meter import neighbour_sum, energy, sweep, ground_energy, simpson, run

def colours_for(L):
    i, j = np.indices((L, L))
    return [(i - j) % 3 == c for c in range(3)]

# ---------- exact references -----------------------------------------------------------------
def exact_e_bruteforce(L, beta, J=1.0, h=0.0):
    """<e> by summing over all 2^N states (L = 3 only)."""
    N = L * L
    E = np.array([energy(np.array(c, np.int8).reshape(L, L), J, h) * N
                  for c in itertools.product((-1, 1), repeat=N)])
    w = np.exp(-beta * (E - E.min()))
    return (w * E).sum() / w.sum() / N

def exact_lnZ_torus(L, beta):
    """ln Z of the L x L AFM torus (J = 1, h = 0) from a row transfer matrix, Z = Tr T^L.
    Independent geometry: row bonds a_j a_{j+1}; inter-row bonds a_j b_j and a_j b_{j-1}.
    Each row pair is a strip of L triangles, so E(a, b) >= -L; shifting by +L keeps T <= 1."""
    st = ((np.arange(1 << L)[:, None] >> np.arange(L)) & 1) * 2 - 1
    Erow = (st * np.roll(st, -1, axis=1)).sum(1)
    Eint = st @ st.T + st @ np.roll(st, 1, axis=1).T
    T = np.exp(-beta * (Erow[:, None] + Eint + L))
    lam = np.linalg.eigvals(T)
    lmax = lam[np.abs(lam).argmax()]
    return L * np.log(lmax.real) + np.log(np.sum((lam / lmax) ** L).real) + beta * L * L

def exact_e_torus(L, beta, d=1e-4):
    return -(exact_lnZ_torus(L, beta + d) - exact_lnZ_torus(L, beta - d)) / (2 * d) / (L * L)

def mc_e(L, beta, J=1.0, h=0.0, seeds=8, neq=2000, nmeas=20000):
    """Equilibrium <e> at one beta; error = SEM over independent seeds."""
    col, means = colours_for(L), []
    for sd in range(seeds):
        rng = np.random.default_rng(1000 + sd)
        s = rng.choice(np.array([-1, 1], np.int8), size=(L, L))
        for _ in range(neq):
            sweep(s, beta, J, h, col, rng)
        acc = 0.0
        for _ in range(nmeas):
            sweep(s, beta, J, h, col, rng)
            acc += energy(s, J, h)
        means.append(acc / nmeas)
    means = np.array(means)
    return means.mean(), means.std(ddof=1) / np.sqrt(seeds)

# ---------- tests ----------------------------------------------------------------------------
def test_geometry():
    for L in (3, 6, 12):
        col = colours_for(L)
        s = np.ones((L, L), np.int8); s[col[2]] = -1                     # up-up-down state
        assert np.isclose(energy(s, 1, 0), -1.0) and np.isclose(energy(s, 1, 0.3), -1.1)
        assert np.isclose(energy(np.ones((L, L), np.int8), 1, 0), 3.0)
        assert np.isclose(energy(np.ones((L, L), np.int8), -1, 0), ground_energy(-1, 0))
        nb = neighbour_sum(np.ones((L, L), np.int8))
        assert np.all(nb == 6)                                           # six neighbours
        for c in range(3):                                               # proper 3-colouring
            x = np.zeros((L, L), np.int8); x[col[c]] = 1
            assert np.all(neighbour_sum(x)[col[c]] == 0)
    # bond list from neighbour_sum must be symmetric: sum_i s_i nb_i = 2 sum_bonds s_i s_j
    rng = np.random.default_rng(0)
    for L in (6, 9):
        s = rng.choice(np.array([-1, 1], np.int8), size=(L, L)).astype(float)
        A = np.zeros((L * L, L * L))
        for k in range(L * L):
            x = np.zeros(L * L); x[k] = 1
            A[:, k] = neighbour_sum(x.reshape(L, L)).ravel()
        assert np.allclose(A, A.T) and np.all(A.sum(0) == 6) and np.all(np.diag(A) == 0)
    print("geometry: ok")

def test_simpson():
    x = np.linspace(0, 6, 61)
    assert np.isclose(simpson(x ** 3, 0.1), 6 ** 4 / 4)                  # exact for cubics
    print("simpson: ok")

def test_exact_lattices():
    # The transfer matrix and the brute-force sum must agree on L = 3 (also checks lattices match)
    for beta in (0.3, 1.0):
        a, b = exact_e_bruteforce(3, beta), exact_e_torus(3, beta)
        assert abs(a - b) < 1e-7, (beta, a, b)
    # Ground-state count at L = 6: ln Z + beta*N at large beta -> ln Omega = ln 263640
    assert abs(exact_lnZ_torus(6, 40.0) - 40.0 * 36 - np.log(263640)) < 1e-6
    print("exact references agree (brute force = transfer matrix; Omega(6) = 263640): ok")

def test_metropolis_vs_exact():
    cases = [(3, 0.5, 0.0), (3, 2.0, 0.0), (3, 1.0, 0.5), (6, 0.3, 0.0), (6, 1.0, 0.0), (6, 2.0, 0.0)]
    for L, beta, h in cases:
        ex = exact_e_bruteforce(L, beta, h=h) if L == 3 else exact_e_torus(L, beta)
        m, err = mc_e(L, beta, h=h, nmeas=20000 if L == 3 else 8000)
        z = (m - ex) / err
        print(f"  L={L} beta={beta} h={h}: MC {m:+.5f} +- {err:.5f}  exact {ex:+.5f}  z={z:+.1f}")
        assert abs(z) < 4
    print("Metropolis e(beta) vs exact: ok")

def test_entropy_L6():
    """Full anneal + Simpson at L = 6 vs the exact count ln(263640)/36 = 0.346732."""
    S = [run(6, sd, nmeas=4000, neq=400, dbeta=0.1, bmax=6.0, J=1.0, h=0.0)["S0"] for sd in range(1, 5)]
    S = np.array(S); m, err = S.mean(), S.std(ddof=1) / 2
    target = np.log(263640) / 36 - 3.4e-5                                # exact count + Simpson bias
    print(f"  L=6: S0 = {m:.5f} +- {err:.5f} (4 seeds), exact count {target + 3.4e-5:.5f}")
    assert abs(m - target) < 4 * err + 2e-4
    print("thermodynamic integration at L = 6: ok")

if __name__ == "__main__":
    test_geometry(); test_simpson(); test_exact_lattices(); test_metropolis_vs_exact()
    if "--quick" not in sys.argv:
        test_entropy_L6()
