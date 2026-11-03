"""entropy_meter.py: the residual entropy of the triangular Ising antiferromagnet from a
Metropolis code, by thermodynamic integration of the energy (NumPy only).

    H = J sum_<ij> s_i s_j - h sum_i s_i   on an L x L triangular torus, k_B = 1
    S0/N = ln 2 - int_0^inf [e(beta) - e0] dbeta     (e = <H>/N, e0 = ground-state energy)

Example:  python entropy_meter.py 24 --seed 1 --out results/L24_s1.json
"""
import argparse, json, time
import numpy as np

def neighbour_sum(s):
    """Sum of the six neighbours of every site: (i+-1, j), (i, j+-1), (i+1, j-1), (i-1, j+1)."""
    r = np.roll
    return (r(s, 1, 0) + r(s, -1, 0) + r(s, 1, 1) + r(s, -1, 1)
            + r(r(s, -1, 0), 1, 1) + r(r(s, 1, 0), -1, 1))

def energy(s, J, h):                       # per spin; each bond is counted twice in the sum
    return (0.5 * J * np.sum(s * neighbour_sum(s)) - h * np.sum(s)) / s.size

def sweep(s, beta, J, h, colours, rng):
    """One Metropolis sweep. Neighbours never share a colour, so a whole colour class is updated
    at once. The colour order is drawn at random every sweep, which keeps detailed balance."""
    for c in rng.permutation(3):
        m = colours[c]
        dE = -2.0 * s[m] * (J * neighbour_sum(s)[m] - h)       # cost of flipping each spin
        flip = rng.random(dE.size) < np.exp(-beta * np.maximum(dE, 0.0))
        s[m] = np.where(flip, -s[m], s[m])

def ground_energy(J, h):
    if J < 0:                              # ferromagnet: every bond satisfied
        return 3 * J - abs(h)
    assert abs(h) < 6 * J, "for |h| >= 6J the ground state changes"
    return -J - abs(h) / 3                 # one frustrated bond per triangle; up-up-down if h != 0

def simpson(y, dx):                        # composite Simpson's rule; needs an odd number of points
    assert len(y) % 2 == 1
    return dx / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())

def run(L, seed, nmeas, neq, dbeta, bmax, J, h):
    assert L % 3 == 0, "L must be a multiple of 3 (three-colour update)"
    rng = np.random.default_rng(seed)
    i, j = np.indices((L, L))
    colours = [(i - j) % 3 == c for c in range(3)]
    betas = np.round(np.arange(0.0, bmax + dbeta / 2, dbeta), 10)
    e, m, var = np.zeros_like(betas), np.zeros_like(betas), np.zeros_like(betas)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    # beta = 0 is NOT simulated: there every flip is accepted, so each sweep just inverts the
    # lattice and the "measured" energy stays frozen at that of the random start. e(0) = 0 exactly.
    var[0] = 3 * J**2 + h**2               # exact var(E)/N for independent random spins
    for k in range(1, len(betas)):         # anneal upward in beta, never restart
        for _ in range(neq):
            sweep(s, betas[k], J, h, colours, rng)
        es, ms = np.empty(nmeas), np.empty(nmeas)
        for n in range(nmeas):
            sweep(s, betas[k], J, h, colours, rng)
            es[n], ms[n] = energy(s, J, h), s.mean()
        e[k], m[k], var[k] = es.mean(), ms.mean(), es.var() * s.size
    e0 = ground_energy(J, h)
    S0 = np.log(2) - simpson(e - e0, dbeta)
    S0_trap = np.log(2) - dbeta * ((e - e0).sum() - 0.5 * (e[0] - e0 + e[-1] - e0))
    return dict(L=L, seed=seed, nmeas=nmeas, neq=neq, dbeta=dbeta, bmax=bmax, J=J, h=h, e0=e0,
                beta=betas.tolist(), e=e.tolist(), m=m.tolist(), varE_over_N=var.tolist(),
                S0=S0, S0_trapezoid=S0_trap, tail=e[-1] - e0)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("L", type=int)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--nmeas", type=int, default=5000, help="measured sweeps per beta")
    p.add_argument("--neq", type=int, default=500, help="discarded sweeps per beta")
    p.add_argument("--dbeta", type=float, default=0.1)
    p.add_argument("--bmax", type=float, default=6.0)
    p.add_argument("--J", type=float, default=1.0, help="J > 0 antiferromagnet, J < 0 ferromagnet")
    p.add_argument("--h", type=float, default=0.0, help="field (exercise; needs --bmax 15 at h = 0.3)")
    p.add_argument("--out", help="write the per-seed result as JSON")
    a = p.parse_args()
    t0 = time.time()
    res = run(a.L, a.seed, a.nmeas, a.neq, a.dbeta, a.bmax, a.J, a.h)
    res.update(wall_s=time.time() - t0, numpy=np.__version__, code="entropy_meter.py v0.1")
    print(f"L={a.L} seed={a.seed} J={a.J:g} h={a.h:g}  S0/N = {res['S0']:.5f} (Simpson), "
          f"{res['S0_trapezoid']:.5f} (trapezoid)  e(bmax)-e0 = {res['tail']:.1e}  {res['wall_s']:.0f} s")
    if res["tail"] > 1e-6:
        print(f"WARNING: e - e0 = {res['tail']:.1e} at beta = {a.bmax}; the truncated tail is not negligible")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
