"""quench_demo.py: what goes wrong without the anneal ("What can go wrong" in the paper).

A random start dropped straight to low temperature (beta = 6) instead of being annealed up from
beta = 0. The energy after n sweeps (log-spaced) is compared with the ground-state energy e0 and with
the annealed value at the same beta (production runs for h = 0, results/field for h != 0).

    python quench_demo.py                      # L = 48, h = 0, 0.3, 1; seeds 1-8; 1e5 sweeps -> results/quench/
    python quench_demo.py --report             # table of e(n) against e0 and the annealed value
"""
import argparse, glob, json, os, time
from multiprocessing import Pool
import numpy as np
import entropy_meter as em

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "quench")

def quench(L, seed, beta, h, nsweeps, J=1.0):
    rng = np.random.default_rng(seed)
    i, j = np.indices((L, L))
    colours = [(i - j) % 3 == c for c in range(3)]
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    marks = set(np.unique(np.logspace(0, np.log10(nsweeps), 41).astype(int)).tolist())
    n_out, e_out, m_out = [0], [em.energy(s, J, h)], [float(s.mean())]
    for n in range(1, nsweeps + 1):
        em.sweep(s, beta, J, h, colours, rng)
        if n in marks:
            n_out.append(n); e_out.append(em.energy(s, J, h)); m_out.append(float(s.mean()))
    return dict(L=L, seed=seed, beta=beta, h=h, J=J, e0=em.ground_energy(J, h), nsweeps=nsweeps,
                n=n_out, e=e_out, m=m_out, code="quench_demo.py + entropy_meter.py v0.1")

def job(args):
    L, seed, beta, h, nsweeps = args
    path = os.path.join(OUT, f"L{L}_h{h:g}_b{beta:g}_s{seed}.json")
    if os.path.exists(path):
        return path, None
    t = time.time(); r = quench(L, seed, beta, h, nsweeps); r["wall_s"] = time.time() - t
    json.dump(r, open(path, "w"), indent=1)
    return path, r

def annealed(L, beta, h):
    sub, pat = ("production", f"L{L}_J1_h0_db0.05_s*.json") if h == 0 else ("field", f"L{L}_J1_h{h:g}_db0.05_s*.json")
    rs = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "results", sub, pat))]
    if not rs:
        return None
    k = int(np.argmin(abs(np.array(rs[0]["beta"]) - beta)))
    e = np.array([r["e"][k] for r in rs])
    return e.mean(), e.std(ddof=1) / np.sqrt(len(e))

def report(L, beta):
    for h in sorted({json.load(open(f))["h"] for f in glob.glob(os.path.join(OUT, f"L{L}_h*_b{beta:g}_s*.json"))}):
        rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, f"L{L}_h{h:g}_b{beta:g}_s*.json")))]
        E = np.array([r["e"] for r in rs]); n = rs[0]["n"]; e0 = rs[0]["e0"]; a = annealed(L, beta, h)
        print(f"\nL = {L}, h = {h:g}, beta = {beta:g}: e0 = {e0:.4f}; annealed e = "
              + (f"{a[0]:.5f} ± {a[1]:.1e}" if a else "(no annealed runs yet)") + f"; {len(rs)} seeds")
        for k in [i for i, x in enumerate(n) if x in (10, 100, 1000, 10000, 100000) or x == n[-1]]:
            print(f"  after {n[k]:>6d} sweeps: e = {E[:, k].mean():.4f} ± {E[:, k].std(ddof=1) / np.sqrt(len(rs)):.4f}"
                  f"  (min {E[:, k].min():.4f}, max {E[:, k].max():.4f}); e - e0 = {E[:, k].mean() - e0:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=int, default=48)
    p.add_argument("--beta", type=float, default=6.0)
    p.add_argument("--h", type=float, nargs="+", default=[0.0, 0.3, 1.0])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 9)))
    p.add_argument("--nsweeps", type=int, default=100000)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if a.report:
        report(a.L, a.beta)
    else:
        os.makedirs(OUT, exist_ok=True)
        todo = [(a.L, s, a.beta, h, a.nsweeps) for h in a.h for s in a.seeds]
        with Pool(a.jobs) as pool:
            for path, r in pool.imap_unordered(job, todo):
                print(f"{os.path.basename(path):30s} " + ("exists" if r is None else
                      f"e(final) = {r['e'][-1]:.4f} (e0 = {r['e0']:.4f})  {r['wall_s']:.0f} s"), flush=True)
