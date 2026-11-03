"""Run entropy_meter.py for several lattice sizes and seeds in parallel, one JSON per run.

    python run_seeds.py                      # production set (PRODUCTION below), all cores
    python run_seeds.py --L 24 --seeds 1 2 3 4 --jobs 4 --outdir results/validation

Existing output files are skipped, so an interrupted run can be resumed.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one core per run
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from entropy_meter import run

# Production parameters (Stage 2). nmeas/neq are sweeps per beta point; beta = 0, 0.1, ..., 6.
PRODUCTION = {12: dict(nmeas=20000, neq=1000), 24: dict(nmeas=20000, neq=1000),
              48: dict(nmeas=10000, neq=1000), 96: dict(nmeas=4000, neq=1000)}
SEEDS = list(range(1, 9))

def job(L, seed, kw, path):
    t0 = time.time()
    res = run(L, seed, **kw)
    res.update(wall_s=time.time() - t0, code="entropy_meter.py v0.1")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    return path, res["S0"], res["wall_s"]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, nargs="+", default=sorted(PRODUCTION))
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--nmeas", type=int, help="override the production value")
    p.add_argument("--neq", type=int, help="override the production value")
    p.add_argument("--dbeta", type=float, default=0.1)
    p.add_argument("--bmax", type=float, default=6.0)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p.add_argument("--outdir", default="results/production")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    todo = []
    for L in a.L:
        kw = dict(PRODUCTION.get(L, dict(nmeas=5000, neq=500)), dbeta=a.dbeta, bmax=a.bmax, J=a.J, h=a.h)
        if a.nmeas: kw["nmeas"] = a.nmeas
        if a.neq: kw["neq"] = a.neq
        for sd in a.seeds:
            tag = f"L{L}_J{a.J:g}_h{a.h:g}_db{a.dbeta:g}_s{sd}"
            path = os.path.join(a.outdir, tag + ".json")
            if not os.path.exists(path):
                todo.append((L, sd, kw, path))
    print(f"{len(todo)} runs to do on {a.jobs} workers -> {a.outdir}", flush=True)
    # slowest runs first, so they do not end up alone at the end. Measured cost per sweep is
    # ~340 us of NumPy call overhead plus ~0.065 us per site.
    todo.sort(key=lambda t: -(t[2]["nmeas"] + t[2]["neq"]) * (340 + 0.065 * t[0] ** 2))
    with ProcessPoolExecutor(a.jobs) as ex:
        for fut in as_completed([ex.submit(job, *t) for t in todo]):
            path, S0, wall = fut.result()
            print(f"{os.path.basename(path):40s} S0/N = {S0:.5f}  {wall:.0f} s", flush=True)
