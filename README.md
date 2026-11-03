# entropy_meter: residual entropy of the triangular Ising antiferromagnet from a Metropolis code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23901447.svg)](https://doi.org/10.5281/zenodo.23901447)

Cite as: T. Rojas Solorzano, *Entropy meter: code and data for "An entropy that does not vanish"*, version 1.0.0, Zenodo (2026), https://doi.org/10.5281/zenodo.23901447 (all versions: https://doi.org/10.5281/zenodo.23901446).

Companion code to *"An entropy that does not vanish"* (T. Rojas Solorzano, in preparation for *The Physics
Educator*). It turns a plain Metropolis Ising code into an entropy meter.

Entropy is not an ensemble average, so we integrate the measured energy per spin e(β) from
infinite temperature, where S/N = ln 2 exactly:

    S0/N = ln 2 − ∫₀^∞ [e(β) − e0] dβ        (k_B = 1)

For the nearest-neighbour triangular Ising antiferromagnet, e0 = −J and the answer is Wannier's
residual entropy, 0.3230659… per spin (the value printed in 1950, 0.3383, was corrected in
1973). The method is that of M. Žukovič, Eur. Phys. J. B 86, 283 (2013); this code is a
classroom version of it.

## Files
| File | What |
|---|---|
| `entropy_meter.py` | The whole simulation: NumPy only. Run it, or import `run()`. |
| `run_seeds.py` | Runs several L and seeds in parallel; one JSON file per run. |
| `exact_refs.py` | Exact references: finite-L ground-state counts (Kasteleyn for even L, transfer matrix for L ≤ 12), infinite-lattice e(β) and S(β), grid errors. Needs SciPy. |
| `analyze.py`, `make_figures.py` | Table 1, the fits and the three figures of the paper. `make_figures.py` needs Matplotlib. |
| `quench_demo.py` | The "skip the anneal" demonstration in a field. |
| `tests/test_entropy_meter.py` | Checks the meter against exact results on 3×3 and 6×6 tori. |
| `tests/test_exact_refs.py` | Checks the Kasteleyn count against the transfer-matrix count. |
| `results/` | Per-seed JSON output (`production/`, `class/`, `field/`, `quench/`), exact references and run logs. |
| `figures/` | The paper's figures and Table 1 as made by `make_figures.py` (set `FIGDIR` to write elsewhere). |
| `CITATION.cff` | How to cite this code. |

## Requirements
The meter itself (`entropy_meter.py`) needs only Python ≥ 3.9 and NumPy (tested with Python 3.12,
NumPy 2.5). `exact_refs.py` also needs SciPy and `make_figures.py` Matplotlib:
`pip install -r requirements.txt`.

## Quick start
```bash
python entropy_meter.py 12 --neq 200 --nmeas 2000           # class settings (paper Sec. 6): under a minute here, 1-2 min with the paper's Appendix code
python entropy_meter.py 24                                   # one run, L = 24, about 2 min on one core
python entropy_meter.py 24 --seed 2 --out results/L24_s2.json
python entropy_meter.py 12 --J -1 --dbeta 0.0125 --neq 500   # ferromagnet control: S0/N -> ln(2)/N; needs the fine grid
python entropy_meter.py 48 --h 1.0 --dbeta 0.05 --bmax 15    # exercise: a field removes S0
python tests/test_entropy_meter.py                           # ~5 min
python tests/test_exact_refs.py                              # seconds (add --full for L = 12)
```

## Model and conventions
- H = J Σ⟨ij⟩ s_i s_j − h Σ_i s_i on an L × L triangular torus, with k_B = 1.
  - J > 0 is the antiferromagnet; J < 0 is the ferromagnet.
  - Site (i, j) has six neighbours: (i ± 1, j), (i, j ± 1), (i + 1, j − 1), (i − 1, j + 1).
- Ground-state energy per spin:
  - antiferromagnet: e0 = −J − |h|/3 for |h| < 6J (up-up-down when h ≠ 0);
  - ferromagnet: e0 = 3J − |h|.
- Single-spin-flip Metropolis on three sublattices ("colours", (i − j) mod 3). No two neighbours
  share a colour, so a whole colour can be updated at once. The colour order is drawn at random
  every sweep, which gives detailed balance; a fixed order also samples the right distribution
  (balance), but without detailed balance.

## Things that go wrong (and what the code does about them)
1. **β = 0 cannot be simulated with sweeps.** Every flip is accepted, so each sweep inverts the
   whole lattice, and the "measured" energy is frozen at that of the random start (±√3/L).
   That is not an equilibrium average. The code sets e(0) = 0, which is exact.
2. **The integration rule matters.** On a grid with dβ = 0.1:
   - Simpson's rule is off by −3.4×10⁻⁵;
   - the trapezoid rule is off by −2.5×10⁻³, which is larger than the finite-size effect at
     L = 24.

   The code reports both.
3. **Truncation.** At β = 6, e − e0 ≈ 10⁻⁹ for h = 0. In a field the gap is only 2h, so use
   `--bmax 15`. The code warns if e(β_max) − e0 > 10⁻⁶.
4. **Lattice size.**
   - L must be a multiple of 3 for the three-colour update; use multiples of 6 for results.
   - The finite-size correction is S0(L) − S0(∞) ≈ 0.8676/L² for L a multiple of 6.
   - Very small lattices mislead: L = 6 gives 0.3467 and L = 9 gives 0.3337 (transfer matrix), both
     close to the misprinted 0.3383.
   - Multiples of 6, not just 3, because `exact_refs.lnOmega` (Kasteleyn) needs even L.
   - The exact counts use the honeycomb-dimer mapping: a dimer across the unhappy bond of every
     triangle turns each ground state into a dimer covering of the honeycomb lattice (two spin
     states per covering; on the torus only some winding sectors occur), which Kasteleyn's method counts.
5. **Error bars.** Use the standard error of the mean over ≥ 8 independent seeds. Successive
   sweeps are correlated, so the naive standard error of one long run is too small.
6. **Anneal, never restart.** Each β starts from the last configuration of the previous β. In a
   field this is essential: a random start put straight at β = 6 and h = J has only reached
   e ≈ −1.17 after 10⁵ sweeps (and is still drifting slowly), far above e0 = −1.333
   (`quench_demo.py`; the energy is −1.08 at 10⁴ sweeps).

## Production runs (Stage 2 of the paper)
```bash
python run_seeds.py            # L = 12, 24, 48, 96 × seeds 1–8, β = 0…6 in steps of 0.1
```

| L | measured sweeps per β | discarded sweeps per β | seeds |
|---|---|---|---|
| 12 | 20000 | 1000 | 1–8 |
| 24 | 20000 | 1000 | 1–8 |
| 48 | 10000 | 1000 | 1–8 |
| 96 | 4000 | 1000 | 1–8 |

Each seed is passed to `numpy.random.default_rng(seed)`. Output goes to
`results/production/L{L}_J{J}_h{h}_db{dβ}_s{seed}.json`. On one free core a measured sweep takes
about 0.30 ms at L = 12 and 0.81 ms at L = 96, so one run takes 4–8 min and the 32 runs at
dβ = 0.1 about 3 h serial. (The wall times in `results/logs/` are longer because 10 jobs shared
6 physical cores.)

The other production sets (all from `results/logs/queue_overnight.sh`):
```bash
python run_seeds.py --dbeta 0.05                                         # AFM, Table 1 second column
python run_seeds.py --J -1 --dbeta 0.0125 --nmeas 2000 --neq 500         # FM control
python run_seeds.py --L 48 --h 0.3 --dbeta 0.05 --bmax 15 --nmeas 5000 --neq 1000 --outdir results/field
python run_seeds.py --L 48 --h 1.0 --dbeta 0.05 --bmax 15 --nmeas 5000 --neq 1000 --outdir results/field
python run_seeds.py --L 12 24 --neq 200 --nmeas 2000 --outdir results/class                 # class settings
python run_seeds.py --L 12 --J -1 --dbeta 0.0125 --neq 200 --nmeas 2000 --outdir results/class          # too few discarded sweeps:
python run_seeds.py --L 12 --J -1 --dbeta 0.0125 --neq 500 --nmeas 2000 --outdir results/class_fm_neq500 # use this one
```

Class-setting results (8 seeds, mean ± SEM; exact in brackets): antiferromagnet L = 12: 0.32889 ± 0.00027
(0.32906), L = 24: 0.32429 ± 0.00022 (0.32457); ferromagnet L = 12 with 200 discarded sweeps:
0.00533 ± 0.00022 (0.00481, +2.4σ, not equilibrated near T_c), with 500: 0.00454 ± 0.00032 (−0.8σ).

## Licence
MIT (see `LICENSE`).
