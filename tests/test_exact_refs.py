"""Checks of the exact finite-lattice ground-state counts in exact_refs.py.

Run:  python tests/test_exact_refs.py          (seconds: L = 3, 6, 9)
      python tests/test_exact_refs.py --full   (adds L = 12, a few minutes)

The Kasteleyn count lnOmega() (honeycomb-dimer mapping, even L only) is compared with an
independent row transfer-matrix count tm_lnOmega() (any L <= 12). L = 3 is also checked against
brute-force enumeration of all 2^9 states with the meter's own energy function.
"""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exact_refs import lnOmega, tm_lnOmega
from entropy_meter import energy

def brute_force_omega(L):
    E = [energy(np.array(c, np.int8).reshape(L, L), 1.0, 0.0) for c in itertools.product((-1, 1), repeat=L * L)]
    return int(np.isclose(E, -1.0).sum())

if __name__ == "__main__":
    assert brute_force_omega(3) == 42
    assert np.isclose(tm_lnOmega(3), np.log(42))
    print("L=3: Omega = 42 (brute force = transfer matrix)")
    assert np.isclose(tm_lnOmega(6), lnOmega(6, 6), rtol=0, atol=1e-9)
    assert round(np.exp(lnOmega(6, 6))) == 263640
    print("L=6: Omega = 263640 (transfer matrix = Kasteleyn)")
    s9 = tm_lnOmega(9) / 81
    assert abs(s9 - 0.333676) < 1e-6
    print(f"L=9: S0/N = {s9:.6f} (transfer matrix; odd L, no Kasteleyn)")
    if "--full" in sys.argv:
        assert np.isclose(tm_lnOmega(12), lnOmega(12, 12), rtol=0, atol=1e-8)
        print("L=12: transfer matrix = Kasteleyn")
    print("all exact-count checks passed")
