"""
Heisenberg-only simulation of H_target = K X1 X2 + h1 X1 + h2 X2 on 8 physical qubits.

Encoding (each block of 4 spins): the 2-dim total-spin-0 (singlet) subspace.
  |-_L> = |s_01> (x) |s_23>                     (antisymmetric under SWAP_01)
  |+_L> = [ |t_01> (x) |t_23> ]_{S=0}            (symmetric under SWAP_01)
  |0_L> = (|+_L> + |-_L>)/sqrt2,  |1_L> = (|+_L> - |-_L>)/sqrt2
so that SWAP_01 (= SWAP_23) restricted to the code space equals logical X.

Couplings (H_ab = X_aX_b + Y_aY_b + Z_aZ_b = 2 SWAP_ab - 1):
  * penalty:  Delta * H_ab for all 6 pairs inside block A {0,1,2,3} and inside block B {4,5,6,7}
  * logical fields: J_01 = (h1 + K/2)/2 ,  J_45 = (h2 + K/2)/2      (added to the penalty bonds)
  * inter-block gadget: J_04 = +sqrt(3 K Delta) ,  J_15 = -sqrt(3 K Delta)
Second-order perturbation theory gives  H_eff = K X1X2 + h1 X1 + h2 X2 + const + O(1/Delta).
(K > 0 assumed, as in the task where K = pi/4.)
"""
import numpy as np

N_QUBITS = 8


def _two_spin_states():
    up = np.array([1.0, 0.0])
    dn = np.array([0.0, 1.0])
    s = (np.kron(up, dn) - np.kron(dn, up)) / np.sqrt(2)          # singlet
    tp = np.kron(up, up)                                          # |1, +1>
    t0 = (np.kron(up, dn) + np.kron(dn, up)) / np.sqrt(2)         # |1,  0>
    tm = np.kron(dn, dn)                                          # |1, -1>
    return s, tp, t0, tm


def code_basis():
    """16 x 2 complex array: (|0_L>, |1_L>) on one block (qubit 0 = most significant bit)."""
    s, tp, t0, tm = _two_spin_states()
    minus = np.kron(s, s)                                         # qubits (0,1) then (2,3)
    # total-spin-0 combination of two spin-1's (Clebsch-Gordan)
    plus = (np.kron(tp, tm) - np.kron(t0, t0) + np.kron(tm, tp)) / np.sqrt(3)
    zero = (plus + minus) / np.sqrt(2)
    one = (plus - minus) / np.sqrt(2)
    return np.stack([zero, one], axis=1).astype(complex)


def build(K, h, Delta):
    """h = (h1, h2). Returns list of {"type": "heisenberg", "sites": (a, b), "J": float}."""
    h1, h2 = float(h[0]), float(h[1])
    K = float(K)
    Delta = float(Delta)
    if K < 0:
        raise ValueError("this construction assumes K >= 0 (task: K = pi/4)")
    J = {}

    def add(a, b, val):
        key = (min(a, b), max(a, b))
        J[key] = J.get(key, 0.0) + float(val)

    # 1) singlet penalty inside each block
    for block in ((0, 1, 2, 3), (4, 5, 6, 7)):
        for i in range(4):
            for j in range(i + 1, 4):
                add(block[i], block[j], Delta)
    # 2) logical X fields (incl. compensation of the -K/2 X shift produced by the gadget)
    add(0, 1, (h1 + K / 2) / 2)
    add(4, 5, (h2 + K / 2) / 2)
    # 3) second-order inter-block gadget
    lam = np.sqrt(3.0 * K * Delta)
    add(0, 4, +lam)
    add(1, 5, -lam)

    return [{"type": "heisenberg", "sites": k, "J": v} for k, v in sorted(J.items())]
