"""Submission 03 (identical Hamiltonian to 01): second-order perturbative gadget, sparse cross-block pattern.

Each logical qubit = 4 physical qubits whose two-dimensional singlet sector is the
code space (Cubitt-Montanaro-Piddock encoding).  Block A = qubits 0-3, block B = 4-7.

  H_sim = Delta * sum_blocks sum_{mu<nu} H_{mu nu}            (penalty, 2 S^2 - 6)
        + sqrt(Delta K) * (H_{1,1'} - 2 H_{2,2'} + H_{3,3'})   (cross-block coupler)
        + sum_blocks [ h/sqrt3 H_{14} + (h/sqrt3 - K/2)/2 H_{13} ]   (first-order terms)

Second order (corrected coefficients):  K [ X_L X_L' - (Z_L + Z_L')/2 ] + const.
The first-order term supplies h X_L + (K/2) Z_L per block, cancelling the Z_L part.
"""
import numpy as np

N_QUBITS = 8
BLOCKS = [(0, 1, 2, 3), (4, 5, 6, 7)]
PATTERN = {0: 1.0, 1: -2.0, 2: 1.0}          # cross-block (mu, mu') weights


def _pair_singlets(pairs):
    """Product of two singlets (|01>-|10>)/sqrt2 on the given qubit pairs of a block."""
    v = np.zeros(16, dtype=complex)
    (a, b), (c, d) = pairs
    for x in (0, 1):
        for y in (0, 1):
            bits = [0] * 4
            bits[a], bits[b], bits[c], bits[d] = x, 1 - x, y, 1 - y
            v[int("".join(map(str, bits)), 2)] += (-1) ** x * (-1) ** y * 0.5
    return v


def code_basis():
    zero = _pair_singlets([(0, 2), (1, 3)])                     # |Psi->_13 |Psi->_24
    one = (2 / np.sqrt(3)) * _pair_singlets([(0, 1), (2, 3)]) - zero / np.sqrt(3)
    return np.stack([zero, one], axis=1)


def build(K, h, Delta):
    terms = []
    for blk, hk in zip(BLOCKS, h):
        for i in range(4):
            for j in range(i + 1, 4):
                terms.append({"type": "heisenberg", "sites": (blk[i], blk[j]), "J": Delta})
        alpha, beta = hk, K / 2
        terms.append({"type": "heisenberg", "sites": (blk[0], blk[3]), "J": alpha / np.sqrt(3)})
        terms.append({"type": "heisenberg", "sites": (blk[0], blk[2]),
                      "J": 0.5 * (alpha / np.sqrt(3) - beta)})
    g = np.sqrt(Delta * K)
    for mu, w in PATTERN.items():
        terms.append({"type": "heisenberg", "sites": (BLOCKS[0][mu], BLOCKS[1][mu]), "J": g * w})
    return terms
