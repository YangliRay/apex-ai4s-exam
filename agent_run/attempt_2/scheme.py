import numpy as np

N_QUBITS = 8

_SQ2 = np.sqrt(2.0)
_SQ3 = np.sqrt(3.0)

# Two-qubit states, index = 2*q_first + q_second (first qubit is the more significant bit)
_S = np.array([0.0, 1.0, -1.0, 0.0]) / _SQ2   # singlet
_TP = np.array([1.0, 0.0, 0.0, 0.0])          # triplet m=+1
_T0 = np.array([0.0, 1.0, 1.0, 0.0]) / _SQ2   # triplet m=0
_TM = np.array([0.0, 0.0, 0.0, 1.0])          # triplet m=-1


def _block_singlets():
    """The two total-spin-0 states of 4 qubits (qubit 0 is the most significant bit):
    ss = singlet(0,1) ⊗ singlet(2,3)
    tt = [triplet(0,1) ⊗ triplet(2,3)] coupled to total spin 0
    """
    ss = np.kron(_S, _S)
    tt = (np.kron(_TP, _TM) - np.kron(_T0, _T0) + np.kron(_TM, _TP)) / _SQ3
    return ss, tt


def code_basis():
    """Return a 16 x 2 complex array: (|0_L>, |1_L>) on one block (4 qubits, qubit 0 is the most significant bit).
    Both blocks use the same code.
    Logical X = |ss><ss| - |tt><tt|, i.e. on the code space X_L = -SWAP_01 = -SWAP_23.
    """
    ss, tt = _block_singlets()
    zero_L = (ss + tt) / _SQ2
    one_L = (ss - tt) / _SQ2
    return np.stack([zero_L, one_L], axis=1).astype(complex)


def build(K, h, Delta):
    """h = (h1, h2). Return a list of terms, each
       {"type": "heisenberg", "sites": (a, b), "J": float}

    H_sim = Delta * sum_{a<b in A} H_ab + Delta * sum_{a<b in B} H_ab        (block code / energy penalty)
          + g * (sigma_0 - sigma_1) . (sigma_4 + sigma_5),  g = sqrt(3 K Delta / 2)   (2nd order -> K X1 X2)
          - (h1 + K)/2 * H_23 + (K - h2/2) * H_67                              (exact field terms + compensation)
    Requires K >= 0 (the task has K = pi/4).
    """
    K = float(K)
    Delta = float(Delta)
    h1, h2 = float(h[0]), float(h[1])
    if K < 0:
        raise ValueError("This scheme is written for K >= 0.")
    g = np.sqrt(1.5 * K * Delta)

    J = {}

    def add(a, b, val):
        key = (min(a, b), max(a, b))
        J[key] = J.get(key, 0.0) + val

    # 1) Intra-block all-to-all coupling Delta: the singlet space is the block ground space, gap 4*Delta
    for base in (0, 4):
        for i in range(4):
            for j in range(i + 1, 4):
                add(base + i, base + j, Delta)

    # 2) Logical fields (exact). On the code space H_23 = -1 - 2 X_A and H_67 = -1 - 2 X_B
    add(2, 3, -(h1 + K) / 2.0)   # gives (h1 + K) X_A, cancelling the 2nd-order -K X_A
    add(6, 7, K - h2 / 2.0)      # gives (h2 - 2K) X_B, cancelling the 2nd-order +2K X_B

    # 3) Inter-block coupling g (sigma_0 - sigma_1).(sigma_4 + sigma_5)
    add(0, 4, g)
    add(0, 5, g)
    add(1, 4, -g)
    add(1, 5, -g)

    return [{"type": "heisenberg", "sites": (a, b), "J": float(v)}
            for (a, b), v in sorted(J.items())]