"""Small exact-diagonalisation toolkit for Heisenberg-only simulator schemes.

Conventions
-----------
* Qubits are labelled 0..N-1; qubit 0 is the most significant bit of a basis index.
* The Heisenberg (exchange) term on a bond (a, b) is
      H_ab = X_a X_b + Y_a Y_b + Z_a Z_b = 2 SWAP_ab - 1.
* A scheme is described by a list of bonds (a, b, J); the Hamiltonian is
      H = sum_bonds J * H_ab.
  Constant shifts are irrelevant for dynamics and are never part of a scheme.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh as _sp_eigh

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def single(op: np.ndarray, site: int, n: int) -> np.ndarray:
    """Embed a single-qubit operator on `site` of an n-qubit register."""
    out = np.array([[1.0 + 0j]])
    for k in range(n):
        out = np.kron(out, op if k == site else I2)
    return out


def heisenberg_bond(a: int, b: int, n: int) -> np.ndarray:
    """X_a X_b + Y_a Y_b + Z_a Z_b on n qubits (built from the SWAP identity)."""
    if a == b:
        raise ValueError("a bond needs two distinct qubits")
    dim = 2 ** n
    swap = np.zeros((dim, dim))
    for idx in range(dim):
        bit_a = (idx >> (n - 1 - a)) & 1
        bit_b = (idx >> (n - 1 - b)) & 1
        jdx = idx
        if bit_a != bit_b:
            jdx ^= (1 << (n - 1 - a)) | (1 << (n - 1 - b))
        swap[jdx, idx] = 1.0
    return 2.0 * swap - np.eye(dim)


def hamiltonian_from_bonds(bonds, n: int) -> np.ndarray:
    """Assemble sum_bonds J * H_ab as a dense real-symmetric matrix."""
    dim = 2 ** n
    H = np.zeros((dim, dim))
    cache: dict[tuple[int, int], np.ndarray] = {}
    for a, b, J in bonds:
        key = (min(a, b), max(a, b))
        if key not in cache:
            cache[key] = heisenberg_bond(key[0], key[1], n)
        H += float(J) * cache[key]
    return H


# ---------------------------------------------------------------- logical ops
def logical_paulis(code_vectors: np.ndarray):
    """Logical X, Z (2x2) are defined by the ordered code basis (|0_L>, |1_L>)."""
    return (np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex))


def effective_hamiltonian(H: np.ndarray, P0_basis: np.ndarray):
    """Exact low-energy effective Hamiltonian on the unperturbed code space.

    Uses the direct-rotation (des Cloizeaux / exact Schrieffer-Wolff) map:
    take the k lowest eigenvectors of H (k = dim of the code space), project the
    code basis onto them, and orthonormalise symmetrically.  The result is the
    k x k matrix of H in the rotated code basis, expressed in the coordinates of
    the original code basis.

    Returns (H_eff, eta) where eta = || U - P0_basis || is the isometry error.
    """
    k = P0_basis.shape[1]
    # only the k lowest eigenpairs are needed
    evals, low = _sp_eigh(H, subset_by_index=[0, k - 1], driver="evr")
    T = low.conj().T @ P0_basis             # k x k overlap
    # symmetric (Loewdin) orthonormalisation: U = low @ T (T^dag T)^{-1/2}
    w, v = np.linalg.eigh(T.conj().T @ T)
    if w.min() < 1e-8:
        raise RuntimeError("code space not adiabatically connected to the low-energy space")
    inv_sqrt = v @ np.diag(w ** -0.5) @ v.conj().T
    U = low @ T @ inv_sqrt                  # columns: dressed code basis
    H_eff = U.conj().T @ H @ U
    eta = np.linalg.norm(U - P0_basis, 2)
    return 0.5 * (H_eff + H_eff.conj().T), eta, evals


def distance_up_to_constant(A: np.ndarray, B: np.ndarray) -> float:
    """min_c || A - B - c*1 ||_op  for Hermitian A, B."""
    ev = np.linalg.eigvalsh(A - B)
    return 0.5 * (ev.max() - ev.min())
