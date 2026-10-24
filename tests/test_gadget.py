"""Independent checks of the gadget algebra used by the evaluator.

Run with:  python -m pytest tests   (or simply: python tests/test_gadget.py)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apex_eval.heisenberg import hamiltonian_from_bonds  # noqa: E402

spec = importlib.util.spec_from_file_location("s01", ROOT / "submissions/01_correct/scheme.py")
s01 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s01)
C = s01.code_basis()
XL = np.array([[0, 1], [1, 0]]); ZL = np.diag([1, -1]); I2 = np.eye(2)


def block_penalty():
    return hamiltonian_from_bonds([(i, j, 1.0) for i in range(4) for j in range(i + 1, 4)], 4) + 6 * np.eye(16)


def test_block_spectrum():
    ev = np.round(np.linalg.eigvalsh(block_penalty()), 8)
    vals, counts = np.unique(ev, return_counts=True)
    assert vals.tolist() == [0, 4, 12] and counts.tolist() == [2, 9, 5]


def test_code_space_is_ground_space():
    H0 = block_penalty()
    assert np.allclose(H0 @ C, 0)


def test_first_order_terms():
    alpha, beta = 0.37, -0.81
    H1 = hamiltonian_from_bonds([(0, 3, alpha / np.sqrt(3)), (0, 2, 0.5 * (alpha / np.sqrt(3) - beta))], 4)
    got = C.conj().T @ H1 @ C
    want = alpha * XL + beta * ZL + 0.5 * (beta - np.sqrt(3) * alpha) * I2
    assert np.allclose(got, want)


def _second_order_two_ways():
    n = 8
    H0 = np.kron(block_penalty(), np.eye(16)) + np.kron(np.eye(16), block_penalty())
    H2 = hamiltonian_from_bonds([(0, 4, 1.0), (1, 5, -2.0), (2, 6, 1.0)], n)
    P = np.kron(C, C)
    # (i) direct resolvent on the excited space
    ev, U = np.linalg.eigh(H0)
    exc = ev > 1e-9
    G = U[:, exc] @ np.diag(1 / ev[exc]) @ U[:, exc].conj().T
    m1 = -P.conj().T @ H2 @ G @ H2 @ P
    # (ii) collapsed single-frequency form  -(1/8) P H2^2 P
    m2 = -(1 / 8) * P.conj().T @ H2 @ H2 @ P
    return m1, m2


def test_second_order_corrected_coefficients():
    m1, m2 = _second_order_two_ways()
    want = np.kron(XL, XL) - 0.5 * (np.kron(ZL, I2) + np.kron(I2, ZL)) - 2 * np.eye(4)
    assert np.allclose(m1, m2)
    assert np.allclose(m1, want)


def test_uncorrected_expansion_is_wrong():
    """The uncorrected manuscript expression -(4/3) XX + (2/3)(Z+Z') - 1/3 does not match."""
    m1, _ = _second_order_two_ways()
    wrong = -(4 / 3) * np.kron(XL, XL) + (2 / 3) * (np.kron(ZL, I2) + np.kron(I2, ZL)) - (1 / 3) * np.eye(4)
    assert not np.allclose(m1, wrong)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok ", name)
