"""L2 test submission (sanity check for evaluate_l2): Ising chain via the sparse gadget.

Target family (for this test only): K sum_i X_i X_{i+1} + sum_i h_i X_i on an open chain,
K = pi/4, h_i ~ U[-1.5, 1.5].  Encoding and couplings as in submissions/01_correct,
applied edge by edge; the first-order counterterm of each block cancels the
-(K/2) Z_L produced by every incident edge.
"""
import numpy as np

BLOCK_SIZE = 4
K = np.pi / 4
PATTERN = {0: 1.0, 1: -2.0, 2: 1.0}


def _pair_singlets(pairs):
    v = np.zeros(16, dtype=complex)
    (a, b), (c, d) = pairs
    for x in (0, 1):
        for y in (0, 1):
            bits = [0] * 4
            bits[a], bits[b], bits[c], bits[d] = x, 1 - x, y, 1 - y
            v[int("".join(map(str, bits)), 2)] += (-1) ** x * (-1) ** y * 0.5
    return v


def code_basis():
    zero = _pair_singlets([(0, 2), (1, 3)])
    one = (2 / np.sqrt(3)) * _pair_singlets([(0, 1), (2, 3)]) - zero / np.sqrt(3)
    return np.stack([zero, one], axis=1)


def sample_params(n, rng):
    return {"h": rng.uniform(-1.5, 1.5, size=n).tolist()}


def target_terms(n, params):
    terms = []
    for i in range(n - 1):
        s = ["I"] * n; s[i] = s[i + 1] = "X"
        terms.append((K, "".join(s)))
    for i, h in enumerate(params["h"]):
        s = ["I"] * n; s[i] = "X"
        terms.append((h, "".join(s)))
    return terms


def build(n, params, Delta):
    terms = []
    deg = [0] * n
    for i in range(n - 1):
        deg[i] += 1; deg[i + 1] += 1
    for i in range(n):
        q = 4 * i
        for a in range(4):
            for b in range(a + 1, 4):
                terms.append({"type": "heisenberg", "sites": (q + a, q + b), "J": Delta})
        alpha, beta = params["h"][i], deg[i] * K / 2
        terms.append({"type": "heisenberg", "sites": (q + 0, q + 3), "J": alpha / np.sqrt(3)})
        terms.append({"type": "heisenberg", "sites": (q + 0, q + 2), "J": 0.5 * (alpha / np.sqrt(3) - beta)})
    g = np.sqrt(Delta * K)
    for i in range(n - 1):
        for mu, w in PATTERN.items():
            terms.append({"type": "heisenberg", "sites": (4 * i + mu, 4 * (i + 1) + mu), "J": g * w})
    return terms
