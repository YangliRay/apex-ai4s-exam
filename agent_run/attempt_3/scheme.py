"""
Heisenberg-only analogue simulation of a #P-hard family of commuting (IQP-type)
Ising dynamics.

Encoding (per logical qubit, m = 4 physical spins, sites 0..3 of a block):
    |0_L> = |s>_{01} |s>_{23}                      (two dimer singlets)
    |1_L> = total-spin-0 state of two triplets (pairs 01 and 23)
  i.e. the S_tot = 0 sector of the block; Z_L = +1 <-> pair (2,3) is a singlet.

Physical Hamiltonian (Heisenberg terms only), Delta -> infinity:
    H_sim = sum_blocks [ Delta * sum_{a<b in block} H_ab  +  gamma_i * H_{23} ]
          + sum_{edges (i,j)} lam_ij * (H_{i0,j0} - H_{i0,j1} - H_{i1,j0} + H_{i1,j1})
  with lam_ij = sqrt(3 |J_ij| Delta / 2)  and  gamma_i = -h_i/2 - sum_j |J_ij|.

Logical target (Z-type Ising, all J_ij < 0, bipartite interaction graph):
    H_target = sum_{(i,j)} J_ij Z_i Z_j + sum_i h_i Z_i ,  J_ij in {-1/8,-1/4,-3/8}.

Exact facts used (see report.md): S^2 of pair (2,3) in every block commutes with
H_sim  =>  H_eff is exactly diagonal in the logical Z basis;  gamma_i*H_23 adds
exactly -gamma_i - 2 gamma_i Z_i;  second order gives -(2/3)(lam^2/Delta) Z_i Z_j
-(4/3)(lam^2/Delta)(Z_i+Z_j);  odd orders vanish on bipartite graphs; the leading
error is 4th order, eps = O(Delta^-1)  (rigorously eps <= 228 |E|^4 / Delta for
Delta >= 13.5 |E|^2, |E| = number of logical bonds; numerically eps ~ 0.1 |E| / Delta).

Hardness use: initial state |+_L>^n, evolve for t = pi, measure every block in the
logical X basis {|+_L>,|-_L>}; p = Prob(+...+).  For the instance sat_instance(...)
of a monotone 2-CNF phi, the ideal p equals #phi^2 * 2^-(2v+m)  (v variables, m clauses).
"""
import numpy as np

BLOCK_SIZE = 4


def code_basis():
    s = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)          # singlet
    tp = np.array([1, 0, 0, 0], dtype=complex)                        # |00>
    tm = np.array([0, 0, 0, 1], dtype=complex)                        # |11>
    t0 = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    k0 = np.kron(s, s)
    k1 = (np.kron(tp, tm) + np.kron(tm, tp) - np.kron(t0, t0)) / np.sqrt(3)
    return np.stack([k0, k1], axis=1)


# ---------------------------------------------------------------- hard family
# Logical instance format (JSON-serialisable):
#   {"edges": [[i, j, a_ij], ...],  "b": [b_0, ..., b_{n-1}],  ...}
# meaning the Z_4 phase polynomial  G(x) = sum a_ij x_i x_j + sum b_i x_i (mod 4)
# on bits x in {0,1}^n, with a_ij in {1,2,3}; the interaction graph is bipartite.
# The IQP amplitude  <+^n| exp(-i*pi*H_target) |+^n> = e^{i phi0} 2^-n sum_x i^G(x).

def _bipartite_random(n, rng):
    if n < 2:
        return {"edges": [], "b": [int(rng.integers(0, 4))], "kind": "random_bipartite"}
    while True:
        side = rng.integers(0, 2, size=n)
        if 0 < side.sum() < n:
            break
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if side[i] != side[j] and rng.random() < 0.7:
                edges.append([i, j, int(rng.integers(1, 4))])
    if not edges:  # guarantee at least one coupling
        i = int(np.where(side == 0)[0][0]); j = int(np.where(side == 1)[0][0])
        edges.append([min(i, j), max(i, j), int(rng.integers(1, 4))])
    b = [int(v) for v in rng.integers(0, 4, size=n)]
    return {"edges": edges, "b": b, "kind": "random_bipartite"}


def sat_instance(nvars, clauses):
    """Hard instances: monotone 2-CNF  AND_j (x_{p_j} OR x_{q_j})  on nvars variables.
    Logical qubits: 0..nvars-1 = variables, then (w_{j1}, w_{j2}) for every clause.
    G = sum_j [ w_j1 (2 + x_p + x_q) + w_j2 (1 - x_p - x_q) ]  (mod 4), so that
    sum_{x,w} i^G = (2 sqrt2 e^{-i pi/4})^m * #SAT."""
    m = len(clauses)
    n = nvars + 2 * m
    edges, b = [], [0] * n
    for j, (p, q) in enumerate(clauses):
        w1, w2 = nvars + 2 * j, nvars + 2 * j + 1
        b[w1] = 2
        b[w2] = 1
        for x in (p, q):
            edges.append([x, w1, 1])
            edges.append([x, w2, 3])
    return {"edges": edges, "b": b, "kind": "mon2sat", "nvars": nvars,
            "clauses": [list(c) for c in clauses]}


def hard_instance(nvars, clauses, c=1.0):
    """The instance used in the hardness theorem for additive error c*2^-n: duplicate the first
    clause until m >= nvars + max(0, ceil(log2 c) + 6) (duplicates do not change #SAT)."""
    clauses = [tuple(cl) for cl in clauses]
    need = nvars + max(0, int(np.ceil(np.log2(c))) + 6)
    while len(clauses) < need:
        clauses.append(clauses[0])
    return sat_instance(nvars, clauses)


def count_from_estimate(params, p_est):
    """Rounding step of the reduction: #SAT = round( sqrt(max(p_est,0)) * 2^(v + m/2) )."""
    v, m = params["nvars"], len(params["clauses"])
    return int(round(np.sqrt(max(p_est, 0.0)) * 2.0 ** (v + m / 2.0)))


def sample_params(n, rng):
    """n <= 3: random member of the bipartite Z_4-Ising family F_n (contains the hard
    instances); n >= 4: random monotone-2SAT gadget instance with m ~ n/3 clauses."""
    if n < 4:
        return _bipartite_random(n, rng)
    m = max(1, int(round(n / 3)))
    m = min(m, (n - 2) // 2)
    v = n - 2 * m
    clauses = []
    for _ in range(m):
        p, q = rng.choice(v, size=2, replace=False)
        clauses.append([int(min(p, q)), int(max(p, q))])
    return sat_instance(v, clauses)


def _ising_coeffs(n, params):
    """J_ij = -a_ij/8 (all negative), h_i = sum_j a_ij/8 + b_i/4  reduced to (-1,1]."""
    J = {}
    h = np.zeros(n)
    for i, j, a in params["edges"]:
        i, j = int(i), int(j)
        key = (min(i, j), max(i, j))
        J[key] = J.get(key, 0.0) - a / 8.0
        h[i] += a / 8.0
        h[j] += a / 8.0
    for i in range(n):
        h[i] += params["b"][i] / 4.0
    # reduce mod 2 (exp(-i*pi*H) is invariant under J -> J+2, h -> h+2)
    for k in J:
        J[k] = ((J[k] % 2.0) + 2.0) % 2.0 - 2.0          # representative in [-2, 0)
        if abs(J[k] + 2.0) < 1e-12:
            J[k] = 0.0
    h = ((h + 1.0) % 2.0) - 1.0                          # representative in [-1, 1)
    return J, h


def target_terms(n, params):
    J, h = _ising_coeffs(n, params)
    terms = []
    for (i, j), c in sorted(J.items()):
        if c != 0.0:
            s = ['I'] * n; s[i] = 'Z'; s[j] = 'Z'
            terms.append((float(c), ''.join(s)))
    for i in range(n):
        if h[i] != 0.0:
            s = ['I'] * n; s[i] = 'Z'
            terms.append((float(h[i]), ''.join(s)))
    return terms


def build(n, params, Delta):
    m = BLOCK_SIZE
    J, h = _ising_coeffs(n, params)
    terms = []
    gamma = -h / 2.0
    for (i, j), c in J.items():
        if c != 0.0:
            gamma[i] -= abs(c)
            gamma[j] -= abs(c)
    for i in range(n):
        base = i * m
        for a in range(m):
            for b in range(a + 1, m):
                Jab = Delta + (gamma[i] if (a, b) == (2, 3) else 0.0)
                terms.append({"type": "heisenberg", "sites": (base + a, base + b), "J": float(Jab)})
    for (i, j), c in sorted(J.items()):
        if c == 0.0:
            continue
        lam = np.sqrt(1.5 * abs(c) * Delta)
        for a, sa in ((0, 1.0), (1, -1.0)):
            for b, sb in ((0, 1.0), (1, -1.0)):
                terms.append({"type": "heisenberg", "sites": (i * m + a, j * m + b),
                              "J": float(sa * sb * lam)})
    return terms
