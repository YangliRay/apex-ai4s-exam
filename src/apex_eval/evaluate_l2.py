"""Evaluator for L2 submissions (agent chooses target family, encoding and layout).

Automatic layers (as in evaluate.py, generalised):
  Gate (a)  every physical term is a Heisenberg exchange on valid, distinct qubits;
  Gate (b)  for n = 2, 3 (physical qubits <= MAX_QUBITS) and hidden instances drawn
            with the submission's own sample_params, the exact low-energy effective
            Hamiltonian on the declared code space converges to the declared target;
  Gate (d)  the error exponent declared in claims.json agrees with re-computation.

Items that need expert judgement (hardness source, reduction chain, J_max(n)
derivation, status labels) are collected verbatim into the report for review;
the evaluator only records the measured J_max at each Delta as supporting data.

Usage:  python -m apex_eval.evaluate_l2 <submission_dir> [--out results]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from .heisenberg import (distance_up_to_constant, effective_hamiltonian,
                         hamiltonian_from_bonds)

DELTAS = np.array([1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5])
FIT_POINTS = 4
N_HIDDEN = 3
SEED = 20261024
MAX_QUBITS = 12
TOL_CONVERGE_EXP = -0.25
TOL_FINAL_REL = 0.02
TOL_EXP_CLAIM = 0.10

PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]], dtype=complex),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0]).astype(complex)}


def pauli_sum(terms, n):
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for coef, s in terms:
        if len(s) != n or any(c not in PAULI for c in s):
            raise ValueError(f"bad Pauli string {s!r} for n={n}")
        op = np.array([[1.0 + 0j]])
        for c in s:
            op = np.kron(op, PAULI[c])
        H += complex(coef) * op
    if np.linalg.norm(H - H.conj().T) > 1e-9:
        raise ValueError("target Hamiltonian is not Hermitian")
    return H


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"l2_{path.name}", path / "scheme.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    claims = json.loads((path / "claims.json").read_text(encoding="utf-8"))
    return mod, claims


def code_space(mod, n):
    m = mod.BLOCK_SIZE
    C = np.asarray(mod.code_basis(), dtype=complex)
    if C.shape != (2 ** m, 2):
        raise ValueError(f"code_basis() must be {2**m} x 2")
    if np.linalg.norm(C.conj().T @ C - np.eye(2)) > 1e-10:
        raise ValueError("code basis is not orthonormal")
    P = np.array([[1.0 + 0j]])
    for _ in range(n):
        P = np.kron(P, C)
    return P


def static_check(mod, n, params):
    N = n * mod.BLOCK_SIZE
    problems, bonds = [], []
    for t in mod.build(n, params, 1e3):
        a, b = tuple(t.get("sites", (None, None)))
        if t.get("type") != "heisenberg":
            problems.append(f"n={n}: 非 Heisenberg 项 type={t.get('type')!r} sites={a},{b}")
        elif a == b or not (0 <= a < N and 0 <= b < N):
            problems.append(f"n={n}: 非法的键 sites={a},{b}")
        else:
            bonds.append((a, b, float(t["J"])))
    return problems


def run_instance(mod, n, params):
    N = n * mod.BLOCK_SIZE
    P0 = code_space(mod, n)
    Ht = pauli_sum(mod.target_terms(n, params), n)
    eps, jmax, etas = [], [], []
    for d in DELTAS:
        bonds = [(t["sites"][0], t["sites"][1], float(t["J"])) for t in mod.build(n, params, d)]
        H = hamiltonian_from_bonds(bonds, N)
        He, eta, _ = effective_hamiltonian(H, P0)
        eps.append(distance_up_to_constant(He, Ht))
        jmax.append(max(abs(J) for _, _, J in bonds))
        etas.append(eta)
    eps = np.array(eps)
    lx, ly = np.log(DELTAS[-FIT_POINTS:]), np.log(np.maximum(eps[-FIT_POINTS:], 1e-300))
    p, logA = np.polyfit(lx, ly, 1)
    rel = eps[-1] / max(np.linalg.norm(Ht, 2), 1e-12)
    ok = bool(p < TOL_CONVERGE_EXP and rel < TOL_FINAL_REL)
    return {"n": n, "params": params, "delta": DELTAS.tolist(), "error": eps.tolist(),
            "eta": etas, "J_max": jmax, "fitted_exponent": float(p),
            "fitted_prefactor": float(np.exp(logA)), "relative_error_at_max_delta": float(rel),
            "pass": ok}


def evaluate(path: Path) -> dict:
    mod, claims = load(path)
    rng = np.random.default_rng(SEED)
    ns = [n for n in (2, 3, 4) if n * mod.BLOCK_SIZE <= MAX_QUBITS]
    report = {"submission": path.name, "level": "L2", "block_size": mod.BLOCK_SIZE,
              "n_tested": ns, "gates": {}, "flags": [], "measurements": [],
              "for_expert_review": {k: claims.get(k) for k in
                                    ("description", "hardness_source", "basis", "J_max_scaling")},
              "claims": claims.get("claims", [])}

    instances = {n: [_jsonable(mod.sample_params(n, rng)) for _ in range(N_HIDDEN)] for n in ns}

    problems = [p for n in ns for prm in instances[n] for p in static_check(mod, n, prm)]
    report["gates"]["a_pure_heisenberg"] = {"pass": not problems, "details": sorted(set(problems))}
    if problems:
        report["verdict"] = "FAIL（门槛 a：含有非 Heisenberg 项或非法的键）"
        return report

    ok_b = True
    for n in ns:
        for prm in instances[n]:
            r = run_instance(mod, n, prm)
            ok_b &= r["pass"]
            report["measurements"].append(r)
    report["gates"]["b_reproduces_target"] = {
        "pass": bool(ok_b),
        "criterion": f"每个 n、每个隐藏实例：拟合指数 < {TOL_CONVERGE_EXP} 且最大 Δ 处相对误差 < {TOL_FINAL_REL}"}

    p2 = [r["fitted_exponent"] for r in report["measurements"] if r["n"] == 2]
    p_meas = float(np.median(p2))
    notes, ok_d = [], True
    try:
        p_claim = float(claims.get("error_exponent"))
        if abs(p_meas - p_claim) > TOL_EXP_CLAIM:
            ok_d = False
            notes.append(f"声明的误差指数 {p_claim} 与 n=2 复算值 {p_meas:.3f} 不符")
    except (TypeError, ValueError):
        notes.append("claims.json 未给出可解析的 error_exponent")
    report["gates"]["d_claims_consistent"] = {"pass": ok_d, "details": notes,
                                              "measured_exponent_n2": p_meas}
    if not ok_d:
        report["flags"].append("自报误差指数与复算不一致")

    report["verdict"] = ("FAIL（门槛 b：有效哈密顿量不收敛到声明的目标）" if not ok_b else
                         "FAIL（门槛 d：自报结果与复算不一致）" if not ok_d else
                         "自动门槛全部通过；困难性来源与 J_max(n) 待专家评审")
    return report


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


def to_markdown(r):
    L = [f"### {r['submission']}（L2）", "", f"**自动评测结论：{r['verdict']}**", ""]
    for name, g in r["gates"].items():
        L.append(f"- 门槛 `{name}`：{'通过' if g['pass'] else '未通过'}")
        L += [f"  - {d}" for d in g.get("details", [])]
    if r["measurements"]:
        L += ["", "| n | 拟合误差指数 | 前置系数 | 最大 Δ 处相对误差 | Δ=10⁵ 时 J_max | 结果 |",
              "|---|---|---|---|---|---|"]
        for m in r["measurements"]:
            L.append(f"| {m['n']} | {m['fitted_exponent']:+.3f} | {m['fitted_prefactor']:.3g} | "
                     f"{m['relative_error_at_max_delta']:.2e} | {m['J_max'][-1]:.3g} | "
                     f"{'通过' if m['pass'] else '未通过'} |")
    fe = r["for_expert_review"]
    L += ["", "**待专家评审（原文摘录）**", "",
          f"- 困难性来源：{fe.get('hardness_source')}",
          f"- 基：{fe.get('basis')}",
          f"- 声明的 J_max(n)：{fe.get('J_max_scaling')}"]
    if r.get("claims"):
        L += ["", "| 结论 | 状态 |", "|---|---|"]
        L += [f"| {c.get('id')}: {c.get('statement')} | {c.get('status')} |" for c in r["claims"]]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for sub in a.submissions:
        r = evaluate(sub)
        (a.out / f"{sub.name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        (a.out / f"{sub.name}_REPORT.md").write_text(to_markdown(r), encoding="utf-8")
        print(f"{sub.name:28s} {r['verdict']}")


if __name__ == "__main__":
    main()
