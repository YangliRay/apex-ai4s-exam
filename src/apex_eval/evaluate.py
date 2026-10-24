"""Prototype evaluator for Heisenberg-only simulator schemes (two-block demo).

This implements, at the smallest non-trivial size, the automatic layers of the
evaluation plan in report/proposal.pdf (Section 5):

  Gate (a)  static check   : every term is a Heisenberg exchange J (XX+YY+ZZ).
  Gate (b)  re-computation : on hidden instances the evaluator builds the
                             Hamiltonian itself, extracts the exact low-energy
                             effective Hamiltonian on the declared code space,
                             and checks that it converges to the encoded target.
  Gate (d)  self-report    : the error exponent/prefactor declared in
                             claims.json must agree with the re-computed ones.

The evaluator never uses numbers reported by the submission for scoring; it only
uses the submission's *scheme* (scheme.py) and compares the *claims* against its
own computation.

Usage:  python -m apex_eval.evaluate <submission_dir> [<submission_dir> ...]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from .heisenberg import (distance_up_to_constant, effective_hamiltonian,
                         hamiltonian_from_bonds)

# ------------------------------------------------------------------ settings
K_TARGET = np.pi / 4                       # Ising coupling of the hard family
DELTAS = np.array([1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5])
FIT_POINTS = 4                             # fit the exponent on the largest Deltas
N_HIDDEN = 3                               # hidden target instances
SEED = 20261023                            # fixed so that reports are reproducible

TOL_CONVERGE_EXP = -0.25                   # gate (b): error must decay at least this fast
TOL_FINAL_REL = 0.02                       # gate (b): relative error at the largest Delta
TOL_EXP_CLAIM = 0.10                       # gate (d): |p_measured - p_claimed|
TOL_PREF_CLAIM = 0.25                      # gate (d): relative prefactor mismatch

XL = np.array([[0, 1], [1, 0]], dtype=complex)
I2 = np.eye(2)


def target_hamiltonian(K: float, h: tuple[float, float]) -> np.ndarray:
    """Two logical qubits of the Ising family:  K X1X2 + h1 X1 + h2 X2."""
    return (K * np.kron(XL, XL) + h[0] * np.kron(XL, I2) + h[1] * np.kron(I2, XL))


# ------------------------------------------------------------ loading inputs
def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location(f"sub_{path.name}", path / "scheme.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    claims = json.loads((path / "claims.json").read_text(encoding="utf-8"))
    return mod, claims


# ----------------------------------------------------------------- gate (a)
def gate_static(mod, h) -> tuple[bool, list[str], list]:
    """Only Heisenberg terms on valid, distinct qubits."""
    problems = []
    n = mod.N_QUBITS
    terms = mod.build(K_TARGET, h, 1e3)
    bonds = []
    for t in terms:
        kind = t.get("type")
        a, b = t.get("sites", (None, None))
        if kind != "heisenberg":
            problems.append(f"非 Heisenberg 项: type={kind!r}, sites={a},{b}")
            continue
        if a == b or not (0 <= a < n and 0 <= b < n):
            problems.append(f"非法的键: sites={a},{b}")
            continue
        bonds.append((a, b, t["J"]))
    return (not problems), problems, bonds


def bonds_of(mod, h, delta):
    return [(t["sites"][0], t["sites"][1], t["J"])
            for t in mod.build(K_TARGET, h, delta) if t.get("type") == "heisenberg"]


def code_space(mod) -> np.ndarray:
    """Tensor product of the declared single-block code bases (16 x 2 each)."""
    C = np.asarray(mod.code_basis(), dtype=complex)
    if C.shape != (16, 2):
        raise ValueError("code_basis() must return a 16 x 2 array")
    if np.linalg.norm(C.conj().T @ C - np.eye(2)) > 1e-10:
        raise ValueError("code basis is not orthonormal")
    return np.kron(C, C)


# ----------------------------------------------------------------- gate (b)
def error_curve(mod, h):
    P0 = code_space(mod)
    Ht = target_hamiltonian(K_TARGET, h)
    eps, etas, jmax = [], [], []
    for d in DELTAS:
        bonds = bonds_of(mod, h, d)
        H = hamiltonian_from_bonds(bonds, mod.N_QUBITS)
        He, eta, _ = effective_hamiltonian(H, P0)
        eps.append(distance_up_to_constant(He, Ht))
        etas.append(eta)
        jmax.append(max(abs(J) for _, _, J in bonds))
    return np.array(eps), np.array(etas), np.array(jmax), np.linalg.norm(Ht, 2)


def fit_power_law(x, y):
    """log y = log A + p log x on the last FIT_POINTS points."""
    lx, ly = np.log(x[-FIT_POINTS:]), np.log(y[-FIT_POINTS:])
    p, logA = np.polyfit(lx, ly, 1)
    return float(p), float(np.exp(logA))


# --------------------------------------------------------------- evaluation
def evaluate(path: Path) -> dict:
    mod, claims = load_submission(path)
    rng = np.random.default_rng(SEED)
    hidden = [tuple(rng.uniform(-1.5, 1.5, size=2)) for _ in range(N_HIDDEN)]

    report = {"submission": path.name, "description": claims.get("description", ""),
              "hidden_instances": [list(map(float, h)) for h in hidden],
              "gates": {}, "flags": [], "measurements": []}

    # (a) static
    ok_a, problems, _ = gate_static(mod, hidden[0])
    report["gates"]["a_pure_heisenberg"] = {"pass": ok_a, "details": problems}
    if not ok_a:
        report["verdict"] = "FAIL（门槛 a：含有非 Heisenberg 项）"
        return report

    # (b) independent re-computation on every hidden instance
    ok_b = True
    for h in hidden:
        eps, etas, jmax, norm_t = error_curve(mod, h)
        p, A = fit_power_law(DELTAS, eps)
        rel_final = eps[-1] / norm_t
        inst_ok = (p < TOL_CONVERGE_EXP) and (rel_final < TOL_FINAL_REL)
        ok_b &= inst_ok
        report["measurements"].append({
            "h": list(map(float, h)),
            "delta": DELTAS.tolist(), "error": eps.tolist(), "eta": etas.tolist(),
            "J_max": jmax.tolist(),
            "fitted_exponent": p, "fitted_prefactor": A,
            "relative_error_at_max_delta": float(rel_final), "pass": bool(inst_ok)})
    report["gates"]["b_reproduces_target"] = {
        "pass": bool(ok_b),
        "criterion": f"拟合指数 < {TOL_CONVERGE_EXP} 且最大 Δ 处相对误差 < {TOL_FINAL_REL}"}

    # (d) consistency between claims and re-computation
    p_meas = float(np.median([m["fitted_exponent"] for m in report["measurements"]]))
    A_meas = float(np.median([m["fitted_prefactor"] for m in report["measurements"]]))
    ok_d, notes = True, []
    if "error_exponent" in claims:
        dp = abs(p_meas - claims["error_exponent"])
        if dp > TOL_EXP_CLAIM:
            ok_d = False
            notes.append(f"声明的误差指数 {claims['error_exponent']} 与复算值 {p_meas:.3f} 不符")
    if "error_prefactor" in claims and ok_b:
        dA = abs(A_meas - claims["error_prefactor"]) / A_meas
        if dA > TOL_PREF_CLAIM:
            ok_d = False
            notes.append(f"声明的前置系数 {claims['error_prefactor']} 与复算值 {A_meas:.3f} 相差 {dA:.0%}")
    report["gates"]["d_claims_consistent"] = {"pass": ok_d, "details": notes,
                                              "measured_exponent": p_meas,
                                              "measured_prefactor": A_meas}
    if not ok_d:
        report["flags"].append("自报与复算不一致（声明审计将其标记为夸大或错误）")

    if not ok_b:
        report["verdict"] = "FAIL（门槛 b：有效哈密顿量不收敛到目标）"
    elif not ok_d:
        report["verdict"] = "FAIL（门槛 d：自报结果与复算不一致）"
    else:
        report["verdict"] = "PASS"
    return report


# ------------------------------------------------------------------ reports
def to_markdown(r: dict) -> str:
    lines = [f"### {r['submission']}", "", f"{r['description']}", "",
             f"**结论：{r['verdict']}**", ""]
    for name, g in r["gates"].items():
        mark = "通过" if g["pass"] else "未通过"
        lines.append(f"- 门槛 `{name}`：{mark}")
        for d in g.get("details", []) or []:
            lines.append(f"  - {d}")
    if r["measurements"]:
        lines += ["", "| 隐藏实例 (h1, h2) | 拟合误差指数 | 前置系数 | 最大 Δ 处相对误差 | 结果 |",
                  "|---|---|---|---|---|"]
        for m in r["measurements"]:
            h = ", ".join(f"{x:+.3f}" for x in m["h"])
            lines.append(f"| ({h}) | {m['fitted_exponent']:+.3f} | {m['fitted_prefactor']:.3f} | "
                         f"{m['relative_error_at_max_delta']:.2e} | {'通过' if m['pass'] else '未通过'} |")
    for f in r["flags"]:
        lines.append(f"\n> 标记：{f}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("submissions", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for sub in args.submissions:
        r = evaluate(sub)
        reports.append(r)
        (args.out / f"{sub.name}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{sub.name:28s} {r['verdict']}")
    md = "# 评测报告\n\n" + "\n".join(to_markdown(r) for r in reports)
    (args.out / "REPORT.md").write_text(md, encoding="utf-8")
    return reports


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
