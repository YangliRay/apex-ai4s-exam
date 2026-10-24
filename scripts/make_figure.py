"""Plot the re-computed error curves stored in results/*.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"01_correct": "01 correct", "02_wrong_coefficient": "02 wrong 2nd-order coefficient",
          "03_overclaimed_scaling": "03 same H as 01 (claims Δ$^{-1}$)",
          "attempt_1": "AI agent, attempt 1",
          "attempt_2": "AI agent, attempt 2 (no code execution)"}
STYLE = {"01_correct": dict(color="#1f77b4", marker="o"),
         "02_wrong_coefficient": dict(color="#d62728", marker="s"),
         "03_overclaimed_scaling": dict(color="#2ca02c", marker="^", linestyle="--", markerfacecolor="none"),
         "attempt_1": dict(color="#9467bd", marker="D"),
         "attempt_2": dict(color="#ff7f0e", marker="v")}

fig, ax = plt.subplots(figsize=(8.4, 4.2))
for name, label in LABELS.items():
    r = json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))
    m = r["measurements"][0]
    ax.loglog(m["delta"], m["error"], label=label, **STYLE[name])
d = np.array(m["delta"])
ax.loglog(d, 0.53 * d ** -0.5, color="gray", lw=0.8, ls=":", label=r"$0.53\,\Delta^{-1/2}$")
ax.loglog(d, 0.53 * d[0] ** 0.5 * d ** -1.0, color="black", lw=0.8, ls="-.", label=r"$\propto\Delta^{-1}$ (claimed by 03)")
ax.set_xlabel(r"penalty strength $\Delta$")
ax.set_ylabel(r"$\min_c\|H_{\rm eff}-H_{\rm target}-c\,\mathbb{1}\|$")
ax.set_title("Independent re-computation by the evaluator (hidden instance 1)")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
out = ROOT / "figures" / "error_vs_delta.png"
fig.savefig(out, dpi=160)
print("saved", out)
