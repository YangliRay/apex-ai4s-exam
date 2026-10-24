# 科研长程 Task 设计：Heisenberg 交换相互作用的经典模拟困难性

杨力睿（Lirui Yang）· ly344@cantab.ac.uk

超衍智能 AI4S 前沿研究员笔试材料：一份任务提案，以及提案中“评估方案”的可运行原型。

## 目录

| 路径 | 内容 |
|---|---|
| `report/` | 任务提案（`proposal.pdf` 及 LaTeX 源文件） |
| `src/apex_eval/` | 评测程序：读取提交的方案，用精确对角化独立复算有效哈密顿量与误差标度，按门槛判定（`evaluate.py`：L1；`evaluate_l2.py`：L2） |
| `submissions/` | 四份测试提交：正确方案、二阶系数错误、夸大误差标度、含非 Heisenberg 项 |
| `submissions_l2/` | L2 评测程序（`evaluate_l2.py`）的自检提交 |
| `agent_run/` | 真实的 AI 智能体运行：任务说明（`TASK.md`：L1，`TASK_L2.md`：L2）、智能体的提交与对话记录 |
| `results/` | 评测报告（`REPORT.md`）与原始数据 |
| `figures/` | 误差随 Δ 变化的图 |
| `tests/` | 二阶有效哈密顿量的独立验证 |
| `scripts/` | 作图脚本 |

## 运行

```bash
pip install -r requirements.txt
cd src && python -m apex_eval.evaluate ../submissions/0* ../agent_run/attempt_* --out ../results
```

## 结果概要

- **四份测试提交**：正确方案通过；系数错误、夸大标度、含非法项的方案分别在门槛 (b)、(d)、(a) 被拦下。
- **AI 智能体运行**（`agent_run/`）：在 claude.ai 全新对话中把 `agent_run/TASK.md` 交给 Claude Opus 5.5（effort: high），不做任何提示或干预，对话记录见各目录下的 `transcript.md`。

| 运行 | 环境 | 方案要点 | 复算误差 | 结论 |
|---|---|---|---|---|
| `attempt_1` | 普通对话，有代码执行 | 置换对称性使有效哈密顿量对块间耦合为偶函数，奇数阶项为零 | $\approx 0.3\text{–}0.7/\Delta$（随 $h$ 变化） | PASS；抽查的数值结论误差约 $2\times10^{-4}$ |
| `attempt_2` | 临时会话，**无代码执行** | 块间耦合 $g(\sigma_0-\sigma_1)\cdot(\sigma_4+\sigma_5)$，场项与其余部分严格对易 | $1.412/\Delta$，与 $h$ 无关 | PASS；手算预测前置系数 $55\pi^2/384\approx1.414$，与复算相差 0.1% |

两次运行都找到了误差为 $\Theta(\Delta^{-1})$ 的方案，优于测试提交 01 的 $\Theta(\Delta^{-1/2})$，且自报结论均与评测程序复算一致。

详见 `results/REPORT.md` 与 `figures/error_vs_delta.png`。
