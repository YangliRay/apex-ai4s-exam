# attempt_3 对话记录（L2）

- 平台：claude.ai，新建普通对话（未放入任何项目；有代码执行与联网检索）
- 模型：Claude Opus 5.5，effort level：high
- 附件：`agent_run/TASK_L2.md`
- 人工干预次数：0（只发送了下面这一条消息）
- 说明：以下为对话的文字记录；中间的工具调用过程未导出。智能体交付的三个文件即本目录下的 `scheme.py`、`claims.json`、`report.md`。

---

## 用户

（附件：TASK_L2.md）

请阅读附件 TASK_L2.md，独立完成其中的任务。全程自主工作，不需要问我问题。最后请提供 scheme.py、claims.json、report.md 三个文件供下载。

## 智能体

I'll work through the theory, verify it numerically, and then write the three files. First I'll check the key citations.

Citations confirmed. Now the core numerics: I'll build the 4-spin singlet-code gadget and check the second-order coefficients and the error exponent.

任务完成，scheme.py、claims.json、report.md 三个文件已提供下载。我用自己复刻的评测程序跑过：n = 2、3 的误差都随 Δ⁻¹ 收敛（拟合指数 −1.000），所有项都是 Heisenberg 项。

**主要结果**

* J_max(n)：严格证明的上界是 J_max = O(n⁴·2^{n/2})。数值支持更紧的 Θ(n·2^{n/2})，但这一条标为猜想。最大的耦合是块内的 Δ，块间耦合只有 O(√Δ)。
* 其他资源：物理比特 4n 个，没有辅助比特；演化时间 t = π。
* 困难性：归约到 #Monotone-2SAT 的 #P-complete 性（Valiant, SIAM J. Comput. 8, 410 (1979)）。这是最坏情形的结果，不依赖任何猜想；加性误差 c·2⁻ⁿ 对任意常数 c 都成立。
* 测量基：输出概率在逻辑基下定义：初态 |+_L⟩ⁿ，测量逻辑 X 基。

**方案**

* 编码：每个逻辑比特用 4 个自旋，码空间是块的总自旋 0 子空间。块与块之间只在位点 0、1 上连 4 条 ±λ 的 Heisenberg 键。
* 严格对角：每块 pair(2,3) 的总自旋与整个 H_sim 对易，所以有效哈密顿量在逻辑 Z 基下严格对角，误差只是相位误差，不会泄漏到别的逻辑态。
* 场项精确：单比特场就用这个守恒量本身来加，因此没有任何高阶修正。
* 二部图条件：逻辑相互作用图是二部图时，奇数阶严格为零，误差从 Δ⁻¹ᐟ² 降到 Δ⁻¹。三角形图作为对照，数值拟合指数是 −0.506。
* 子句 gadget：每个子句用两个辅助逻辑比特和一个只含两体项的精确恒等式。它保持二部性，并给出逻辑振幅 = #φ·2^{−v−m/2}。300 个随机实例核对，误差在 10⁻¹⁶ 量级。

**数值检查**

* 在 16 和 20 个物理自旋上做了端到端计算：物理系统的振幅收敛到理想值，误差 ∝ 1/Δ。
* 对"不做绝热缀饰、直接接通哈密顿量"的协议，我证明了误差界，数值也吻合。

**需要你知道的几点**

* 物理基不成立：我证明了单块的计算基测量无法区分两个码态，所以只声称逻辑基下的困难性。
* 制备与测量是假设：报告假设能理想地制备码态 |+_L⟩ 并在逻辑 X 基测量。实际做的话需要时变的交换门脉冲，超出了静态哈密顿量的设定。
* padding 不算作主结果：判据的容差随总比特数 n 收缩。只要加入空闲逻辑比特，就能在字面上把 J_max 降到 2^{O(n^δ)}，但这只是让判据变严，物理代价并没有降低。报告里单独列为 C13，主结果仍是 2^{n/2} 量级。

Sources:

* [Bremner–Montanaro–Shepherd, arXiv:1504.07999](https://arxiv.org/abs/1504.07999v2)
* [Cubitt–Montanaro–Piddock, arXiv:1701.05182](https://www.arxiv.org/abs/1701.05182)
* [Bremner–Jozsa–Shepherd, arXiv:1005.1407](https://arxiv.org/pdf/1005.1407)
* [Fujii–Morimae, NJP 19, 033003](https://repository.kulib.kyoto-u.ac.jp/dspace/handle/2433/226522)
* [Bravyi–DiVincenzo–Loss, arXiv:1105.0675](https://www.arxiv.org/pdf/1105.0675)
* [Valiant 1979, SIAM J. Comput.](https://epubs.siam.org/doi/abs/10.1137/0208032)
* [♯SAT (Wikipedia)](https://en.wikipedia.org/wiki/%E2%99%AFSAT)
* [Greenhill 2000, Comput. Complexity](https://unpaywall.org/10.1007%2FPL00001601)
* [Dyer–Greenhill](https://algorithms.leeds.ac.uk/wp-content/uploads/sites/117/2017/09/dg00b.pdf)
