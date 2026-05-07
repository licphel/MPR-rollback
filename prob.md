# Known Problems

## P1: 修复成功定义与"分布式违规"论点脱钩（最严重）
**FIXED**

`base.py` 中：
```python
post_repair_violated = not bool(orig_violated_indices & sanitized_set)
```

定义为"至少脱敏了一个违规步骤"即算成功。这使得 single-pivot 只需碰巧选中那一个违规步骤就能达到 VR=1.0——无法体现多点归因的优势。论文核心论点是"多步骤分布式违规单点回滚不够用"，但实验设计实际上让 single-pivot 用单点就完美解决了所有问题（VR=1.0），与论点自相矛盾。

**实际数据**：数据集中全部30条违规轨迹均有 ≥2 个违规步骤（2–7个）。用**严格定义**（所有违规步骤均被脱敏）重新统计：

| Strategy | Strict SR (all vio steps covered) | Partial only | Loose SR (≥1 covered) |
|---|---|---|---|
| no_repair | 0.000 (0/30) | 0 | 0.000 |
| full_sanitization | 0.767 (23/30) | 7 | 1.000 |
| single_pivot | 0.567 (17/30) | 13 | 1.000 |
| multi_pivot | 0.700 (21/30) | 9 | 1.000 |

宽松定义下四组策略全部 VR=1.0，掩盖了真实差异；严格定义下 multi-pivot (0.700) 显著优于 single-pivot (0.567)，差异 +13.3pp，这才是支撑核心论点的直接证据。

**Fix**: 在 paper 中同时报告 loose SR 和 strict SR（将后者命名为 Full-Coverage Rate 或 FCR），并以 strict SR 作为主要安全指标。`base.py` 中也应增加严格定义下的指标计算。

---

## P2: 违规标注与风险估计器之间的循环依赖
**FIXED**

`evaluate.py` dry_run 模式：
```python
return 0.75 if bool(traj.violated) else 0.0
```

风险估计器在 dry_run 下退化为直接读标签。实验用 DeepSeek 运行了3次但 std 极小（如 `0.007`），说明实际结果几乎确定性地与标签对齐。这使得指标与标注规则高度相关，并非真正意义上的独立安全评估。

**Fix**: 在 paper 中说明 risk estimator 输出与 ground-truth labels 的 Spearman 相关系数，以及两者之间存在的 feedback loop，并讨论其对实验有效性的影响。

---

## P3: Projected risk 公式与代码不自洽
**FIXED**

`03_preliminary.tex` 公式：
$$\hat{r}_t(S) = r_t - \sum_{i \in S} w_i$$

但 $r_t = \sum c_i$（未加权），$w_i = c_i \cdot \frac{i}{t}$（加权），两者量纲不同，相减无意义——早期步骤的风险贡献会被系统性低估。

`multi_pivot.py:47` 实际执行 `projected -= contrib`（contrib = $c_i \cdot \text{weight}$），与公式行为一致但数学上同样错误。

**Fix**: 贪心循环中每选入一个步骤后，将其 risk 置零并重新对整个前缀求和，而非做减法估计。已在 `multi_pivot.py` 中修复。论文公式也应更新为：选入步骤 $i$ 后，$\hat{r}_t = \sum_{j \notin S} c_j$。

---

## P4: 数据集规模太小，统计检验说明不足

130条轨迹（30违规/100安全），3次重复。问题：
- VR=1.0 这个"完美修复"结论建立在 30 条违规轨迹上，置信区间极宽
- DeepSeek temperature=0.3，3次结果 std 极小（Final Risk std≈0.007），说明输出高度确定，"3次重复"实际上没有意义
- `05_experiments.tex` 声称 "paired t-test, p<0.001" 但未说明被测对象。需在 paper 中明确：t-test 的被测变量是 **30条违规轨迹上逐案的 final_risk 差值**，而非3次运行的均值差

**Fix**: 用30条违规轨迹上 bootstrapping（B=1000）计算指标的95% CI，替代或补充跨运行的 ±std；明确说明 paired t-test 的具体设置。

---

## P5: n=10 结果与核心论断方向相反

`results/n=10/aggregate.json` 中，multi-pivot 的 SC=0.9235，而 full sanitization 的 SC=1.0——multi-pivot 的脱敏代价接近全量脱敏，远高于 single-pivot 的 SC=0.595。与论文主结论（multi-pivot 代价更低）方向相反。

**Fix**: 解释 n=10 是噪声（并说明为何），或删除这组结果以免引发疑问。

---

## P6: 容忍阈值设计缺乏消融

$$\tau(t) = 0.5 + 0.001t$$

阈值随步数线性增长，意味着越长的轨迹越"宽容"——这是否反直觉？论文中没有消融实验验证这个设计对结果的敏感性。

**Fix**: 增加 $\tau(t) = 0.5$（常数阈值）或 $\tau(t) = 0.5 + 0.01t$ 的消融结果，证明主要结论对阈值设计鲁棒。

---

## P7: Recency weighting 的必要性未验证

$$w_i = c_i \cdot \frac{i}{t}$$

使得较早的步骤即使 $c_i$ 很高也可能得分偏低。论文没有实验验证这个权重是否真的有帮助（vs 直接用 $w_i = c_i$）。

**Fix**: 增加 ablation：去掉位置权重的 multi-pivot vs 带权重的 multi-pivot，或在讨论中承认这是一个未验证的设计选择。

---

## P8: 模型名称不一致

`05_experiments.tex:25` 写的是 `DeepSeek-v4-flash`，代码 `evaluate.py:44` 中是 `MODEL = "deepseek-chat"`（DeepSeek-V3）。实验不可复现。

**Fix**: 核实实际调用的模型，统一论文与代码中的名称。

---

## P9: "safety drift" 引用可信度存疑

`dhodapkar2026safetydrift` 在 introduction 和 related work 中均出现，但这是 2026 年的文献——需要确认这篇文章是否已发表/公开，否则 reviewer 会 flag。

**Fix**: 核实引用可达性；若尚未 arxiv，改为用自己的语言描述该现象并 self-contain。

---

## 优先级总结

| 优先级 | 问题 | 改动量 |
|--------|------|--------|
| **必修** | P1: 多步骤违规子集分析（支撑核心论点） | 中等（需跑实验） |
| **必修** | P3: projected risk 公式与代码不一致 | 小（修公式） |
| **必修** | P8: 模型名称核实 | 小 |
| **建议** | P2: risk estimator 与标注的循环依赖说明 | 小（加讨论） |
| **建议** | P4: bootstrapping CI + t-test 说明 | 小 |
| **建议** | P5: n=10 结果处理 | 小 |
| **建议** | P6: 阈值消融实验 | 中等（需跑实验） |
| **建议** | P7: recency weighting ablation | 中等（需跑实验） |
| **建议** | P9: 引用核实 | 小 |
