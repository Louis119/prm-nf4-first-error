# NF4 量化下过程奖励模型首错定位的数值敏感性

本仓库公开论文《NF4 量化过程奖励模型的首错定位可靠性研究》的全部实验数据、分析脚本与图表。

**一句话结论**：NF4 对首错定位的影响不能用一个数概括；它的符号由三件事决定——阈值定在哪里、阈值处的并列算不算报警、阈值按哪一臂校准。权重精度本身带来的变化，小于这三项选择带来的变化。

## 数据说明了什么

| 现象 | 数值 | 数据 |
|---|---|---|
| 固定 0.5 的定位差 | 严格 `<` 得 +32 条，改 `≤` 变 −2 条 | `data/readout_panel/`, `data/main_pool_tables/` |
| 源 F20 的定位差 | 规则切换只让 ΔL 移动 0.2–0.3 个百分点，两模型结论不变 | `data/main_pool_rescore/` |
| 校准方式 | 各自阈值 +0.12 → 共同阈值 −0.74 [−1.35,−0.12]（Math） | `data/main_pool_rescore/` |
| 恢复目标 | 各自阈值 −8 条，共同阈值 +8 条（反向） | `data/restoration/` |
| 读出机制 | 25 个改变判定的条目，触发步两个 logit 完全相等、评分恰为 0.5 | `data/readout_panel/` |

## 1/8 栅格

评分是 `sigmoid(logit_pos − logit_neg)`。BF16 读出的 logit 量级在 19–28，落在 [16,32) 区间，ULP 恰为 0.125，所以差值只能取 0.125 的整数倍，评分落在 1/8 的 logit 栅格上。条件 A 与 C 的 logits **3930/3930** 在栅格上，条件 B（FP32 输出头）为 **0/3930**。这不是权重量化的产物，是读出算术的产物。

## 目录

```
data/
  readout_panel/      24 份逐条评分（2 模型 × 2 主干 × A/B/C × 2 重复，各 200 条）
  main_pool_rescore/  4 份全池重评（2 模型 × BF16/NF4，各 4,996 / 4,677 条）
  restoration/        MathShepherd 14 个恢复配置 + RLHFlow FSR/Step-BCE 两个配置
  derived/            分析输出的 JSON（A1_FULL、A2_*、A3_* 等）
  main_pool_tables/   论文主表的聚合 CSV（H800 / Ascend 原始运行）
code/                 全部分析脚本，纯本地、无 GPU 依赖
figures/              论文插图（矢量/位图）
paper/                论文 PDF
```

字段说明见 [DATA_DICTIONARY.md](DATA_DICTIONARY.md)，逐条复现命令见 [REPRODUCE.md](REPRODUCE.md)。**哪些结果能重算、哪些只能核对汇总数据、哪些未公开，见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。**

## 复现

无需 GPU。以下三组结果可由 `data/` 中的逐条评分在笔记本上重算（脚本使用相对路径，无需改配置）：

```bash
python code/a3_readout_panel.py     # 读出面板：A/B/C 的判定改变与栅格统计
python code/a3b_ties.py             # 面板上的并列频率与判定裕度
python code/analyze_full.py         # 源 F20 的并列与规则敏感性（含 bootstrap）
python code/a2_bootstrap.py         # FSR vs Step-BCE 同阈值对照
```

## 数据说明与边界

- 评价集：ProcessBench、PRMBench、Hard2Verify 的匹配池，4,673 条 / 2,853 个关联题组；留出集 1,796 条与面板无重叠。
- 重采样单位是全局 `group_id`；跨来源或跨含错/正确层的同一题组不拆分。
- 论文主表来自原始运行（H800 与 Ascend）；`main_pool_rescore/` 是在另一套加速卡上的同环境重评，用于并列检验，与发布值的跨环境漂移 ≤3 条。
- 所有指标都是**配对**的：同一批题目上的 BF16 与 NF4。

## 引用

```
杨宛榆. NF4 量化过程奖励模型的首错定位可靠性研究. 2026.
```

## License

代码 MIT，数据 CC BY 4.0。

---

## English summary

This repository releases the raw per-step scores, derived statistics and analysis code for a study on how NF4 weight quantization changes first-error localization in process reward models. The headline result is negative and methodological: the sign of the NF4 effect is set by the threshold location, by whether ties count as alarms, and by whether thresholds are calibrated per arm; the effect of weight precision itself is smaller than these three choices. No GPU is required to reproduce any reported number.
