# 复现范围说明

仓库里的材料分三档。论文里每个数字都能归到其中一档。

## A. 可从原始逐条评分独立重算

| 结果 | 数据 | 脚本 |
|---|---|---|
| 读出面板 A/B/C 的判定改变、1/8 栅格、并列频率 | `data/readout_panel/`（24 份） | `code/a3_readout_panel.py`、`code/a3b_ties.py` |
| 源 F20 处的并列计数与 `<`/`≤` 规则敏感性 | `data/main_pool_rescore/`（4 份） | `code/a1_f20_rules.py` |
| 同阈值下 FSR − Step-BCE 的 ΔL 与区间 | `data/restoration/`（16 份） | `code/a2_same_threshold.py` |

运行方式（在仓库根目录）：

```bash
python code/reproduce_all.py
```

所有脚本使用相对路径，`data/gold_labels.csv` 提供 `id → label / group_id / source`，无需修改任何路径。

## B. 只能核对汇总数据

论文主表（表 4-2、4-3、4-5、4-6 及附录 B–D）来自 H800 与另一套加速卡上的原始运行。逐条评分的成百上千次前向不在本仓库内，仓库提供的是聚合结果：

- `data/main_pool_tables/primary_f20.csv`、`primary_fixed.csv`、`rlhflow_threshold_curves.csv` 等；
- 每个聚合文件都含计数、分母、阈值与区间，可据此核对论文表中的数字。

若要从头重算这些表，需要在有模型权重的机器上重跑评分；`code/score_full_server.py` 记录了做法，但它依赖 Ascend 环境与模型权重，**不在本地复现范围内**。

## C. 未公开、仅作者本地归档

- 留出题组八配置（BF16 / NF4 / FSR10 / Step-BCE10）的逐条评分；
- `SCIENTIFIC_PROTOCOL.md`（留出评价的冻结协议）；
- 原始运行日志与作业清单。

这几项涉及尚未整理归档的原始运行数据，论文表 A-1 已如实标注为“仅本地归档”。

## 数据与隐私

- 仓库中的论文为**公开版**，封面学号已移除；含完整信息的学生版由作者本人保存。
- 分析脚本不包含个人目录、主机名或账号信息。
- 基准数据（ProcessBench、PRMBench、Hard2Verify）的题目文本不随仓库分发，仅保留 `id`、标签与题组编号。
