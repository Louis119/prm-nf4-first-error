# 字段说明

所有 JSONL 都是「一行一个条目」。`scores` 与 `logits` 按步骤顺序排列，索引 0 为第 1 步。

## data/readout_panel/{model}/{backbone}/{readout}/repeat_{1,2}.jsonl

200 条机制面板，24 份文件。

| 字段 | 含义 |
|---|---|
| `id` | 轨迹 ID |
| `model` | `math_shepherd7b` 或 `rlhflow8b` |
| `backbone` | 主干精度：`bf16` 或 `nf4`（只量化 Transformer 层内投影） |
| `readout` | 读出条件：`A` 原生 BF16 输出头；`B` 同一权重转 FP32；`C` 对 B 的输出作 BF16 舍入 |
| `repeat` | 独立进程重复编号；两次逐值一致 |
| `label` | 标注首错的零基索引；`-1` 表示全正确 |
| `n_steps` | 步骤数 |
| `group_id` | 全局关联题组 ID（重采样单位） |
| `source` | 来源，如 `ProcessBench/math` |
| `step_index` / `positions` | 步骤编号与对应的 token 位置 |
| `logits` | 每步两个候选词的词表对数几率 `[[pos, neg], ...]` |
| `scores` | `softmax(logits)[pos]`，即步骤分 |
| `hidden_sha256` | 隐藏状态哈希；A/B/C 相同，证明读出面共享同一份隐藏状态 |
| `n_tokens` / `token_sha256` | 输入长度与哈希，用于核对文本未变 |

## data/main_pool_rescore/{model}/{bf16,nf4}.jsonl

全池同环境重评，A 读出、严格 `<`。

| 字段 | 含义 |
|---|---|
| `id` | 轨迹 ID |
| `n_steps` | 步骤数 |
| `scores` | 每步评分 |
| `token_sha256` | 输入哈希 |

标签与题组信息在 `data/main_pool_tables/` 与论文附录的匹配池说明中；分析脚本从 `evaluation_primary` 侧连接。

## data/restoration/

- `V11-{model}-{config}.jsonl`：`{id, scores, label, source}`，4,673 条。
- `rlhflow8b/{FSR-b10,Step-BCE-b10}.jsonl`：多出 `logits`、`positions`、`logits_dtype` 三个字段。

配置含义：`bf16` 原生基线、`nf4` 纯量化基线、`FSR-*` 面向首错的最大步骤损失恢复、`Step-BCE-*` 逐步 BCE 恢复、`Output-MSE-*` 输出重建、`Random-*` 随机恢复（后缀为随机种子）；`b10`/`b20` 为 10%/20% 额外权重载荷上限。

## data/derived/

| 文件 | 内容 |
|---|---|
| `A1_FULL.json` | 源 F20 的并列计数、`<`/`≤` 两规则下的 L/F、ΔL 与 bootstrap 区间 |
| `A1_F20_TIES_MATH.json` | MathShepherd 七个配置在 F20 阈值处的并列统计 |
| `A2_BOOTSTRAP.json` | FSR vs Step-BCE 同阈值对照（MathShepherd，含 10,000 次题组 bootstrap） |
| `A2_RLH.json` | 同上，RLHFlow |
| `A3_PANEL.json` | 读出面板逐条判定改变与栅格统计 |
| `A3_TIES.json` | 面板上的并列频率与判定裕度 |

## data/main_pool_tables/

论文主表的聚合结果（原始运行）：`primary_f20.csv`、`primary_fixed.csv`、`rlhflow_threshold_curves.csv`、`prefix_ordering.csv`、`calibration_uncertainty.csv` 等。字段与原分析流水线一致。
