# 逐条复现

## 环境

Python 3.10+，仅需 `numpy`（脚本主要用标准库）。无需 GPU、无需模型权重。

```bash
python -V
```

## 1. 读出面板：判定改变全部来自阈值并列

```bash
python code/a3_readout_panel.py   # 产出 A3_PANEL.json
python code/a3b_ties.py           # 产出 A3_TIES.json
```

预期：A 与 C 的 logits 3930/3930 落在 1/8 栅格，B 为 0/3930；改变判定的条目数 Math BF16 9、NF4 10，RLH BF16 2、NF4 4；每条的触发步在 A 下两个 logit 完全相等。

## 2. 源 F20 的并列与规则敏感性

```bash
python code/a1_f20_rules.py       # 源 F20 的并列与规则敏感性
```

预期（严格 `<`，各自源 F20 阈值）：ΔL Math +0.12、RLH −0.59；改 `≤` 后 Math −0.18、RLH −0.79；并列步占比 1.50%/1.43%（Math）、0.76%/0.71%（RLH）。

## 3. 同阈值对照：恢复目标的差值来自阈值

```bash
python code/a2_same_threshold.py  # 两个模型，各自阈值 vs 共同阈值
```

预期（MathShepherd）：各自阈值 ΔL=−0.235；同阈值 +0.235 [−0.17,+0.64]，ΔF=+0.942 [+0.38,+1.57]。

## 4. 仅想核对论文数字

直接读 `data/derived/*.json`，每个文件都含点估计、区间、样本量与 bootstrap 设定（seed 20261110，10,000 次）。

## 注意事项

- 所有比较都是配对的：同一批题目上的两种精度。
- 重采样以 `group_id` 为单位，跨层同组不拆。
- 脚本使用相对路径，在仓库根目录直接运行即可。
