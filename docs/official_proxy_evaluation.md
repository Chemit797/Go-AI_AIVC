# 官方评分代理评估（v1）

## 目的与边界

截至本版本，官方手册给出了虚拟细胞主榜的六个评分模块和方向性定义，但没有提供可执行评分脚本、完整聚合公式、零方差处理、缺失值细则或最终对照匹配实现。本模块因此是**本地模型选择代理**，不是对线上分数的声称或替代。

实现位置为 `goai_baseline.official_metrics`，命令入口为：

```powershell
python -m goai_baseline.score --config configs/baseline.yaml --baseline b0_mean
python -m goai_baseline.score --config configs/baseline.yaml --baseline b1_matched_control
python -m goai_baseline.score --config configs/baseline.yaml --run-dir runs\p0_verified
```

每次运行会输出按 `val_chem_only`、`val_strain_only`、`val_both` 和 `val_time` 拆分的 CSV。它不生成一个虚假的单一总分。

## 对应关系

| 官方手册模块 | 本地代理实现 | 当前限制 |
|---|---|---|
| 绝对保真度（20%） | 逐样本及逐蛋白 PCC/R2 的中位数 | 不假定最终的官方聚合方式 |
| 匹配对照原始 FC（25%） | `y_hat_treat - y_control` 与 `y_treat - y_control` 的 PCC | 使用手册列出的精确匹配字段及已观测对照；只在可匹配位置评分 |
| 上下文均值残差（20%） | 从训练处理样本 FC 冻结同匹配上下文的均值，再计算残差 PCC | “冻结批次”的精确定义待官方脚本确认；当前采用全部匹配字段 |
| 药物均值残差（20%） | 从训练处理样本 FC 冻结同化合物均值，再计算残差 PCC | 不假定额外分层或加权 |
| 双重未知/时间外推（10%） | 单独报告绝对保真度和原始 FC | 不构造未发布的权重组合 |
| 高效应蛋白（5%） | 在 `|FC_true| > 1` 上报告方向准确率、PCC、precision、recall、F1 | 阈值及检出形式取自手册摘要，最终细节须以官方为准 |

## 数据纪律

- 用于上下文和药物残差的均值只从 `split_final == "train"` 的匹配处理样本计算。
- 评估时的对照仅用于构造隐藏评估标签对应的 FC，不是推理输入。
- 无匹配对照、无共同观测蛋白或无法定义相关性的情况保留为缺失并报告覆盖率，不用零填充制造分数。
- 预测和真值的绝对保真度依旧通过观测 mask 计算。

## 版本决策

这个版本只扩展评估，没有改动基线训练、特征或预测逻辑。下一版必须先用该代理评估现有 `B0`、`B1`、`P0` 至 `P4`，确定哪一类模块最弱；随后才引入 FC/残差模型，且每次只改变一个假设。

首次完整运行的数值、解释与下一版决策见 [实验记录：官方评分代理 v1](experiments/official_evaluation_v1.md)。
