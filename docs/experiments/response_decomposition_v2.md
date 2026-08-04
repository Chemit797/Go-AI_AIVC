# 实验记录：响应分解 v2（负结果）

## 假设与唯一设计变化

- 分支：`codex/response-decomposition-v2`
- 父版本：`codex/official-evaluation-v1`
- 假设：将绝对蛋白质组拆为“无化合物背景 + 化合物条件响应”，并用训练内匹配对照的 FC 监督响应分支，可以减少模型向背景均值收缩，提升官方响应模块。
- 保持不变：训练/验证切分、蛋白筛选、P0 的完整条件特征、50 epoch、Adam、学习率、种子和缺失 mask。

模型为：

```text
background(strain, medium, temperature, time)
  + response(strain, chemical, medium, temperature, time)
  = predicted log2 proteome
```

背景分支刻意不接收化合物。响应分支在 751 个训练对照行上以零 FC 监督，在 5,066 个可精确匹配的训练处理行上以 `treatment - matched_control` 监督。总共使用 21,479,484 个已观测 FC 目标值。训练目标为：

```text
masked_MSE(predicted_proteome, observed_proteome)
+ 1.0 * masked_MSE(predicted_response, matched_FC)
```

本版本也提供独立的训练、推理和提交校验入口：

```powershell
python -m goai_baseline.response_train --config configs/baseline.yaml --run-dir runs\v2_response_fc1 --fc-weight 1.0
python -m goai_baseline.response_predict --config configs/baseline.yaml --run-dir runs\v2_response_fc1
```

## 运行结果

训练可以完成，且仅用提交元数据可生成 4,454 行、4,422 个蛋白列的有限 `log2` 预测并通过列顺序和 ID 校验。然而结果不支持该配置进入主线。

| 方法 | S1 abs R2 | S1 FC PCC | S2 abs R2 | S2 FC PCC | S3 abs R2 | S3 FC PCC | time abs R2 | time FC PCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 蛋白均值 | 0.860 | 0.152 | 0.906 | 0.181 | 0.862 | 0.146 | 0.902 | 0.164 |
| P0 one-hot MLP | -0.146 | 0.124 | -0.499 | 0.133 | -4.078 | 0.083 | 0.858 | 0.127 |
| v2 背景 + 响应，FC 权重 1.0 | 0.289 | 0.054 | -1.344 | 0.062 | -2.570 | 0.037 | 0.431 | 0.049 |

训练曲线进一步解释了失败原因：absolute MSE 在 50 epoch 降至 5.876，但 FC MSE 从 epoch 1 的 0.145 上升至 61.605。训练 FC 的标准差仅为 0.376，说明响应分支为帮助绝对值拟合而产生了远大于真实扰动幅度的输出；当前权重不足以维持可识别的背景/响应分解。

## 决策与下一版

本版本是**可复现的负结果**，不合并到基线或候选提交模型。它排除了“仅增加一个同权重 FC 辅助损失即可改善响应预测”的假设。

下一版只修改优化策略：对响应监督使用更高权重与逐 epoch 的响应幅度监控，检验能否先阻止响应分支承担背景丰度。不会同时引入外部表征、测量上下文或新的特征交叉；如果 FC 仍无法改善，响应分解架构将被停止，而不是靠继续调参掩盖问题。
