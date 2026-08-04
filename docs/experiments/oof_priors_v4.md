# 实验记录：OOF 统计先验 v4

## 变化与数据纪律

- 分支：`codex/oof-priors-v4`
- 父版本：`codex/official-evaluation-v1`
- 唯一变化：`P1` 的训练行先验由完整训练聚合改为 leave-one-row-out 聚合；模型、P1 特征组、训练轮数、随机种子与冻结验证切分不变。

对每一个训练行，菌株蛋白均值不包含该行自身的已观测蛋白；可精确匹配的处理行，其化合物 FC 均值同样不包含自身。验证和提交行仍仅查询完整训练聚合。这避免了训练特征直接含有当前行标签，同时不向验证或提交阶段引入任何额外数据。

运行：

```powershell
python -m goai_baseline.train --config configs/baseline.yaml --variant p1_oof_priors --run-dir runs\v4_p1_oof
python -m goai_baseline.score --config configs/baseline.yaml --run-dir runs\v4_p1_oof
```

## 结果

| 方法 | S1 abs R2 | S1 FC PCC | S2 abs R2 | S2 FC PCC | S3 abs R2 | S3 FC PCC | time abs R2 | time FC PCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 蛋白均值 | 0.860 | 0.152 | 0.906 | 0.181 | 0.862 | 0.146 | 0.902 | 0.164 |
| P1 完整训练聚合 | 0.816 | 0.137 | 0.801 | 0.162 | 0.807 | 0.131 | 0.800 | 0.147 |
| v4 P1 OOF 聚合 | 0.828 | 0.137 | 0.819 | 0.162 | 0.822 | 0.131 | 0.819 | 0.147 |

OOF 版本在四个场景的绝对 R2 均较原 P1 小幅提高，FC PCC 基本不变。它仍未超过 B0 的 FC PCC，且逐蛋白 R2 中位数依然为负，因此不能视为响应预测突破。

## 决策

v4 取代原 P1 作为今后统计先验模型的**可信训练参考**：它修正了训练行目标统计泄漏，并没有造成验证退化。该修正不改变“未见实体缺乏语义表征”的根本限制。

下一阶段应在 v4 这一无泄漏参照上分别测试观测校准和实体可迁移表征；两者不得与新的目标函数或模型结构同时引入。
