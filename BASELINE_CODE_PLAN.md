# GOAI 虚拟细胞文档基线复现：代码计划

## 1. 目标与边界

第一阶段只复现《虚拟细胞-解题思路》中定义的 Baseline，不提前引入第五章的提分模块。

本阶段交付包括：

1. 蛋白均值基线。
2. Exact Matched Control 诊断基线。
3. 条件 one-hot 编码的两层 MLP。
4. 文档建议的 P1-P4 累积特征实验。
5. 四个冻结验证场景的独立评估。
6. 使用官方提交 metadata 生成预测文件的推理流程。
7. 可复现配置、实验清单、测试和数据泄漏检查。

本阶段明确不做：残差分解、共享 Encoder 多头、Transformer、菌株基因组表征、SMILES/Morgan FP、蛋白序列或通路先验、批次校准分支、多目标 loss、后处理校准和集成。这些内容只保留接口和后续路线说明，不进入首版 Baseline。

“忠实复现”的含义是复现文档的方法、模型和实验顺序；文档中的错误字段名、错误示例代码和与真实数据冲突的硬编码数字不照抄，而是通过兼容层修正并记录。

## 2. 公平性与数据使用边界

允许读取的输入只有：

- `WAYB_WAYC_metadata_train_val.csv`
- `WAYB_WAYC_proteome_raw_train_val.csv`
- `WAYB_WAYC_metadata_test.csv`

提交 metadata 仅用于推理条件编码、行顺序和提交索引。所有需要目标值的统计量只能由 `split_final == "train"` 的蛋白质组计算。

实现时增加 `data_audit` 预检：校验声明的文件路径、样本 ID 对齐和 feature contract；Git 忽略官方数据、预测、模型、缓存和运行日志。

## 3. 真实数据契约

### 3.1 元数据

真实 train/validation metadata 为 8,958 行、15 列，冻结划分为：

| split_final | 行数 |
|---|---:|
| train | 5,920 |
| val_chem_only | 1,065 |
| val_strain_only | 1,547 |
| val_both | 269 |
| val_time | 157 |

首版生物条件输入只使用：

- 菌株：`Strains`
- 化合物/扰动：`perturbation_no_concentration`
- 培养基：`Medium`
- 温度：`Temperature`
- 时间：`pert_time`，单位由 `pert_time_unit` 校验为分钟

`split_final`、`strain_role`、`chemical_role` 仅用于划分和审计，不能进入模型。`data_source`、`instrument`、`Yeast_cell_plate`、`protein_well` 只用于 Matched Control 匹配或后续批次研究，不进入文档首版 MLP。

### 3.2 蛋白矩阵

- 原始矩阵：8,958 行、5,243 个蛋白。
- 元数据和蛋白矩阵必须通过 `sample_ID` 显式一对一连接，禁止依赖当前行顺序。
- 只用 5,920 个训练行计算每个蛋白的缺失率。
- 保留条件：训练缺失率严格小于 0.80。
- 真实数据按该规则保留 4,422 个蛋白，删除 821 个；不得硬编码 PDF 中的 4,232。
- 所有有限观测强度必须大于 0，之后执行 `log2`；缺失位置保持 `NaN`，训练张量中可填 0，但必须由 mask 屏蔽。

预处理产出一个不可变 feature contract，至少记录：蛋白名及顺序、训练样本 ID、阈值、原始/保留维度、字段映射、类别词表、数据文件哈希和代码版本。

## 4. 文档字段兼容层

| PDF 名称 | 真实字段/实现 | 处理原则 |
|---|---|---|
| `strain` | `Strains` | 只改字段名 |
| `chemical` | `perturbation_no_concentration` | 不使用非全局唯一的裸 `pert_id` |
| `medium` | `Medium` | 只改字段名 |
| `temperature` | `Temperature` | 只改字段名 |
| `time` | `pert_time` + `pert_time_unit` | 真实数据为分钟 |
| `plate` | `Yeast_cell_plate` | 只用于 control 匹配 |
| `product_id` | 扰动名称规则 | Water/DMSO 为 control，Quality Control 为 QC |
| `np.concat` | `np.concatenate` | 修复不可运行的示例代码 |
| 4,232 个蛋白 | 训练行动态计算为 4,422 | 保留 PDF 的 `<80%` 方法，不保留错误结果 |

所有兼容修正写入运行 manifest 和解读文档，不静默修改。

## 5. 要实现的基线阶梯

### B0：蛋白均值

对每个保留蛋白计算训练行非缺失 `log2` 均值，对任意验证或 test 样本输出同一个向量。

用途：验证 sample 对齐、蛋白过滤、尺度、mask、指标和提交格式。

### B1：Exact Matched Control

对每个处理样本查找 Water/DMSO 对照。精确匹配键采用：

```text
data_source
instrument
Yeast_cell_plate
Strains
Medium
Temperature
pert_time
pert_time_unit
```

同一键下存在多个对照时，对每个蛋白按非缺失 `log2` 值求均值。处理真值或 matched control 对应蛋白缺失时，该位置不参与指标；完全找不到 control 的样本不进入 matched-subset 指标。

该基线有两个严格分开的用途：

- 在训练集内部，为 P1 化合物 delta 先验匹配训练 control。
- 在本地验证中，复现 PDF 的 Matched Control 诊断结果。

它是验证诊断模型；提交预测统一由条件模型生成。任何 Matched Control 验证值都不得进入 MLP 训练或特征拟合。

### B2/P0：条件 one-hot MLP

严格采用 PDF Demo 的固定设置：

| 项目 | 设置 |
|---|---|
| 输入 | 菌株、化合物、培养基、温度、时间 one-hot |
| 隐层 | 256 -> 256 |
| 激活 | ReLU |
| Dropout | 每个隐层后 0.1 |
| 输出 | 4,422 维 log2 蛋白向量，由 feature contract 动态决定 |
| 优化器 | Adam |
| 学习率 | `1e-3` |
| Epoch | 50 |
| Loss | 全训练矩阵上的 mask-aware MSE |
| Seed | 42 |

为忠实复现 Demo，首个实验使用 full-batch、无 scheduler、无 early stopping、保存第 50 个 epoch。工程实现允许 CPU/GPU，但两者都要固定随机种子并记录设备和软件版本。

类别词表只在训练行上拟合；验证和 test 的新实体采用全零 one-hot，不把实体是否属于 val/test 的角色字段提供给模型。

### P1：加入训练统计先验

在 P0 条件编码后拼接：

1. 菌株蛋白均值向量：按训练菌株对 4,422 个 log2 蛋白求非缺失均值。
2. 化合物 delta 均值向量：仅对训练处理样本，与训练 Water/DMSO exact control 匹配后计算逐蛋白 `treatment - control`，再按化合物求均值。

新菌株使用训练集全局蛋白均值；新化合物使用训练集中全部有效 treatment delta 的逐蛋白均值。局部缺失先验也按对应全局向量补齐。

本阶段按 PDF 原意使用完整训练集统计，不做 leave-one-out/OOF target encoding。它可能造成训练侧 target-encoding 偏乐观，作为复现风险记录，待“复现后修正”阶段再做 OOF 对照。

### P2：加入条件交叉

在 P1 基础上加入 PDF 明确给出的两个类别交叉：

- `Strains x Medium`
- `perturbation_no_concentration x Temperature`

交叉词表只在训练行拟合，未知组合变为全零。

### P3：加入时间 sin/cos

在 P2 基础上加入：

```text
theta = 2 * pi * pert_time_minutes / max_train_time_minutes
time_sin = sin(theta)
time_cos = cos(theta)
```

真实数据直接使用 15、30、60、90、120、240 分钟，不使用 PDF 示例中的小时序列。

### P4：加入化合物名称 hash

在 P3 基础上加入 32 维、确定性的化合物名称 hash 向量，数值映射到 `[0, 1]`。PDF 的 MD5 示例只有 16 个字节，却索引 32 个维度，原代码无法运行；实现采用带计数器的 MD5 扩展得到 32 字节，保持“名称 -> 固定 32 维 hash”的方法意图，并在 manifest 中记录算法版本。

## 6. 固定实验矩阵

实验按下列顺序累积运行，每一步只增加一组特征：

| 实验 ID | 模型/特征 | 目的 |
|---|---|---|
| `b0_mean` | 训练蛋白均值 | 流水线与最低基线 |
| `b1_matched_control` | exact control | 复现强诊断基线 |
| `p0_onehot` | one-hot + MLP | PDF 主模型 |
| `p1_priors` | P0 + 菌株/化合物统计先验 | 复现 P1 |
| `p2_crosses` | P1 + 两个条件交叉 | 复现 P2 |
| `p3_time` | P2 + time sin/cos | 复现 P3 |
| `p4_hash` | P3 + 32 维化合物 hash | 复现 P4 完整版 |

本轮不做超参数搜索。所有 MLP 共用同一训练配置，确保差异只来自新增特征。

## 7. 评估口径

每个实验必须分别报告：

- `val_chem_only`
- `val_strain_only`
- `val_both`
- `val_time`

基础复现指标为 PDF 使用的三项：

1. `log2 RMSE`：只在真实观测位置计算。
2. `Global R2`：在有效观测位置按统一实现计算，并用单元测试锁定口径。
3. `protein R2 median`：逐蛋白计算 R2 后取中位数；观测不足或真值方差为 0 的蛋白记为不可评估，不以 0 代替。

结果同时生成两张表：

- 全部验证样本上的模型评估表。
- exact matched-control 可用子集上的 B0/B1/MLP 公平对比表。

PDF 的诊断表只作为复现对标，不冒充本地实验结果。由于真实训练过滤得到 4,422 而非 4,232 个蛋白，本地数值允许与 PDF 表不同；差异必须结合样本子集、蛋白集合和指标实现进行解释。

## 8. 拟建代码仓库结构

```text
go-AI/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   └── baseline.yaml
├── src/goai_baseline/
│   ├── __init__.py
│   ├── schema.py
│   ├── audit.py
│   ├── data.py
│   ├── preprocess.py
│   ├── controls.py
│   ├── features.py
│   ├── datasets.py
│   ├── model.py
│   ├── loss.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── manifest.py
├── scripts/
│   ├── run_baseline_ladder.ps1
│   └── verify_submission.py
├── tests/
│   ├── test_alignment.py
│   ├── test_train_only_filter.py
│   ├── test_masked_loss.py
│   ├── test_control_matching.py
│   ├── test_feature_no_leakage.py
│   ├── test_metrics.py
│   └── test_submission_contract.py
├── docs/
│   ├── baseline_method.md
│   └── reproduction_errata.md
└── runs/                  # 本地生成，Git 忽略
```

不提交官方数据文件、checkpoint 或预测结果。README 只说明用户如何把官方文件放到本地配置路径。

## 9. 命令与运行产物

预期统一命令：

```powershell
python -m goai_baseline.audit --config configs/baseline.yaml
python -m goai_baseline.preprocess --config configs/baseline.yaml
python -m goai_baseline.evaluate --config configs/baseline.yaml --variant b0_mean
python -m goai_baseline.evaluate --config configs/baseline.yaml --variant b1_matched_control
python -m goai_baseline.train --config configs/baseline.yaml --variant p0_onehot
python -m goai_baseline.train --config configs/baseline.yaml --variant p1_priors
python -m goai_baseline.train --config configs/baseline.yaml --variant p2_crosses
python -m goai_baseline.train --config configs/baseline.yaml --variant p3_time
python -m goai_baseline.train --config configs/baseline.yaml --variant p4_hash
python -m goai_baseline.predict --config configs/baseline.yaml --run runs/<run_id>
python scripts/verify_submission.py runs/<run_id>/prediction.csv
```

每个 run 独立保存：解析后的配置、feature contract、训练日志、最终 checkpoint、四场景指标、逐蛋白指标、预测摘要、环境版本和 manifest。运行目录不进入 Git。

## 10. 测试与验收标准

### 数据验收

- `sample_ID` 在各表唯一，元数据与蛋白行集合完全一致。
- 训练/验证划分行数与数据契约一致。
- 过滤只读取训练行，得到 4,422 个蛋白。
- log2 前所有有限值严格为正。
- mask 与原始 `NaN` 位置逐元素一致。

### 泄漏验收

- 特征编码器和统计先验只能 `fit(train)`。
- 验证标签变化不会改变训练特征或模型输入。
- 提交 metadata 可以 transform，但不能触发任何目标统计更新。
- 输入审计只接受配置中声明的文件路径。

### 模型验收

- mask 为 0 的位置改变目标填充值时，loss 不变。
- P0 参数、层数、激活、dropout、优化器和 epoch 与 PDF 一致。
- 固定 seed 的 CPU 重复运行指标一致；GPU 非确定算子须显式报出。
- 所有四个验证场景都有独立结果，不输出混合总分代替分场景结果。

### 提交验收

- 行顺序与 `WAYB_WAYC_metadata_test.csv` 完全一致，共 4,454 行。
- 索引列名为 `sample_ID`，无重复 ID。
- 蛋白列及顺序与 feature contract 完全一致。
- 无 `NaN`、`inf`、重复列和额外索引列。
- 数值为 log2 尺度，运行记录声明 `prediction_scale=log2`。

## 11. 实施顺序与完成定义

1. 建立仓库骨架、依赖、配置和 Git 数据隔离。
2. 完成数据审计、字段兼容、对齐、过滤、log2 和 mask。
3. 实现并核验 B0 蛋白均值。
4. 实现 exact control 匹配，复现 B1 诊断子集。
5. 实现 P0 MLP 和 mask-aware 训练。
6. 依次加入 P1、P2、P3、P4，每步产出独立 run。
7. 生成四场景对比表和复现差异表。
8. 用 P4 最终 run 对官方提交 metadata 推理，执行提交契约检查。
9. 补齐 README、方法解读、测试，并做一次从空 `runs/` 开始的完整复现。

第一阶段只有在上述九步全部可由命令重跑、测试通过且结果有 manifest 时才算完成。之后再进入方法纠错和创新实验，不在复现过程中边跑边改基线定义。
