# GOAI 虚拟细胞 Baseline

[English README](README_EN.md) | [方法说明](docs/baseline_method.md) | [文档与真实数据勘误](docs/reproduction_errata.md)

这是世界人工智能开源大赛 GOAI AI for Research 虚拟细胞方向的**可复现文档基线**。任务是在不给定处理样本 protein profile 或 control profile 的前提下，根据菌株、化合物和实验条件，直接预测酵母处理后的 `log2` 蛋白质组强度向量。

这个仓库的目的不是抢先堆叠复杂模型，而是把官方解题思路中的基线做成可审计、可复跑、可比较的工程参照。后续的残差分解、分子结构、菌株基因组、批次校准和蛋白先验，都应当在此基线上逐项比较。

## 1. Baseline 思路

文档将基线设计为由简单到复杂的实验阶梯：先确认数据流与评估口径无误，再逐组加入特征。这里完整复现该顺序，而不是直接跳到提分模型。

```mermaid
flowchart LR
    A["官方 metadata + train/validation 蛋白矩阵"] --> B["sample_ID 对齐"]
    B --> C["仅训练行缺失过滤"]
    C --> D["log2 target + mask"]
    D --> E["B0 / B1 诊断"]
    E --> F["P0-P4 条件特征 + MLP"]
    F --> G["四个冻结验证场景"]
    G --> H["提交条件推理"]
```

| 实验 | 方法 | 它回答的问题 |
|---|---|---|
| `b0_mean` | 每个蛋白的训练集非缺失 `log2` 均值 | sample 对齐、过滤、尺度、mask、指标和提交格式是否正确？ |
| `b1_matched_control` | 相同测量/生物上下文的 Water/DMSO 对照均值 | “假设没有额外药物效应”能达到多强？模型是否必须超过这个强诊断基线？ |
| `p0_onehot` | 五类条件 one-hot -> 两层 MLP | 最小可训练条件响应模型是否跑通？ |
| `p1_priors` | P0 + 菌株蛋白均值 + 化合物平均 delta | 训练统计先验是否有价值？ |
| `p2_crosses` | P1 + 菌株×培养基、化合物×温度 | 显式条件交互是否有价值？ |
| `p3_time` | P2 + 时间 sin/cos | 连续时间编码是否有价值？ |
| `p4_hash` | P3 + 32 维化合物名称 hash | 在没有外部化学结构时，能否区分新化合物名称？ |

P0-P4 是**累积式**消融：P4 包含 P0、P1、P2、P3 的全部特征。首轮不进行超参数搜索，保证新增特征是实验间唯一的核心变化。

## 2. 数据契约与预处理

本项目使用下列三个官方文件；它们默认放在仓库根目录，也可以在 `configs/baseline.yaml` 改成本地路径。

| 文件 | 用途 | 是否包含目标值 |
|---|---|---|
| `WAYB_WAYC_metadata_train_val.csv` | 训练/验证条件和冻结划分 | 否 |
| `WAYB_WAYC_proteome_raw_train_val.csv` | 训练/验证原始蛋白强度 | 是 |
| `WAYB_WAYC_metadata_test.csv` | 提交条件、sample_ID 和提交行顺序 | 否 |

### 预处理纪律

1. 用 `sample_ID` 显式对齐 metadata 和蛋白矩阵，绝不依赖 CSV 行顺序。
2. 仅以 `split_final == "train"` 的行计算每个蛋白缺失率；保留缺失率严格小于 `0.80` 的蛋白。
3. 对观测到的正 raw intensity 取 `log2`；`NaN` 表示未检测到，不是零。
4. 构建同形状 `mask`。训练中缺失目标可临时填 0，但 mask 为 0，因此不会产生误差或梯度。
5. 每次运行写出 `feature_contract.json`，冻结蛋白名称和顺序、过滤阈值、训练样本数和输出尺度。

当前发布数据按这一规则保留 **4,422** 个蛋白。这个值由代码动态计算，不能硬编码为 PDF 示例中的数字。

### 真实字段映射

PDF 示例使用了若干占位字段名。代码使用真实发布字段：

| 语义 | 真实字段 |
|---|---|
| 菌株 | `Strains` |
| 化合物/扰动名称 | `perturbation_no_concentration` |
| 培养基 | `Medium` |
| 温度 | `Temperature` |
| 时间 | `pert_time` + `pert_time_unit` |
| 实验板 | `Yeast_cell_plate` |

裸 `pert_id` 在不同 `data_source` 下不是全局唯一，不能作为化合物实体。完整兼容记录见 [reproduction_errata.md](docs/reproduction_errata.md)。

## 3. 三类模型与特征细节

### B0：蛋白均值

对每个保留蛋白计算训练集非缺失 `log2` 均值，对所有验证或 test 条件重复输出这一向量。它不利用任何条件信息，因而不是有效的科学模型；它的价值在于快速发现数据处理、输出列、尺度或评分实现问题。

### B1：Exact Matched Control

对每个**处理**样本，查找 Water/DMSO 对照。以下八个字段必须完全相同：

`data_source`、`instrument`、`Yeast_cell_plate`、`Strains`、`Medium`、`Temperature`、`pert_time`、`pert_time_unit`。

多个可用对照时按蛋白取非缺失均值。只有处理与对应 control 都观测到的蛋白位置参与 B1 指标。B1 是本地验证的强诊断基线，以及 P1 训练 delta 的构造工具；提交预测统一由条件模型生成。

### P0：条件编码 MLP

P0 使用训练行拟合的菌株、化合物、培养基、温度和时间 one-hot，随后使用固定结构：

```text
input
  -> Linear(input_dim, 256) -> ReLU -> Dropout(0.1)
  -> Linear(256, 256)       -> ReLU -> Dropout(0.1)
  -> Linear(256, n_proteins)
```

训练严格使用全训练矩阵的 mask-aware MSE、Adam (`lr=1e-3`)、50 epoch、seed 42、无 early stopping、无 scheduler。新菌株或新化合物在 P0 中会得到全零的未知类别块，这正是它作为最小基线的局限。

### P1-P4：文档特征阶梯

| 阶段 | 新增特征 | 训练范围与新实体 fallback |
|---|---|---|
| P1 | 每个菌株的训练蛋白均值向量；每种化合物的训练 `(treatment-control)` 平均 delta 向量 | 所有目标统计量仅由训练行计算；新菌株回退到全局蛋白均值，新化合物回退到全局 delta |
| P2 | `Strains × Medium`、`chemical × Temperature` one-hot | 词表只在训练行拟合；未知组合为全零 |
| P3 | `sin(2πt/Tmax)`、`cos(2πt/Tmax)` | `Tmax` 只由训练时间点决定；发布数据使用分钟 |
| P4 | 化合物名称的确定性 32 维 hash | 任意名称都可生成值，但它没有化学语义 |

P1 首轮遵循文档，使用完整训练集聚合。它可能造成训练侧 target encoding 偏乐观；OOF/leave-one-group-out 先验属于“复现完成后的修正实验”，不能和这套基准混在一起。

## 4. 验证与结果如何解读

所有变体都必须分别报告：

| 验证集 | 泛化问题 |
|---|---|
| `val_chem_only` | 新化合物 |
| `val_strain_only` | 新菌株 |
| `val_both` | 新菌株与新化合物同时出现 |
| `val_time` | 时间外推 |

每个报告包含 `log2 RMSE`、`Global R2` 和 `protein R2 median`。由于蛋白绝对丰度相似性很高，`Global R2` 可能很高却没有学到扰动差异；因此需要始终在 exact-control 可用子集上将 MLP 与 B1 并排比较，并观察逐蛋白 R2。

运行输出不会纳入 Git。实际结果位于 `runs/<run_id>/metrics.csv`、`protein_r2.csv` 和 `metrics.json`，应按相同的 feature contract、样本子集与指标口径比较。

## 5. 安装与运行

### 从 GitHub 克隆

```powershell
git clone https://github.com/Chemit797/Go-AI_AIVC.git
Set-Location Go-AI_AIVC
python -m pip install --no-deps --no-build-isolation -e ".[dev]"
```

依赖为 Python 3.10+、numpy、pandas、PyYAML、PyTorch 和 pytest。`--no-deps` 适用于已具备这些依赖的环境；新环境可使用 `python -m pip install -e ".[dev]"` 安装全部依赖。

### 审计、预处理与无训练基线

```powershell
python -m goai_baseline.audit --config configs/baseline.yaml
python -m goai_baseline.preprocess --config configs/baseline.yaml --output runs/preprocess/feature_contract.json
python -m goai_baseline.evaluate --config configs/baseline.yaml --variant b0_mean --output-dir runs/b0_mean
python -m goai_baseline.evaluate --config configs/baseline.yaml --variant b1_matched_control --output-dir runs/b1_matched_control
```

### 训练 MLP 阶梯

```powershell
python -m goai_baseline.train --config configs/baseline.yaml --variant p0_onehot
python -m goai_baseline.train --config configs/baseline.yaml --variant p1_priors
python -m goai_baseline.train --config configs/baseline.yaml --variant p2_crosses
python -m goai_baseline.train --config configs/baseline.yaml --variant p3_time
python -m goai_baseline.train --config configs/baseline.yaml --variant p4_hash
```

或一次运行完整阶梯：

```powershell
.\scripts\run_baseline_ladder.ps1 -IncludeP1ToP4
```

每个 MLP run 写入：`checkpoint.pt`、`training_history.csv`、`metrics.csv`、`protein_r2.csv`、`feature_contract.json`、`feature_summary.json`、`manifest.json`。manifest 记录配置、输入哈希、环境版本、模型参数量与设备。

### 生成并校验提交

```powershell
python -m goai_baseline.predict --config configs/baseline.yaml --run-dir runs\p4_hash-YYYYMMDD-HHMMSS
python -m goai_baseline.submission runs\p4_hash-YYYYMMDD-HHMMSS\prediction.csv --config configs/baseline.yaml --feature-contract runs\p4_hash-YYYYMMDD-HHMMSS\feature_contract.json
```

校验器要求：4,454 行的官方 sample 顺序、冻结的蛋白列顺序、无重复 ID、无额外索引列、无 `NaN/inf`，并且预测尺度为 `log2`。

## 6. 公平性与可复现性

- 所有有目标值的统计量只在训练行拟合；验证标签不参与特征拟合。
- 运行前会校验配置的数据路径与 sample_ID 对齐；Git 不跟踪官方 CSV、模型权重、预测、运行日志、参考 PDF 或临时目录。
- `pytest` 覆盖 sample 对齐、训练集过滤、control 键、mask loss、未知实体 fallback、control 缺失 mask，以及 P0 训练到 test 提交的完整小数据流程。

```powershell
pytest
```

## 7. 下一阶段，不在当前 Baseline 中

当前基线稳定后，再按单变量、可消融的顺序推进：OOF 统计先验 -> control/delta 残差目标 -> 化合物结构与菌株可迁移表征 -> 独立批次分支 -> 蛋白结构先验 -> 多目标 loss 与校准。不要把这些模块混入 P0-P4，否则会失去可比较的基准。
