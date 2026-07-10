# 光谱 Coding Looping

## 任务描述

科学家在材料分析、环境检测、和样品鉴定中最核心的问题是，未知样本由哪些成分构成，各成分大致比例是多少，哪些特征峰支持该判断，以及当前结果是否合理

我们设计这个任务也是希望 agent 能够像实验科学家一样，围绕一条不干净的未知光谱反复运行代码，检查中间状态、调整分析策略，最后完成成分识别、比例估计和可信性诊断。

## 数据来源
pip install ramaspy
使用数据集 RamanSPy 来加载 RRUFF 这是一个单矿物质拉曼光谱数据集，挑出两三种成分进行混合，构造合成的带噪声光谱例子

## Pipeline

去噪声 归一化 基线校正 去尖峰

谱图库的匹配

候选成分筛选

多成分线性接触混杂，非负最小二乘法拟合

残差分析

判断：是否楼成分，是否匹配有无，是否误差难以容忍

修改 config 重新匹配

---

# 实现说明（已完成并验证）

上面的粗糙计划已落地为一个**可执行、可复现、可量化**的闭环 pipeline。

## 快速开始

分两步：先**离线合成算例数据集**，再对某个算例跑**盲目分析循环**。

```bash
pip install ramanspy            # 连带 scipy / scikit-learn / pybaselines / cvxopt / matplotlib

# 1) 生成算例（合成脏光谱 + 隐藏真值），落盘到 cases/case1, case2, ...
python make_cases.py --n 5 --seed 0

# 2) 对某个算例跑分析（循环全程看不到真值）
python run.py --case cases/case1 --max-components 8
# 产物: outputs/run_case1/{report.md, iterations.jsonl, figures/*.png}
```

`make_cases.py` 参数: `--n`（算例数）`--seed`（第 i 个算例用 seed+i）`--out`（输出根目录，默认 `cases`）
`--n-components`（2/3，默认随机）`--max-minerals`（谱库矿物数）`--noise`（噪声水平）。

`run.py` 参数: `--case`（算例文件夹路径，必填）`--budget`（循环预算轮数）`--seed`（搜索随机种子）`--max-minerals`。

## 算例目录结构（`cases/caseN/`）

| 文件           | 内容                                                                     |
| -------------- | ------------------------------------------------------------------------ |
| `spectrum.csv` | 两列 `raman_shift,intensity`——**唯一**允许分析读取的可观测输入           |
| `truth.json`   | `{true_names, true_fractions, meta}`——隐藏答案，只在循环结束后评估用一次 |
| `input.png`    | 脏光谱可视化                                                             |

## 代码结构（`src/`）

| 模块            | 职责                                                                                                                                                                                   |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`     | 默认 config + 配置搜索空间（`SEARCH_SPACE`）                                                                                                                                           |
| `data.py`       | RRUFF 下载/缓存、**容错解析**（ramanspy 自带 loader 有 bug）、每矿物取代表谱、**重采样到统一波数网格**、建参考矩阵 A                                                                   |
| `synth.py`      | 从谱库合成带噪混合谱（基线漂移+高斯噪声+宇宙射线尖峰+强度缩放），带已知真值；强制最小成分占比避免退化为单成分；**`save_case`/`load_spectrum`/`load_truth` 将可观测谱与真值分文件落盘** |
| `preprocess.py` | 去尖峰→去噪→基线校正→归一化，全部 config 驱动                                                                                                                                          |
| `matching.py`   | 候选筛选：**greedy/OMP 残差匹配**（默认）或全局 top-K                                                                                                                                  |
| `unmix.py`      | 候选子矩阵 NNLS 解混，低占比剔除+重拟合                                                                                                                                                |
| `diagnose.py`   | 残差分析、支持峰提取、可信度诊断、决策提示                                                                                                                                             |
| `metrics.py`    | 相似度（pearson/cosine/SID/RMSE）+ 真值评估（成分 P/R/F1、比例 MAE）                                                                                                                   |
| `loop.py`       | 外层配置搜索：**尺度无关相对残差**为盲目主目标 + 简约罚，**全程不接触真值**；`evaluate()` 仅在循环结束后被调用一次                                                                     |
| `make_cases.py` | **数据集生成入口**：建库→批量合成→落盘 `cases/caseN/`                                                                                                                                  |
| `run.py`        | 分析入口：建库→载入算例(仅可观测谱)→跑盲目循环→循环后载入真值评估一次→出报告与图                                                                                                       |

## 相对原计划的关键改进（均由实测驱动）

1. **容错 RRUFF 解析器** —— ramanspy `rp.datasets.rruff` 遇非数据行崩溃，自建解析。
2. **统一波数网格重采样** —— 各谱波数轴不同，NNLS 前必须对齐（原计划缺失）。
3. **尺度无关的循环目标** —— 直接比较残差 RMSE 会偏向让数值变小的归一化方式；改用相对残差 ‖resid‖/‖target‖，使不同 config 可公平比较。
4. **greedy/OMP 候选筛选** —— 全局 top-K 会被最强成分的"同类矿物"占满、埋没弱成分；改为匹配最强→扣除→在残差上再匹配，弱成分（如沸石 Clinoptilolite）得以被找回。
5. **可信度诊断以稳健连续量为准** —— 残差残留峰易受预处理伪影干扰，故可信度主要看拟合相关+解释能量。
6. **数据集与分析解耦** —— 合成算例由 `make_cases.py` 单独落盘为 `cases/caseN/`（`spectrum.csv` + `truth.json`），可复用、可检视、可分享。
7. **循环严格盲目** —— `search()` 只接收可观测谱，循环内不再计算任何真值指标；`component_prf` / `fraction_mae` 仅在循环结束后由 `evaluate()` 调用一次，真实还原科学家不知道答案时的分析流程。

## 验证结果（8 个随机种子，120 矿物谱库）

- 平均成分识别 **F1 = 0.964**，平均比例 **MAE = 0.032**，6/8 完美识别。
- 少数不完美算例（多识别出 1 个假阳性）恰好被 pipeline 自身诊断标为
  moderate/low 可信度，完美算例标为 high —— **盲目诊断与真实精度一致**。

示例产物见 `outputs/run_seed0/` 与 `outputs/run_seed3/`。
