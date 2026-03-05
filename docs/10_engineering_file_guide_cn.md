# 工程文件技术说明（中文）

## 1. 文档目标

本文件面向当前仓库的开发/复现维护，回答两个问题：

1. 目前工程中各个关键文件分别负责什么。
2. 哪些代码会直接影响物理数值结果（`Mf`、`chi_f`、后验轮廓形状、SNR、收敛性），应优先审计。

说明范围：

- 重点覆盖源码、脚本、配置、文档。
- 不逐条解释 `results/` 下大量中间产物文件（图片、诊断 CSV、样本 NPZ）。

---

## 2. 顶层目录作用

| 路径 | 作用 |
|---|---|
| `src/ringdown/` | 主复现库：波形读入、预处理、QNM 频率、拟合与频域似然。 |
| `scripts/` | Phase2~Phase6 可执行脚本：图表复现、采样、发布图生成、资产清理。 |
| `docs/` | 复现计划、阶段报告、数值审计记录。 |
| `data/` | 输入数据缓存（波形、PSD、注入数据等）。 |
| `results/` | 运行产物（图、诊断、样本、审计报告）。 |
| `深度学习工程SNPE/` | SNPE 子工程（训练/推断/评估，独立配置与输出）。 |
| `README.md` | 项目入口说明。 |
| `requirements.txt` | Python 依赖版本约束。 |

---

## 3. `src/ringdown` 文件说明

| 文件 | 作用 |
|---|---|
| `types.py` | 基础数据结构（`Waveform22`）。 |
| `io.py` | 通用波形 I/O（CSV/NPZ）。 |
| `sxs_io.py` | SXS 数据加载与 remnant 元数据提取。 |
| `preprocess.py` | 峰值对齐、裁剪、重采样、`t0` 网格构造。 |
| `frequencies.py` | Kerr QNM 频率接口（依赖 `qnm` 包），含负自旋对称处理。 |
| `fit.py` | 复线性最小二乘求 `C_n`，可选常数偏置项。 |
| `scan.py` | 固定频率下 `t0` 扫描、`Mf-chi` 网格搜索。 |
| `metrics.py` | 内积、mismatch、remnant 误差指标。 |
| `compare.py` | 波形窗口化、复插值、时间/相位对齐工具。 |
| `fd_likelihood.py` | 频域 PSD 加权似然、SNR、彩色噪声生成核心。 |
| `__init__.py` | 对外 API 汇总导出。 |

---

## 4. `scripts` 文件说明

| 文件 | 作用 |
|---|---|
| `phase2_demo_preprocess.py` | Phase2 预处理示例（对齐、裁剪、重采样）。 |
| `phase3_demo_solver.py` | Phase3 合成数据求解演示。 |
| `phase4_figure1_mismatch_vs_t0.py` | Fig.1 mismatch-vs-`t0` 曲线。 |
| `phase4_figure2_waveform_residual.py` | Fig.2 波形残差图。 |
| `phase4_figure3_mf_chif_landscape.py` | Fig.3 参数地形图。 |
| `phase4_fig1_contract_audit.py` | Fig.1 合同式审计（阈值检查、报告输出）。 |
| `phase5_sxs_error_distribution.py` | 批量 SXS 误差分布统计。 |
| `phase6_figure10_posterior.py` | 早期 Fig.10 后验脚本（包含 dynesty 方案）。 |
| `phase6_fig10_minimal_dynesty.py` | 最小化 dynesty 管线。 |
| `phase6_fig10_kombine_full.py` | kombine/emcee 全图版本（现已加入严格门禁与口径修正）。 |
| `phase6_n3_kombine_emcee_repro.py` | N=3 专项 kombine+emcee 脚本。 |
| `phase6_fig10_emcee_full_strict.py` | **当前严格门禁版主脚本**（emcee 主采样，N=3 双链一致性）。 |
| `phase6_fig10_publication_plot.py` | 发表级平滑图后处理（KDE，可做一致性门禁）。 |
| `cleanup_fig10_assets.py` | Fig.10 资产清理与保留策略执行。 |
| `github_autosync.ps1` | 自动 git 提交推送辅助脚本。 |

---

## 5. `docs` 文件说明

| 文件 | 作用 |
|---|---|
| `00_reproduction_plan.md` | 总体复现路线图与阶段目标。 |
| `01_phase1_theory_equations_baseline.md` | 理论基线与公式约束。 |
| `02_phase2_data_preprocessing.md` | 预处理阶段记录。 |
| `03_phase3_core_solver.md` | 求解器阶段记录。 |
| `04_phase4_figure1_pipeline.md` | Fig.1 管线文档。 |
| `05_phase4_fig2_fig3_pipeline.md` | Fig.2/Fig.3 管线文档。 |
| `06_fig2_reaudit.md` | Fig.2 复审记录。 |
| `07_numerical_stability_controls.md` | 数值稳定性控制策略。 |
| `08_phase5_sxs_distribution.md` | SXS 统计阶段说明。 |
| `09_fig1_contract_audit_report_20260225.md` | Fig.1 合同式审计输出。 |
| `10_engineering_file_guide_cn.md` | 本文档（工程文件与数值敏感代码索引）。 |

---

## 6. SNPE 子工程文件说明（`深度学习工程SNPE/ringdown_fig10_snpe`）

### 6.1 配置与入口

| 文件 | 作用 |
|---|---|
| `configs/fig10.yaml` | Fig.10 物理设定与先验配置。 |
| `configs/snpe.yaml` | SNPE 训练轮次、网络与输出路径配置。 |
| `configs/mcmc_smoke.yaml` | MCMC smoke 配置。 |
| `README.md` | 子工程使用说明。 |

### 6.2 `src` 代码

| 文件 | 作用 |
|---|---|
| `config_io.py` | YAML 读取与路径解析。 |
| `qnm_kerr.py` | QNM 插值器（训练阶段高频调用）。 |
| `ringdown_eq1.py` | Eq.(1) 风格 ringdown 正向模型（plus 通道）。 |
| `noise.py` | PSD 构建与彩色噪声生成。 |
| `units_scaling.py` | 单位换算常量。 |
| `peak_alignment.py` | 峰值对齐检查与可视化。 |
| `phase_a_build_observation.py` | 生成观测注入数据。 |
| `mcmc_smoke.py` | 传统采样 smoke 基线。 |
| `snpe_train.py` | SNPE 主训练脚本。 |
| `snpe_infer_plot.py` | SNPE 后验绘图（N=0..3）。 |
| `summarize.py` | 特征构造与汇总工具。 |
| `sxs_io.py` | 子工程内部 SXS 读入。 |
| `npe_coldstart_round3_try1.py` | 冷启动实验脚本（试验分支）。 |
| `npe_coldstart_round3_try2_dynamic_noise.py` | 冷启动+动态噪声实验分支。 |
| `eval/phase_b_assess.py` | 阶段评估脚本。 |

---

## 7. 物理数值敏感代码（优先审计）

下面这些函数/代码块对结果最敏感，建议先读：

| 路径 | 关键点 | 影响 |
|---|---|---|
| `src/ringdown/fd_likelihood.py` | `one_sided_inner_product`、`draw_colored_noise_rfft`、`FrequencyDomainRingdownLikelihood.log_likelihood` | 决定 PSD 加权似然曲率与 SNR 口径。 |
| `src/ringdown/frequencies.py` | `kerr_qnm_omega_lmn` | `Mf, chi_f -> omega_n` 映射是否正确。 |
| `src/ringdown/preprocess.py` | `peak_time_from_strain`、`build_start_time_grid` | `t0` 定义与扫描范围是否偏移。 |
| `scripts/phase6_fig10_emcee_full_strict.py` | `detector_strain_from_mode22`、`t_hpeak` 平移、SNR 重标定、门禁检查 | 直接影响 Fig.10 是否“看起来合理且统计健康”。 |
| `scripts/phase6_fig10_publication_plot.py` | KDE 平滑 + 一致性检查 | 只影响展示，不应反向污染物理推断。 |
| `深度学习工程SNPE/.../ringdown_eq1.py` | Eq.(1) 波形生成 | 训练数据模拟器的物理正确性。 |
| `深度学习工程SNPE/.../qnm_kerr.py` | QNM 插值与单位换算 | SNPE 中参数到频率映射稳定性。 |
| `深度学习工程SNPE/.../snpe_train.py` | 参数采样、nuisance 边缘化、噪声注入、轮次 proposal 更新 | 决定 SNPE 学到的后验是否与目标任务一致。 |
| `深度学习工程SNPE/.../snpe_infer_plot.py` | credible contour 阈值算法 | 影响后验图像解释口径。 |

---

## 8. 建议阅读顺序（新成员上手）

1. `docs/00_reproduction_plan.md`
2. `src/ringdown/fd_likelihood.py`
3. `scripts/phase6_fig10_emcee_full_strict.py`
4. `scripts/phase6_fig10_publication_plot.py`
5. SNPE 子工程：`ringdown_eq1.py -> qnm_kerr.py -> snpe_train.py -> snpe_infer_plot.py`

---

## 9. 维护约定

1. 发布图前必须先看诊断 CSV，确认 `acceptance` 与 `tau` 门禁通过。  
2. 平滑脚本只做可视化后处理，不允许替代采样质量控制。  
3. 物理口径修改（`t_h-peak`、PSD、先验边界）必须同步写入文档与配置。  
