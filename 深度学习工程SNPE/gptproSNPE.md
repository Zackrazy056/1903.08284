# 上 SNPE（先用 MCMC 做少量 sanity check）的完整实验工程设计 + 推进方案

目标：最忠实复现 1903.08284 的 Fig.10（SXS:BBH:0305 注入、aLIGO design 高斯噪声、Δt0=0、N=0..3 的 \((M_f,\chi_f)\) 后验对比）。Fig.10 的定义与口径（Δt0=0、N=0..3、幅度相位边缘化、真值点）需要严格对齐。

---

## 0) 复现 Fig.10 的不可变规格（写进 config，任何人都能复跑）

### 0.1 注入/观测设定（NR injection）

- NR：SXS:BBH:0305
- 仅用 \((\ell=m=2)\) 模，注入 aLIGO design 高斯噪声
- 几何：face-on，无倾角；选天区使探测器对 plus 最优、对 cross 为零（等价设置 \((F_+=1,F_\times=0)\)）
- 缩放：总初始质量 72 \(M_\odot\)（detector frame）+ 距离 400 Mpc
- post-peak optimal SNR \(\sim 42.3\)
  - SNR 定义：仅积分 \((f>154.68)\) Hz（峰值瞬时频率）

### 0.2 推断模型（teacher / student 都一致）

- 模板：Eq.(1) 的 Kerr QNM overtone ringdown，\((\ell=m=2)\)，\((N=0,1,2,3)\)
- 起始：\(\Delta t_0=(t_0-t_{h\text{-}peak}=0)\)，即从 detector strain 的峰值开始
- 真值点：\((M_f=68.5M_\odot,\chi_f=0.69)\)（图中虚线交点）

### 0.3 先验（必须逐字照搬）

- \(M_f\in[10,100]M_\odot\)，\(\chi_f\in[0,1]\)
- phases \(\in[0,2\pi]\)
- amplitudes（在 \(t=t_{h\text{-}peak}\) 测量）\(\in[0.01,250]h_{\rm peak}\)，\(h_{\rm peak}=2\times10^{-21}\)
- 论文用 kombine + emcee（用 emcee 做 sanity check 完全匹配其验证角色）
- 最高只用到 \(N=3\)（\(N\ge3\) 的差异在该 SNR 下不可测；\(N=4\) 只引入退化）

---

## 1) 工程目录与可审计工件（Artifacts）

```text
ringdown_fig10_snpe/
  configs/
    fig10.yaml          # 上面的全部规格：M,D,PSD,fmin,t0,T,N,priors     （理基准配置文件。必须包含引力波源的总体质量（$72 M_\odot$）、光度距离（$400 \text{ Mpc}$）、时间截断点 $t_0$ 的定义标准、泛音阶数 $N$ 的范围、以及所有十维参数（含振幅、相位）的均匀先验边界函数定义。 ）
    mcmc_smoke.yaml     # MCMC sanity 参数  传统基准采样器的运行参数。包含预热步数（Burn-in）、提议分布（Proposal distribution）的方差设定以及链的稀疏化（Thinning）尺度。 
    
    
     （walkers/steps/thin）   
    snpe.yaml           # SNPE 训练：rounds、模拟预算、网络结构神经网络架构规范。定义一维残差网络（1D-ResNet）的层数与膨胀率（Dilation rate）、掩码自回归流（MAF）或神经样条流（NSF）的变换层数（Flow steps）与隐藏层神经元数量，以及前向模拟器的总预算（Simulation budget，如 $10^6$ 次生成）。


  data/   #用于隔离静态的观测数据与仪器特性数据。
    sxs_cache/          # 下载缓存  存放从 Simulating eXtreme Spacetimes (SXS) 星表下载的原始数值相对论（NR）波形 HDF5 文件（特指 SXS:BBH:0305）。

    injection/          # 观测数据 d_obs（固定一个noise realization）固化用于最终推断的单一观测数据 $\mathbf{d}_{\text{obs}}$。该数据必须由缩放后的 SXS 波形注入到特定随机数种子生成的 Advanced LIGO 有色噪声中构成，以确保 MCMC 与 NPE 针对同一似然面进行评估。

    psd/                # aLIGO design PSD + 频率网格
  
  存储 Advanced LIGO 的设计灵敏度单边功率谱密度（PSD）数组及对应的离散频率网格，用于后续内积计算与噪声白化。
  
  
  
  src/   核心计算栈   正向物理模拟、特征提取与统计推断引擎。
    sxs_io.py           # 读取 SXS:BBH:0305 的 (2,2) 模  波形解析器。利用球面调和函数提取 NR 数据的特定模式（主导模式 $\ell=m=2$），并输出复数应变序列。

    units_scaling.py    # 72Msun/400Mpc 缩放到 detector strain   物理量纲转换器。严格执行从无量纲的几何单位制（$G=c=1$）到国际单位制（SI）及探测器应变振幅的线性映射。

    peak_alignment.py   # 找 t_h-peak，定义 Δt0=0 的 t0

峰值对齐模块。应用样条插值寻根算法（Root-finding algorithm）精确测定复数应变模长 $|h(t)|$ 的最大值时间 $t_{\text{h-peak}}$，根除引发高阶泛音误差的时移问题（Time-shift problem）。

    noise.py            # 生成 design PSD 高斯噪声；whiten; SNR 检验
随机过程与白化算子。依据 PSD 生成平稳高斯色噪声，执行频域信号的白化操作（除以 $\sqrt{S_n(f)}$），并包含严格的信噪比（SNR = 42.3）强制缩放断言函数。

    qnm_kerr.py         # ω_22n(Mf,χf) 计算（建议用 qnm 包）

微扰理论接口。输入 $M_f$ 与 $\chi_f$，调用广义相对论特征值求解器（如 qnm 库），输出各阶泛音的复频率 $\omega_{22n}$。

    ringdown_eq1.py     # Eq.(1) 生成 h(t|θ,N)
正向生成器（Forward Simulator）。执行方程 Eq. (1) 的批量计算。为满足 NPE 的吞吐量需求，该模块必须实现包含时间截断效应（Heaviside 阶跃函数）的解析连续傅里叶变换，直接输出频域波形阵列。


    summarize.py        # 数据压缩（推荐频域压缩特征）

    特征降维映射。定义 1D-ResNet 架构，将白化后的高维频域波形张量映射为低维特征向量（Context Vector），以消除高频白噪声的冗余自由度。

    mcmc_smoke.py       # emcee 快速短链：验证似然/先验/峰值对齐
 基准推断器。构建严格的解析频域 PSD 加权似然函数，运行短链贝叶斯采样，用于快速验证物理生成器与似然面拓扑的正确性。   
    snpe_train.py       # sbi 的 SNPE（只输出 Mf, χf）
SBI 训练主循环。实例化密度估计器（Density Estimator），最小化负对数似然（Negative Log-Likelihood），学习联合后验分布 $p(M_f, \chi_f | \mathbf{d})$，隐式边缘化所有振幅与相位参数。

    snpe_infer_plot.py  # 生成 Fig10_NPE 与对比图
后处理与可视化。在完成训练的网络中输入 $\mathbf{d}_{\text{obs}}$，通过单次前向传播生成数万个等权重的后验样本，并绘制标准的 1D/2D 边缘分布角图。

    eval/ 统计检验协议  执行贝叶斯推断的严格有效性验证。
      posterior_compare.py  # NPE vs MCMC：contour IoU/距离等  计算 NPE 输出后验与 MCMC 基准后验之间的詹森-香农散度（Jensen-Shannon Divergence），定量评估两者的分布重合度（IoU）。


      sbc.py                # 最小 SBC（校准）测试
      
 基于模拟的校准（Simulation-Based Calibration）。在大规模测试集上生成概率-概率图（P-P Plot），验证神经网络输出后验的统计无偏性与置信度覆盖准确率。
 
 
  outputs/
    fig10_mcmc_smoke.png
    fig10_snpe.png
    diagnostics/
      snr_report.json
      mcmc_traceplots.pdf
      sbc_rank_hist.png
```

---

## 2) 核心理念：SNPE 学的是边缘后验，不用学全维后验

Fig.10 的口径是：\((M_f,\chi_f)\) 的 90% contour，并且 amplitudes & phases 已边缘化。
因此 SNPE 的目标直接设为：

\[
q_\phi(M_f,\chi_f \mid x, N)\approx p(M_f,\chi_f\mid x, N)
\]

而在模拟数据时，把 \(\{A_n,\phi_n\}\) 按论文先验随机化并积分掉（nuisance marginalization）。这会把学习难度从高维压到二维，同时仍然严格对齐论文输出口径。

> 实施建议：对每个 \(N\) 单独训练一个 SNPE（四个模型），因为 Fig.10 本来就是按 \(N\) 分别给出后验。

---

## 3) Phase A：构建观测数据 d_obs（NR 注入）并做两项硬校验

### A1) t0（Δt0=0）的峰值对齐

论文把起始时间参数化为 \(\Delta t_0=t_0-t_{h\text{-}peak}\)，并给出
\(t_{h\text{-}peak}\approx t_{peak}-0.48\text{ ms}\approx t_{peak}-1.3M\) 的关系。
代码里必须明确区分：

- \(t_{peak}\)：复应变 \(|h|\) 的峰
- \(t_{h\text{-}peak}\)：detector strain 峰（Fig.10 用它做 Δt0=0）

验收工件：`outputs/diagnostics/peak_alignment.png`（两种峰值标线 + 偏移量报告）

### A2) post-peak SNR 复核

论文：同设定下 post-peak optimal SNR \(\sim42.3\)。
并注明 SNR 定义在 \(f>154.68\) Hz。

验收工件：`snr_report.json`

- 字段：`snr_postpeak`, `fmin=154.68`, `Mf_total=72`, `D=400Mpc`, `psd=aligo_design`

> 这一步不通过，后面所有推断形状都不可信（缩放/单位/PSD 任一项错都会把 Fig.10 改形）。

---

## 4) Phase B：MCMC sanity check（短链，不追求高精度）

目的不是复现 Fig.10，而是证明 forward model + likelihood + priors + t0 处理逻辑正确。论文用 MCMC（kombine/emcee）采样，这正好给出最小教师检查点。

### B1) 只跑一个 N（建议 N=1 或 N=2）+ 很短链

- 参数：\((M_f,\chi_f,A_0,\phi_0,...,A_N,\phi_N)\)
- 先验照搬论文（见 0.3）
- walkers：≥ 4×dim；steps：几千级别；多条链对比
- 输出：粗略 corner（只看 \((M_f,\chi_f)\)）+ traceplots + acceptance fraction

通过标准（最低）：

- 链能跑、无数值崩溃；
- 后验不贴满先验边界（尤其 Mf、χf）；
- 真值点 \((68.5,0.69)\) 落在合理概率区域附近（不要求完全匹配 Fig.10）。

验收工件：

- `outputs/fig10_mcmc_smoke.png`（只需 1 个 N 的 contour）
- `outputs/diagnostics/mcmc_traceplots.pdf`

---

## 5) Phase C：SNPE 主流程（sequential + 局部摊销），复现 Fig.10

下面是完整训练协议，可直接执行。

### C0) 四个独立任务（对应 Fig.10 的 N=0..3）

对每个 \(N\in\{0,1,2,3\}\) 单独训练一个 SNPE 模型，最终输出四条 contour。Fig.10 就是以不同 N 对比后验收缩与偏差纠正趋势。

### C1) Simulator 定义（用于 SNPE 生成训练样本）

参数采样：

- 目标参数（network 输出）：\(\vartheta=(M_f,\chi_f)\)
- nuisance（仅用于模拟）：\(\nu=\{A_n,\phi_n\}_{n=0}^{N}\)

采样规则完全照论文：

- \(M_f\sim U(10,100)\)，\(\chi_f\sim U(0,1)\)
- \(\phi_n\sim U(0,2\pi)\)
- \(A_n(t_{h\text{-}peak})\sim U(0.01,250)\,h_{peak}\)，\(h_{peak}=2\times10^{-21}\)

模板生成（Eq.1）：

给定 \((M_f,\chi_f)\) 计算 Kerr QNM 复频率 \(\omega_{22n}(M_f,\chi_f)\)，再生成

\[
h(t)=\sum_{n=0}^{N} C_{22n}\exp[-i\omega_{22n}(t-t_0)],\quad t\ge t_0
\]

（Eq.1 形式同 Fig.10 的推断模型）

加噪声与预处理：

- 噪声：aLIGO design PSD 生成的高斯噪声（与 Phase A 同实现）
- 截取窗口：从 \(t_0=t_{h\text{-}peak}\) 起（Δt0=0）
- 白化：建议对每条模拟数据做 whitening（同 PSD）

### C2) Summary / embedding（推荐频域压缩，更稳更省样本）

为高效复现 Fig.10（不是做端到端波形理解），建议用固定频点的白化频域特征：

- 对窗口数据做 FFT → 白化 → 取 \(K\) 个频点（覆盖 ringdown 主信息）
- 特征向量：

\[
x=[\Re\tilde d_w(f_1),\Im\tilde d_w(f_1),...,\Re\tilde d_w(f_K),\Im\tilde d_w(f_K)]
\]

这样网络学习更接近识别 QNM 频率/阻尼与 \((M_f,\chi_f)\) 的映射，训练更稳定。

> 端到端时域 CNN 可作为第二阶段；先把 Fig.10 做出来优先级更高。

### C3) SNPE 训练协议（3 轮足够，且模拟预算可控）

Round 0（粗探索）：

- 从先验抽 \((M_f,\chi_f)\) + nuisance → 生成 \((x,\vartheta)\)
- 模拟数：每个 N 先做 2–5×10⁴
- 训练一个 NPE（density estimator：normalizing flow），得到 \(q_0(\vartheta\mid x)\)

Round 1（集中后验高密度区）：

- 用 \(q_0(\vartheta\mid x_{obs})\) 产生提议分布 \(r_1(\vartheta)\)（可截断到 99% HPD）
- 从 \(r_1\) 抽 \(\vartheta\)，补齐 nuisance 生成新样本
- 模拟数：再做 2–5×10⁴
- 训练/微调得到 \(q_1\)

Round 2（收敛与光滑 contour）：

- 同样做一次更窄提议 \(r_2\)，再采样 1–3×10⁴
- 得到最终 \(q_2\)

> 关键收益：不用 10⁶ 样本、也不必等待高维 MCMC 收敛，同时保持条件后验估计的统计语义。

### C4) 输出与 Fig.10 复现（SNPE 版）

对同一个观测数据 \(x_{obs}\)，分别用 \(q_N(\vartheta\mid x_{obs})\)（N=0..3）采样后验并作图：

- 主 panel：四条 90% contour
- top/right：1D marginals
- 虚线十字：真值 \((M_f=68.5,\chi_f=0.69)\)
- 图注写清：Δt0=0；N=0..3；幅度相位已边缘化

验收标准（论文一致的现象级标准）：

- \(N=0\) 在 Δt0=0 时明显偏离真值；随 \(N\) 增大逐步靠近；\(N=3\) 最好。
- 论文指出 \(N=3\) 使真值落在 top 40% credible region 内。

---

## 6) 质量保障：必须做两类评估（否则“像 Fig.10”也可能是错的）

### E1) 与 MCMC 的局部一致性

用 Phase B 的 MCMC smoke（即便精度一般）做 sanity overlay：

- NPE contour 与 MCMC contour 的重叠度（IoU/面积比）
- 1D marginals 的 Wasserstein / KS（粗评估即可）

### E2) 最小 SBC（simulation-based calibration）

固定一个 N（建议 N=1 或 N=2），重复：

- 从先验采样真值 \(\vartheta^*\)，生成模拟观测 \(x^*\)
- 用训练好的 \(q(\vartheta\mid x^*)\) 计算 rank statistics
- rank 直方图接近均匀，说明校准未崩

> 这是后续做 \(\alpha\) 参数推断时避免“虚假偏离”的底层保障。

---

## 7) 三个高风险坑（都与论文口径强绑定）

1. Δt0=0 的峰值定义错：Fig.10 明确用 \(t_{h\text{-}peak}\)（detector strain 峰），不是随手取的 NR \(|h|\) 峰。
2. 振幅先验测量时刻不一致：论文幅度在 \(t=t_{h\text{-}peak}\) 定义并设范围。
3. N=4 之后的退化：该 SNR 下 \(N\ge3\) 差异不可测，N=4 会引入退化；不要用 N=4 的不稳定否定前面实现。

---

## 8) 最小落地清单（下一步就做这些）

1. 写 `configs/fig10.yaml`，把 0) 的规格固化（质量/距离/PSD/fmin/priors/Δt0=0/N=0..3）。
2. 完成 Phase A：生成 `d_obs` + 输出 `snr_report.json`（SNR 接近 42.3）。
3. 做 Phase B：只跑一个 N 的 emcee 短链 smoke，确认 likelihood 与 priors 正常。
4. 上 Phase C：先跑 N=0 的 3 轮 SNPE 得到 contour，再复制到 N=1,2,3，最后拼成 Fig.10。

---

如果你愿意，下一条我可以补充：

- `fig10.yaml` 的字段表（每个字段单位与默认值）
- SNPE 具体网络建议（embedding 维度、flow 类型、batch size、early stopping、学习率范围）

你只要告诉我运行平台（本地 GPU / Colab / 服务器）和偏好的输入表示：

- A) whitened time series
- B) whitened FFT features



## 第一轮pro交互

According to a document from **9 Jan 2020**, 你要“最忠实复现”的目标可以被精确定义为：用 **SXS:BBH:0305** 的 **NR 注入**，在**单台 Advanced LIGO 设计灵敏度**的高斯彩噪中做 **ringdown-only** 的贝叶斯推断，采用文中 **Eq.(1) 的 Kerr QNM overtone ringdown 模板**，在 **起始时刻取峰值处（Δt0=0）** 时，对不同 overtone 截断阶数 **N=0,1,2,3** 得到 **(Mf, χf)** 的后验，并画出与 Fig.10 一致的 **90% credible contours + 1D marginals**，其中真值为 **Mf=68.5 M⊙、χf=0.69**。

下面我给你一个**工程化、可执行、可审计**的完整复现方案（含：基准 MCMC + 可扩展到 NPE 的深度学习版本），并把论文里必须对齐的实验设定逐项固化成“硬约束”。

---

## 0. 复现 Fig.10 的“硬约束规格”清单（必须对齐）

### 0.1 注入与观测设置（B: NR injection）

* NR 波形：**SXS:BBH:0305**。
* 只注入主导角模：**ℓ=m=2**。
* 源几何：**face-on（无倾角）**，并选取天空位置使探测器对 **plus 极化最优、对 cross 极化为零**。
* 缩放为类似 GW150914：**总初始质量 72 M⊙（探测器参考系）**、**距离 400 Mpc**。
* 噪声：**单台 Advanced LIGO 设计构型**的**高斯噪声**。
* 校验量：post-peak optimal SNR **≈ 42.3**（论文脚注说明此 SNR 定义在 **154.68 Hz 以上**）。

> 解释：这组设定把 extrinsic 全“冻结”，让难点集中在 **ringdown 模板是否能在峰值处拟合**以及 **N 对偏差/方差的影响**，正是 Fig.10 想展示的核心。

### 0.2 模板与参数

* ringdown 模板：文中 Eq.(1) 的 damped sinusoids 叠加
  [
  h_{\ell m}^N(t)=\sum_{n=0}^N C_{\ell mn}, e^{-i,\omega_{\ell mn}(M_f,\chi_f),(t-t_0)},\quad t\ge t_0
  ]

* 分析使用数据起点：**Δt0 = t0 − t_{h-peak} = 0**（即从注入应变峰值开始）。
* 真值（用于图中十字虚线交点）：**Mf=68.5 M⊙, χf=0.69**。
* 最高 N：论文的推断只做到 **N=3**，理由是该 SNR 下 **N≥3 的差异不可分辨**，N=4 只会引入退化。

### 0.3 先验（这点对 Fig.10 形状影响很大）

论文明确给出了推断先验与做法（你需要原样复刻）：

* 对所有参数用**均匀先验**；
* (M_f \in [10,100],M_\odot)，(\chi_f\in[0,1])；
* 相位 (\phi\in[0,2\pi]) 均匀；
* 幅度在 (t=t_{h-peak}) 处限制为 ([0.01, 250],h_{\rm peak})，且 (h_{\rm peak}=2\times10^{-21})；
* 采样器：kombine 与 emcee（emcee 用于验证）。

---

## 1. 总体工程结构（把“复现”做成可维护管线）

建议你用“配置驱动”的 repo（后续无缝扩展到 α / LISA）：

```
ringdown-fig10-repro/
  environment/          # conda/pip 依赖锁定
  configs/
    fig10.yaml           # 所有硬约束都写进去（质量、距离、PSD、t0等）
    mcmc.yaml
    npe.yaml
  data/
    sxs_cache/           # SXS waveform 缓存
    injections/          # 生成的注入数据（h, n, d）
  src/
    sxs_io.py            # 下载/读取 SXS:BBH:0305
    scaling.py           # M, D 缩放与单位换算
    detector.py          # F+, Fx 投影（这里用 F+=1,Fx=0）
    noise.py             # PSD + 高斯噪声生成 + whiten
    qnm.py               # ω_lmn(Mf, χf) 计算
    ringdown_model.py    # Eq(1) 生成模板
    likelihood.py        # 高斯噪声 logL
    inference_mcmc.py    # emcee 采样
    inference_npe.py     # sbi 的 NPE/SNPE
    plot_fig10.py        # 复现 Fig.10 风格图
  tests/
    test_units.py
    test_snr.py
    test_peak_alignment.py
    test_qnm_freqs.py
    test_mcmc_smoke.py
```

依赖建议（都是“行业常用”、且与你后续 LISA/α 项目兼容）：

* **SXS 数据读取**：`sxs.load` 支持自动下载与缓存
* **QNM 频率**：`qnm`（Kerr QNM，含缓存与插值）
* **MCMC**：`emcee` EnsembleSampler
* **NPE/SNPE（SBI）**：`sbi`

---

## 2. Stage A：先做“无噪声/最小噪声”单元测试（避免一上来就 MCMC 黑箱）

你想“最忠实复现 Fig.10”，真正的难点不是采样器，而是**对齐：t0 定义、单位缩放、QNM 频率、数据窗口**。因此强烈建议在进入 MCMC 前做两类“不可妥协的单测”。

### A1. t_peak 与窗口上限 T 的一致性（来自论文的明确设定）

论文在 mismatch 实验里把内积上限取为 **T = t_peak + 90M**（在 NR 衰减到数值噪声之前）【173:4†1903.08284v2.pdf†L35-L47】。
虽然 Fig.10 的贝叶斯段落没再重复 T，但**你用同一窗口**能最大化一致性与可解释性。

工程做法：

1. 从 SXS 读出复应变 (h(t)=h_+ - i h_\times) 的 (2,2) 模；
2. 找到 **|h| 的峰值时刻** (t_{\rm peak})；
3. 定义分析片段：([t_0, T])，其中 Fig.10 用 (t_0=t_{h\text{-}peak}) 且 Δt0=0【173:0†1903.08284v2.pdf†L54-L56】；你可按文中近似关系把 (t_{h\text{-}peak}) 与 (t_{\rm peak}) 做毫秒级对齐（论文给出 (t_{h}-0.48,\mathrm{ms}\approx t_{\rm peak}-1.3M)）【173:11†1903.08284v2.pdf†L40-L42】。

> 审计点：你一旦把峰值对齐搞错，Fig.10 的“峰值处加入 overtone 可纠偏”这个现象会明显变弱或形态改变。

### A2. SNR 校验（这相当于你的“端到端一致性 checksum”）

你需要复现 **post-peak optimal SNR ≈ 33-L34】。论文脚注进一步指出 SNR 的积分频段定义在 **154.68 Hz 以上**【173:0†1903.08284v2.pdf†L29-L29】。

工程做法（频域定义）：
[
\rho^2 = 4\int_{f_{\min}}^\infty \frac{| \tilde h(f)|68,\mathrm{Hz}
]

* 用你选定的 aLIGO design PSD；
* 用窗口 ([t_0,T]) 的信号（post-peak）；
* 看是否得到 ~42 的量级。

> 如果 SNR 偏差很大：先别调 MCMC；优先排查（i）质量/距离缩放，（ii）h 是否与你实现一致。

---

## 3. Stage B：MCMC 基准复现（这是逐字复刻论文）

对每个 N（0..3），参数向量可写为：
[
\theta = (M_f,\chi_f,{A_n,\phi_n}_{n=0}^N)
]
先验按论文：

* (M_f \sim \mathcal U(10,100)M_\odot), (\chi_f\sim\mathcal U(0,1))【173:11†1903.08284v2.pdf†L44-L45】
* (\phi_n\sim \mathcal U(0,2\pi))【173:11†1903.08284v2.pdf†L45-L46】
* (A_n(t_{h\text{-}peak}) \sim \mathcal U(0.01,250),h_{\rm peak}), (h_{\rm peak}=2\times10^{-21})【173:11†1903.08284v2.pdf†L46-L48】

并且论文强调他们**直接采样**幅度与相位（而不是像某些工作那样解析边缘化）【173:11†1903.08284v2.pdf†L43-L44】。

### B2. 似然（建议用标准高斯噪声频域内积）

你要实现：
-h(\theta),|,d-h(\theta)) + \text{cofty \tilde a(f)\tilde b^*(f)/S_n(f),df)。

> 关键工程选择：你可以把数据截取到 ([t_0,T]) 后再 FFT。即使存在矩形窗导致的频谱展宽，只要数据和模板一### B3. 采样器设置（emcee 作为主实现）
> 论文用 kombine + e49-L53】；你可直接用 emcee 做主版本（kombine 不是必须）。

* walkers：建议 ≥ 4×dim（dim=2+2(N+1)）
* burn-in：用自相关时间估计；或用多段检查（trace + ESS）
* 输出：保留后验样本的 (M_f,\chi_f) 边缘

### B4. Fig.10 的“图像验收标准”

你输出图必须包含：

* 主 panel：四条 90% credible contours（N=0..3），且都在 **Δt0=0 ms** 同一起点下对比【173:0†1903.08284v2.pdf†L45-L56】
* 上/右：Mf 与 χf 的 1D posterior
* 用虚线十字标出真†1903.08284v2.pdf†L57-L58】

此外，论文对现象的定性判断你可以用作“语义验收”：

* N=0 在峰值处会显著偏离真值；
* 随 N 增大，后验区域向真值移动并收缩，N=3 接近最好【173:0†1903.08284v2.pdf†L45-L58】；
* N≥3 在该 SNR 下基本不可分辨，N=4 不改善且引入退化【173:13†1903.08284v2.pdf†L31-L34】。

---

## 4. Stage C：用 NPE 复现 Fig.10（把 如果你只想复现一篇论文的一张图**，完全没必要把 NPE 做成“全先验空间的通用摊销推断器”。你应该做的是：**围绕单个观测 d_obs 的高密度区，从而大幅减少训练量，同时保持统计一致性。

### C1. 关键建模决策：只输出 (Mf, χf)，把 (A_n, φ_n) 当 nuisa练时你仍然按论文先验采样 **全参数**

(\theta=(  [
q_\varphi(M_f,\chi_f\mid x)
]
其中 (x) 是模拟得到的数据（时间序列或其压缩特征）。

* 由于训练数据中 nuisance 是按先验采样的，网络学习到的是严格意义上的**边缘后验**：
  [
  p(M_f,\chi_f\mid x)=\int p(M_f,\chi_f,A,\phi\mid x),dA,d\phi
  ]
  这与 Fig.10 的“amplitudes and phases marginalized over”完全对齐【173:0†1903.08284v2.pdf†L56-L57】。

> 这一步非常关键：它把深度学习任务从“高维后验拟合”降到“二维后验拟合”，并且仍然忠实保留论文的统计含义。

### C2. 数据表示（x 怎么喂给网络）

给你两个“都可行，但复杂度不同”的选项：

**选项 1：喂 whitened time series（最直观、最贴近端到端）**

* 截取 ([t_0,T]) 的单通道应变；
* whiten 后得到向量 (x\in\mathbb R^L)；
* embedding：1D CNN / dilated CNN；
* density estimator：normalizing flow（MAF/NSF）。

**选项 2：喂频域压缩特征（更稳、更省参数）**  down 相关频段选取 K 个频点（例如围绕 200–400 Hz 或自适应围绕预测的 f_{220}）；

* 用 ([\Re \tilde d(f_k),\Im \tilde d(f_k)]) 拼成特征。

> 对“复现 Fig.10”这个小目标，我更推荐选项 2：网络更容易学到“频率/阻尼随 (Mf,χf) 变化”的结构，且更少被时间对齐误差干扰。

### C3. 推断算法：sbi 的 NPE / SNPE

`sbi` 明确支持 amortized NPE 与 sequential SNPE。
你要做的是 **sequential（针对单个 d_obs）**：

* Round 0：从先验采样 (Mf,χf)，同时采样 nuisance (A,φ)，生成数据 x
* 训练 q_\varphi(Mf,χf|x)
* 用 q_\varphi 在 d_obs 上得到近似后验
* Round 1/2：从该后验（或其截断版本）再采样生成更聚焦数据，继续训练

这能非常有效地把模拟集中在 Fig.10 轮廓附近。

### C4. NPE 的“可审计验收”：你至少要做两件事

1. **与 MCMC posterior 对比**：

   * overlay 2D contours（NPE vs MCMC）
   * 比较 1D marginals（KS 或 Wasserstein 距离）
2. **SBC（simulation-based calibration）**：
   你可以固定 N 与 t0 设定，随机抽若干注入（从先验抽真值），用训练好的 NPE 推断，检查 rank statistics 是否近似均匀。
   这一步是你将来做 **“可校准、端到端 LISA 管线”** 的统计底座。

---

## 5. 你最可能踩的坑（以及快速定位方式）

### 5.1 t0 / peak 对齐不一致

* 论文对起始时刻参数化是 (\Delta t_0=t_0-t_{h\text{-}peak})【173:11†1903.08284v2.pdf†L40-L42】；Fig.10 固定 Δt0=0【173:0†1903.08284v2.pdf†L54-L56】。
* 你必须明确：你代码里的 “peak” 到底是 (|h|) 的峰，还是 detector strain 的峰；二者在文中差约 0.48 ms【173:11†1903.08284v2.pdf†L40-L42】。

**诊断**：把你生成的注入（无噪声）与 NR 原始 (2,2) 在峰值附近叠加，看相位是否一致；SNR 校验是否正确。

### 5.2 单位换算/缩放错一个因子，后面全错

这类错误最隐蔽，但 SNR check 能一票否决。

**诊断**：只要 post-peak SNR 明显偏离 ~42【173:11†1903.08284v2.pdf†L33-L34】，就先别谈 Fig.10 的轮廓像不像。

### 5.3 N 增加导致的退化与先验主导

论文明确说 N=4 在该 SNR 下无提升且引入退化【173:13†1903.08284v2.pdf†L31-L3是统计事实（不可辨识）。

**=3 是否稳定。

---

## 6. 你接下来怎么把它无缝接到“α 参数 + LISA ringdown”的总课题

一旦 Fig.10 复现成功，你 推断；

* nuisance（幅度相位）边缘化；
* 可用于 ML 的 simulator（生成 x）；
* 校准框架（SNR check + SBC）。

接入 α 的最小改动就是：把 (\omega_{\ell mn}(M_f,\chi_f)) 的 Kerr 关系替换为 (\omega_{率/阻尼的分数偏差、或你 Cardoso-2019 参数化势的映射），然后重复同样的 MCMC/NPE 流程。你现在做 Fig.10，本质是在-




