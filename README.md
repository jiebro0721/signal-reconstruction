# 信号重构问题的相关探索 —— 优化建模与求解

本仓库对应训练题目《信号重构问题的相关探索》：以**图像恢复 / 稀疏信号重构**为背景，
建立优化模型并设计算法，共三个问题：

1. **问题一（已完成）**：两阶段椒盐噪声恢复 —— AMF 检测 + 保边泛函的**光滑化模型** +
   文献 [1] 提出的 **GPSR 投影梯度算法**求解，构建迭代次数 / 时间 / SNR / PSNR 评价体系；
2. **问题二（已完成）**：5 种保边势函数 $\varphi_\alpha$ 对恢复效果的影响
   （各自参数标定下的公平比较 + 鲁棒性/收敛性分析）；
3. **问题三（已完成）**：修正 PRP+ 共轭梯度法（光滑化 + μ 续延）重构 n=4096、
   160 尖峰的稀疏信号，与文献 [1] GPSR-BB 的完整对比（同目标值口径 + 去偏置）。

## 目录结构

```
signal-reconstruction/
├── data/
│   └── test_images/         # 512×512 灰度标准测试图 (USC-SIPI)
├── src/                     # 核心算法库
│   ├── noise_model.py       #   椒盐噪声生成 / 图像读取
│   ├── amf.py               #   第一阶段: 自适应中值滤波 (AMF) 检测
│   ├── restoration_model.py #   第二阶段: 光滑化保边泛函 (5种势函数/光滑函数/梯度)
│   ├── solvers.py           #   GPSR-Basic / GPSR-BB 投影梯度求解器 (含续延)
│   ├── sparse_reconstruction.py # 问题三: 稀疏信号/测量矩阵生成 + 光滑化 ℓ1-二次模型
│   ├── cg_solvers.py        #   问题三: 非线性CG(PRP+/FR/HS+/DY, 强Wolfe) + GPSR-BCQP
│   └── metrics.py           #   PSNR / SNR / MAE / SSIM / 边缘PSNR / 检测统计
├── exp/
│   ├── problem1/            # 问题一实验
│   │   ├── sanity_check.py      #   单元验证 (梯度 FD 校验 / AMF / 求解器)
│   │   ├── tune.py              #   求解器与 μ 快速调优
│   │   ├── param_grid.py        #   (α, β) 网格 + 数据项有无
│   │   ├── verify_params.py     #   参数在多个图像上的验证
│   │   ├── run_problem1.py      #   ★ 主实验 (12 图 × {30%,50%} × 3 种子)
│   │   ├── strategy_compare.py  #   Basic/BB1/BB2, μ 固定/续延, 数据项有无
│   │   ├── make_figures.py      #   收敛曲线 / 视觉对比 / PSNR 汇总图
│   │   └── results/             #   表 (csv/txt), 图 (png), 恢复图 (tif, 不入库)
│   └── problem2/            # 问题二实验
│       ├── check_potentials.py  #   5 种势函数梯度 FD 校验(20样本) + 凸性抽查
│       ├── tune_potentials.py   #   (α, β) 独立标定(v2: β 至 640, 选优优先收敛解)
│       ├── merge_tuning.py      #   并行调参日志/参数合并 → problem2_params.json
│       ├── run_problem2.py      #   ★ 主实验 (3图×{30%,50%}×2种子×5势函数, 含 OOS 汇总)
│       ├── make_figures2.py     #   PSNR-α 曲线 / 指标条形图 / 恢复对比图
│       └── results/             #   表 (csv/json/txt), 图 (png)
│   └── problem3/            # 问题三实验
│       ├── sanity_check.py      #   实例生成/梯度FD/两算法收敛性验证
│       ├── run_problem3.py      #   ★ 主实验 (10实例 × 8方法, 含同目标值口径)
│       ├── make_figures3.py     #   收敛曲线/信号对比/汇总条形图
│       └── results/             #   表 (csv/json), 图 (png)
└── docs/
    ├── problem1_model.md    # ★ 问题一完整数学建模与算法原理文档
    ├── problem2_model.md    # ★ 问题二完整数学建模与实验分析文档
    └── problem3_model.md    # ★ 问题三完整数学建模与对比实验文档
```

## 运行环境

- Python 3.11 (conda env `cumcm2025c`), 依赖: `numpy scipy matplotlib pillow tifffile`
  （`tifffile` 用于读取带多余 alpha 通道的标准测试 TIFF）

## 快速复现

```bash
# 问题一
python exp/problem1/sanity_check.py          # 单元验证
python exp/problem1/run_problem1.py          # 主实验 (约 10 分钟)
python exp/problem1/strategy_compare.py      # 策略对比
python exp/problem1/make_figures.py          # 图
# 问题二
python exp/problem2/check_potentials.py      # 势函数梯度校验
python exp/problem2/tune_potentials.py       # (α,β) 标定 (约 30 分钟)
python exp/problem2/run_problem2.py          # 主实验 (约 30 分钟)
python exp/problem2/make_figures2.py         # 图
```

## 方法摘要

- **第一阶段**：文献 [2] 自适应中值滤波 (AMF)，按噪声等级取最大窗口
  $w_{\max}$（30%→7×7，50%→9×9），输出噪声候选集 $\Omega$ 与 AMF 图像；
- **第二阶段**：最小化保边泛函（题目式 (2)）
  $$F_\beta(u)=\sum_{p\in\Omega}|u_p-y_p|+\beta\!\!\sum_{\substack{\text{边}(p,q):\\p\in\Omega}}\!\!\varphi_\alpha(u_p-\tilde u_q),$$
  非光滑项 $|\cdot|$ 以 $\rho_\mu(t)=\sqrt{\mu^2+t^2}$ 光滑化（$|\rho_\mu-|\cdot||\le\mu$）；
  由最大值原理约束到盒 $[s_{\min},s_{\max}]$；
- **求解**：文献 [1] 投影梯度框架 —— BB 步长 + 投影 + 非单调 Armijo 线搜索
  (GPSR-BB)，终止准则为投影间隙 $\|P(u-\nabla F)-u\|_\infty\le 10^{-2}$。
  问题一主配置 $(\alpha,\beta)=(300,40)$；问题二对 5 种势函数各自标定 $(\alpha^{*},\beta^{*})$。

## 主要结论

- 问题一：12 图 × 2 噪声等级 × 3 种子共 72 算例全部收敛；检测严格采用文献的
  两条件候选集定义，TPR=100%、误检率≤0.7%；30%/50% 平均 PSNR 37.33 / 33.76 dB
  （Lena 30% 达 38.35 dB，比 AMF 直接输出高 5.2 dB、比 7×7 中值滤波高 10.1 dB），
  平均迭代 87/92 次，平均用时 2.9 / 5.4 s；BB 步长相比 GPSR-Basic 迭代数降低
  60–88 倍。
- 问题二（v3 协议：α、β 独立标定，幂函数型经 $(t^2+\mu^2)^{\alpha/2}$ 光滑化
  后正常收敛）：五种势函数最优参数下 PSNR 极差 ≤0.19 dB（恢复质量不敏感，
  out-of-sample 一致）；参数敏感性呈宽平台、悬崖、正常平台三类行为；收敛速度
  差 6–14 倍，平方根型/幂函数型最快。
- 问题三：CG-PRP+（Aᵀb 初值 + 相对变化准则分段停止 + μ 续延至 1e-6）与
  GPSR-BB 均恢复全部 160 个尖峰；同目标值口径 122.7 步/1.64 s，为 GPSR-BB
  （29.1 步/0.14 s）的 4.2 倍；去偏置后相对误差 0.0286 vs 0.0265，几乎持平；
  共轭参数对照：PRP+/HS+ ~170 步，FR/DY ~1080 步；原始 PRP 与修正 PRP 数值
  一致，印证截断修正的价值在理论保障。

## 审查修复记录

Zcode 审查报告所列为 19 项，全部处理完毕（含 1 🔴 / 6 🟠 / 12 🟡），
详见 `docs/CHANGES_after_review.md`。核心修复：GPSR-Basic 对照数据重跑至
真实收敛、PSNR-α 曲线解析修复、β 网格扩展消除调参伪影、AMF 对齐论文语义、
softplus 实现修正、初值统一、out-of-sample 汇总。

## 参考文献

见题目附件。核心：Figueiredo et al. *Gradient Projection for Sparse Reconstruction*,
IEEE JSTSP 2008; Chan, Ho & Nikolova, IEEE TIP 2005; Cai, Chan & Di Fiore, JMIV 2007;
Wu et al., JSC 2021; Chen & Zhou, SIAM J. Imaging Sci. 2010。
