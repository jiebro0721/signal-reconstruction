# 信号重构问题的相关探索 —— 优化建模与求解

本仓库对应训练题目《信号重构问题的相关探索》：以**图像恢复 / 稀疏信号重构**为背景，
建立优化模型并设计算法，共三个问题：

1. **问题一（已完成）**：两阶段椒盐噪声恢复 —— AMF 检测 + 保边泛函的**光滑化模型** +
   文献 [1] 提出的 **GPSR 投影梯度算法**求解，构建迭代次数 / 时间 / SNR / PSNR 评价体系；
2. **问题二（已完成）**：5 种保边势函数 $\varphi_\alpha$ 对恢复效果的影响
   （各自参数标定下的公平比较 + 鲁棒性/收敛性分析）；
3. **问题三（待做）**：修正 PRP 共轭梯度法重构稀疏信号并与 GPSR 比较。

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
│       ├── check_potentials.py  #   5 种势函数梯度 FD 校验 + 凸性抽查
│       ├── tune_potentials.py   #   (α, β) 独立标定 + PSNR-α 敏感性
│       ├── run_problem2.py      #   ★ 主实验 (3 图 × {30%,50%} × 2 种子 × 5 势函数)
│       ├── make_figures2.py     #   PSNR-α 曲线 / 指标条形图 / 恢复对比图
│       └── results/             #   表 (csv/json/txt), 图 (png)
└── docs/
    ├── problem1_model.md    # ★ 问题一完整数学建模与算法原理文档
    └── problem2_model.md    # ★ 问题二完整数学建模与实验分析文档
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

- 问题一：12 图 × 2 噪声等级 × 3 种子共 72 算例全部收敛；30%/50% 平均 PSNR
  36.39 / 33.57 dB，平均迭代 86/89 次，平均用时 2.6/4.2 s；BB 步长相比
  GPSR-Basic 提速约 85 倍；Lena 30% 达 37.41 dB。
- 问题二：最优参数下 5 种势函数 PSNR 差距 ≤0.25 dB（恢复质量不敏感）；
  $\sqrt{t^2+\alpha}$ 与 Huber 对 $\alpha$ 鲁棒且收敛快，$|t|^{\alpha}$（α<2）
  导数无界导致上千步不收敛，$\log\cosh$/$|t|/\alpha-\log$ 在 α 偏大时正则失效。

## 参考文献

见题目附件。核心：Figueiredo et al. *Gradient Projection for Sparse Reconstruction*,
IEEE JSTSP 2008; Chan, Ho & Nikolova, IEEE TIP 2005; Cai, Chan & Di Fiore, JMIV 2007;
Wu et al., JSC 2021; Chen & Zhou, SIAM J. Imaging Sci. 2010。
