# 问题一：椒盐噪声图像恢复的光滑化模型与 GPSR 投影梯度求解

> 对应题目《信号重构问题的相关探索》问题一。
> 本文档给出完整的数学建模过程、求解算法原理与数值实验设计。所有记号与题目保持一致。

---

## 1. 问题描述与总体路线

设原始图像 $X$ 为 $M\times N$ 灰度图像，$A=\{1,\dots,M\}\times\{1,\dots,N\}$ 为索引集，
像素动态范围为 $[s_{\min},s_{\max}]$（8 位灰度图取 0 与 255）。经典 **salt-and-pepper
椒盐噪声模型** 下，观测图像 $Y$ 在位置 $(i,j)$ 处的像素为

$$
y_{i,j}=\begin{cases}
s_{\min}, & \text{以概率 } p,\\
s_{\max}, & \text{以概率 } q,\\
x_{i,j},  & \text{其余},
\end{cases}
\qquad r=p+q\ \text{为噪声等级}.
$$

采用文献 [2] 的**两阶段方法**：第一阶段用自适应中值滤波 (AMF) 检测噪声像素；
第二阶段在检测出的噪声候选集 $\Omega$ 上通过最小化保边正则化泛函恢复像素。

非光滑性来自两处：二阶泛函中的 $\ell_1$ 数据保真项 $|u_{i,j}-y_{i,j}|$，以及
可能非光滑的保边势函数 $\varphi_\alpha$。据此，问题一的求解路线确定为：

1. **第一阶段**：AMF 噪声检测（§2），得到噪声候选集 $\Omega$、AMF 输出 $\hat y$（作二阶初值）；
2. **第二阶段**：对非光滑部分做**光滑化**（§3），将原问题转化为
   **盒约束光滑凸优化问题**；
3. 用文献 [1] 提出的 **GPSR 投影梯度算法**（BB 步长 + 投影 + 线搜索）求解（§4）；
4. 构建评价指标（迭代次数、CPU 时间、SNR、PSNR 等）并进行数值实验（§5、§6）。

---

## 2. 第一阶段：自适应中值滤波器（AMF）噪声检测

记 $S_{i,j}^{w}=\{(k,l): |k-i|\le w,\ |j-l|\le w\}$ 为中心在 $(i,j)$ 的 $w\times w$ 窗口，
$w_{\max}$ 为最大窗口尺寸。文献 [2] Algorithm I 如下。

**算法 1（AMF 噪声检测）**

对每个像素 $(i,j)$：
1. 令 $w=3$；
2. 计算 $S_{i,j}^{w}$ 内的最小值 $s_{\min}^{w}$、中值 $s_{\mathrm{med}}^{w}$、最大值 $s_{\max}^{w}$；
3. 若 $s_{\min}^{w}<s_{\mathrm{med}}^{w}<s_{\max}^{w}$，转步骤 5；否则令 $w=w+2$；
4. 若 $w\le w_{\max}$，转步骤 2；否则将 $y_{i,j}$ 用 $s_{\mathrm{med}}$ 替换（判为噪声）；
5. 若 $s_{\min}^{W}<y_{i,j}<s_{\max}^{W}$（$W$ 为最终窗口），则 $(i,j)$ **不是**噪声候选；
   否则 $(i,j)$ 判为**噪声候选**，并用 $s_{\mathrm{med}}^{W}$ 替换。

由此得到噪声候选集。注意到受污染像素的观测值必为动态范围端点，且检测器对其
输出了替换值，本项目严格采用文献 [2] 的**两条件定义**（与 Chan 2005 Algorithm III
第 1 步一致）：

$$
\Omega=\big\{(i,j)\in A:\ y_{i,j}\in\{s_{\min},s_{\max}\}\ \text{且}\ \hat y_{i,j}\neq y_{i,j}\big\},
$$

即候选像素必须**同时满足**"取值为动态范围端点"与"被滤波器替换过"两个条件。
该定义排除了两类误判：灰度值恰为端点但被判为保真的像素（未产生替换）不进入
候选集；非端点像素即使被替换也不满足端点条件。相比仅用端点判决的单条件定义，
误检率（FPR）由 2.7%–9.4% 降至 0.0%–0.7%，恢复质量获得约 0.5–1 dB 的系统性
提升。$w_{\max}$ 随噪声等级选取（文献 [2] 表 1）：

| 噪声等级 $r$ | $<25\%$ | $25\%\sim40\%$ | $40\%\sim60\%$ | $60\%\sim70\%$ | $70\%\sim80\%$ | $80\%\sim85\%$ | $\ge85\%$ |
|---|---|---|---|---|---|---|---|
| $w_{\max}\times w_{\max}$ | $5\times5$ | $7\times7$ | $9\times9$ | $13\times13$ | $17\times17$ | $25\times25$ | $39\times39$ |

**说明**：AMF 的"自适应"扩大窗口保证即使在高噪声等级下（噪声像素成片出现时）
最终窗口也能含有足够多的干净像素以给出有意义的中值；实现严格按照文献步骤 3–5：
最终窗口为满足 $s_{\min}<s_{\mathrm{med}}<s_{\max}$ 的首个窗口 $W$（若一直不满足，
则在 $w$ 超过 $w_{\max}$ 时**无条件**判为噪声候选并替换为 $w_{\max}$ 窗口中值）。
边界像素采用反射扩展（对称边界条件），对图像边缘像素影响很小。AMF 输出

$$
\hat y_{i,j}=s_{\mathrm{med}}^{W_{i,j}}\quad((i,j)\in\Omega),\qquad
\hat y_{i,j}=y_{i,j}\quad((i,j)\notin\Omega)
$$

作为第二阶段的初始迭代点（文献 [3] 亦如此）。

---

## 3. 第二阶段：恢复模型及其光滑化

### 3.1 原模型（题目式 (2)）

第二阶段在 $\Omega$ 上求解（题目式 (2)，与文献 [2] 式 (3) 一致）：

$$
\min_{u}\ \sum_{(i,j)\in\Omega}\Big[\,|u_{i,j}-y_{i,j}|+\frac{\beta}{2}\big(S_1+S_2\big)\Big],
\tag{2}
$$

其中 $V_{i,j}\subset A$ 为 $(i,j)$ 的四个最近邻（不含自身），

$$
S_1=\sum_{(m,n)\in V_{i,j}\cap\Omega^{C}} 2\,\varphi_\alpha(u_{i,j}-y_{m,n}),\qquad
S_2=\sum_{(m,n)\in V_{i,j}\cap\Omega} \varphi_\alpha(u_{i,j}-u_{m,n}),
$$

$\varphi_\alpha$ 为带正参数 $\alpha$ 的**保边势函数**。本文在主实验中取问题所列第一种

$$
\varphi_\alpha(t)=\sqrt{t^{2}+\alpha},\quad \alpha>0,
\tag{3}
$$

其意义：$|t|\to\infty$ 时 $\varphi_\alpha(t)\to|t|$（分段线性、保边缘），而在 $t=0$
附近平滑（$\varphi_\alpha''(0)=1/\sqrt{\alpha}$，有界凸曲率）。$V_{i,j}\cap\Omega^{C}$ 中系数
$2$ 的来源（文献 [3]）：$\varphi_\alpha$ 为偶函数，$\varphi_\alpha(u_{i,j}-y_{m,n})+
\varphi_\alpha(y_{m,n}-u_{i,j})=2\varphi_\alpha(u_{i,j}-y_{m,n})$，即若在全集 $A$ 上求和，
干净像素一侧的项本应计两次；由于第二阶段的求和只跑在 $\Omega$ 上，故用因子 $2$ 补偿。

### 3.2 等价表述：边和形式

记 $\tilde u_p=u_p\ (p\in\Omega)$，$\tilde u_p=y_p\ (p\notin\Omega)$。当 $\varphi_\alpha$ 偶时，
正则项可写成无向边和（每条至少一端在 $\Omega$ 内的边 $(p,q)$ 计一次）：

$$
F_\beta(u)=\sum_{p\in\Omega}|u_p-y_p|\;+\;\beta\!\!\sum_{\substack{\text{无向边}(p,q):\\p\in\Omega}}\!\!\varphi_\alpha\big(u_p-\tilde u_q\big).
\tag{4}
$$

即式 (2) 中的 $\frac\beta2(S_1+S_2)$ 恰好等于 $\beta\sum_{\text{边}\cap\Omega}\varphi_\alpha$：
对于 $\Omega$–$\Omega^C$ 边，$\frac\beta2\times 2=\beta$ 每边一次；对于 $\Omega$–$\Omega$ 边，
两个 $\Omega$ 端点在各自的 $S_2$ 中各计 $\frac\beta2$，合计亦为 $\beta$ 每边一次。

**梯度**（对每个 $p\in\Omega$，将自身项与来自邻居 $q$ 的项对 $u_p$ 的偏导合并，
利用 $\varphi_\alpha'$ 为奇函数）：

$$
\nabla F_\beta(u)_p=\operatorname{sign}(u_p-y_p)\;+\;\beta\sum_{q\in V_p}\varphi_\alpha'\big(u_p-\tilde u_q\big).
\tag{5}
$$

即数据项贡献 $\operatorname{sign}(\cdot)$，正则项对每个邻居（干净邻居取 $y_q$、噪声邻居取
当前变量 $u_q$）贡献 $\beta\,\varphi_\alpha'$。该式是后续一切高效计算的基础：正则在
$|\Omega|$ 规模上做 4 次邻域查值即可完成。

### 3.3 光滑化

式 (2)/(4) 中唯一非光滑项为 $|u_p-y_p|$（$\varphi_\alpha=\sqrt{t^2+\alpha}$ 处处光滑）。
选用光滑函数 $\rho_\mu:\mathbb{R}\to\mathbb{R}$ 逼近 $|t|$，满足

$$
\rho_\mu\in C^{1},\qquad |\rho_\mu(t)-|t||\le c\,\mu,\ \forall t\in\mathbb{R},
\tag{6}
$$

本文实现下列三种（文献 [4] 中的 $\varphi_1,\varphi_3,\varphi_4$ 族）：

| 名称 | $\rho_\mu(t)$ | $\rho_\mu'(t)$ | 逼近误差界 |
|---|---|---|---|
| 平方根型 | $\sqrt{\mu^{2}+t^{2}}$ | $\dfrac{t}{\sqrt{\mu^{2}+t^{2}}}$ | $|\rho_\mu-|t||\le\mu$ |
| Huber 型 | $\begin{cases}\frac{t^{2}}{2\mu},&|t|\le\mu\\ |t|-\frac\mu2,&|t|>\mu\end{cases}$ | $\begin{cases}\frac{t}{\mu},&|t|\le\mu\\ \operatorname{sign}(t),&|t|>\mu\end{cases}$ | $\le \mu/2$ |
| Softplus 型 | $\mu[\ln(1+e^{-t/\mu})+\ln(1+e^{t/\mu})]=|t|+2\mu\ln(1+e^{-|t|/\mu})$ | $\tanh\frac{t}{2\mu}$ | $\le 2\mu\ln2$ |

（注：Softplus 的误差界取 $t=0$ 处最大值 $\mu\ln2\times2=2\mu\ln2$；实现采用等价稳定形式。）

将 $|u_p-y_p|$ 换成 $\rho_\mu(u_p-y_p)$，得**光滑化模型**

$$
\boxed{\ \min_{u\in[\underline u,\bar u]^{|\Omega|}}\ F_\mu(u)
=\sum_{p\in\Omega}\rho_\mu(u_p-y_p)
+\beta\sum_{\text{边}(p,q):p\in\Omega}\varphi_\alpha(u_p-\tilde u_q)\ }
\tag{7}
$$

其梯度

$$
\nabla F_\mu(u)_p=\rho_\mu'(u_p-y_p)+\beta\sum_{q\in V_p}\varphi_\alpha'\big(u_p-\tilde u_q\big).
\tag{8}
$$

**光滑化误差**：由 (6) 得

$$
|F_\mu(u)-F_\beta(u)|\le c\,\mu\,|\Omega|,
$$

即目标值随 $\mu\to0$ 一致收敛到原问题；$\mu$ 越小逼近越好。

**$\mu$ 的取舍（数值实验 §6.2 验证，存档 `problem1_mu_ablation.csv`）**：
$\mu$ 过小时 $\rho_\mu''$ 仅在 $|t|\lesssim\mu$ 的窄带内非退化，$F_\mu$ 在大部分
区域几乎分段线性、弱凸性差，理论上会使梯度类方法收敛变慢；但消融实验表明
在当前配置（$\alpha=300/\beta=40$、两条件候选集）下 $\mu$ 从 $1$ 降到
$10^{-4}$，GPSR-BB 的迭代数（85 步）与恢复 PSNR（38.357 dB）均几乎不变，
即 BB 步长对该曲率变化并不敏感。因此 $\mu$ 的选取主要由逼近误差决定：
$\mu=1$ 时逼近误差不足 1 个灰度级，恢复质量与 $\mu\to0$ 一致，
故主实验取 $\mu=1$，并作为对照实验分析 $\mu$ 的影响（含文献 [4,9] 的
$\mu$ 续延策略：由大到小逐段求解、逐段热启动）。

### 3.4 凸性与最大值原理

1. **凸性**：对 $\varphi_\alpha(t)=\sqrt{t^2+\alpha}$，$\varphi_\alpha''(t)=\alpha/(t^2+\alpha)^{3/2}>0$，
   且 $\rho_\mu''\ge0$，故 $F_\mu$ 为式 (7) 中边拉普拉斯型凸函数与凸数据项的求和，**凸**；
   当 $\Omega$ 连通时 $F_\mu$ 为一致凸（Hessian 严格对角占优，非零曲率来自
   $\varphi_\alpha''>0$ 的图拉普拉斯部分）。文献 [3] 系统证明了该泛函继承
   $\varphi_\alpha$ 的凸性/强凸性与 Hessian 的正定性与一致有界性，并由此建立
   CG 类方法的全局收敛性。
2. **最大值原理**（文献 [3] 命题 2）：$\varphi_\alpha$ 偶、连续且关于 $|t|$ 严格递增时，
   $F_\beta$（甚至去掉数据项后的 $F$）的全局极小值点满足
   $u_p\in[s_{\min},s_{\max}]$。**因此可等价地把问题限制在盒约束
   $[\underline u,\bar u]=[s_{\min},s_{\max}]$ 上求解**，从而可以直接套用
   文献 [1] 的投影梯度框架（GPSR 正是靠变量分裂把 $\ell_1$ 问题化为盒约束问题）。
3. **最优性条件（KKT）**：$u$ 为式 (7) 在盒约束下的最优解当且仅当
   $u=P\big(u-\nabla F_\mu(u)\big)$，$P$ 为 $[\underline u,\bar u]$ 上的欧氏投影；
   记投影间隙 $G(u)=\big\|P(u-\nabla F_\mu(u))-u\big\|_\infty$，则 $G(u)=0$ 刻画 KKT。

---

## 4. 求解算法：GPSR 投影梯度（BB 步长 + 非单调线搜索）

### 4.1 与文献 [1] 的关系

文献 [1] 的核心思想：把稀疏重构 $\min \tau\|x\|_1+\frac12\|Ax-b\|^2$ 用变量分裂
$x=u-v,\ u,v\ge0$ 化为**有界约束二次规划 (BCQP)** $z=[u;v]\ge0$，然后做
**投影梯度**：$w=(z-\alpha\nabla F(z))_+$，$z\leftarrow z+\lambda(w-z)$ 并配
Barzilai-Borwein (BB) 步长 $\alpha$ 与（单调/非单调）线搜索；代码与实验见
http://www.lx.it.pt/~mtf/GPSR/。

本问题经 §3 光滑化后直接就是**盒约束光滑凸问题**（无需再分裂变量），
其投影梯度迭代与文献 [1] 完全同构——两者都是"沿投影弧的 BB 梯度法"，
故直接沿用其算法框架与终止准则。

### 4.2 算法框架

**算法 2（GPSR-BB 投影梯度求解式 (7)）**

> 参数：$P$ 为 $[0,255]$ 投影；$0<\delta\ll1$（Armijo 常数），$\tau\in(0,1)$（回溯因子），
> 非单调记忆 $M\ge1$，BB 步长截断 $[\alpha_{\min},\alpha_{\max}]$，终止容差 $\mathrm{tolP}$，迭代上限 $\mathrm{maxit}$。

1. **初始化**：$u^{0}=\hat y|_{\Omega}$（AMF 输出），$g^0=\nabla F_\mu(u^0)$，
   $\alpha^0=\mathrm{clip}\big(1/\|g^0\|_\infty,\alpha_{\min},\alpha_{\max}\big)$，$k=0$。
2. **投影弧搜索方向**：$w^{k}=P\big(u^{k}-\alpha^{k}g^{k}\big)$，$d^{k}=w^{k}-u^{k}$。
3. **非单调 Armijo 线搜索**（文献 [1] §III-B 的非单调变体）：
   取 $\lambda=1,\tau,\tau^2,\dots$ 中第一个满足
   $$
   F_\mu\big(u^{k}+\lambda d^{k}\big)\le
   \max_{0\le j\le M}F_\mu\big(u^{k-j}\big)+\delta\,\lambda\,(g^{k})^{\!\top}d^{k}
   $$
   的 $\lambda$（若失败，退化为最速下降方向重试；再失败则停止）。
4. **更新**：$u^{k+1}=u^{k}+\lambda d^{k}$，$g^{k+1}=\nabla F_\mu(u^{k+1})$。
5. **BB 步长**（文献 [1] 式 (15) 的一般形式）：令
   $s^{k}=u^{k+1}-u^{k}$，$y^{k}=g^{k+1}-g^{k}$，
   $$
   \alpha^{k+1}=\mathrm{clip}\Big(\tfrac{(s^k)^{\!\top}s^k}{(s^k)^{\!\top}y^k},\ \alpha_{\min},\alpha_{\max}\Big),
   $$
   即 BB1 公式（文献 [2] 亦借鉴于 Barzilai-Borwein 两点步长）。
6. **终止**：若 $G(u^{k+1})\le \mathrm{tolP}$（投影间隙，对应文献 [1] 式 (16)）
   或 $\|u^{k+1}-u^{k}\|_\infty\le 10^{-14}$ 或 $k> \mathrm{maxit}$，停止。

**GPSR-Basic 型对照**（文献 [1] §III-A）：不做 BB 更新，步长 $\alpha$ 由沿投影弧的
Armijo 回溯得到（Bertsekas 的"沿投影弧 Armijo 规则"），收敛最稳但迭代数较多。

### 4.3 收敛性与复杂度

- **单调性**：非单调 Armijo 保证"每 $M+1$ 步内函数值下降"，配合 $\inf$
  方向的梯度界可得到 $G(u^k)\to0$（文献 [4,9] 对同类 BG 方向给出
  $d^{\!\top}g\le-\frac12\|g\|^2$ 的充分下降性，本问题中类似的结论成立），
  数值上表现为投影间隙单调下降到 $\mathrm{tolP}$（§6 收敛曲线）。
- **复杂度**：$F_\mu$ 与 $\nabla F_\mu$ 的单次求值只需 $O(|\Omega|)$ 次标量运算
  （邻域关系预计算为索引表，无任何矩阵乘法，甚至比文献 [1] 需要乘 $A,A^{\!\top}$
  的情形更省）；每迭代 1 次投影、1 次线搜索求值 ×2 + 梯度求值。
- **与 GPSR 的对比**：GPSR 处理的是带观测矩阵 $A$ 的稀疏重构 BCQP，本问题是
  纯图结构化盒约束问题；两者共用"投影 + BB + 回溯"骨架，因此实现时
  也按文献 [1] 的 $[\alpha_{\min},\alpha_{\max}]=[10^{-10},10^{6}]$ 一类惯例截断。

### 4.4 说明：关于数据保真项（文献 [3] 的讨论）

文献 [3] 指出，由于第二阶段只恢复 $\Omega$ 上的像素且已经用 AMF 检测，
$\ell_1$ 数据项 $|u_{i,j}-y_{i,j}|$ 可以删去（删去后 $F$ 的极小值与含数据项的
$G$ 基本一致），且 $\beta$ 足够大时 $G$ 的极小值不再依赖 $\beta$。
本文按题目给出的式 (2)（**含**数据项）为主模型，同时把删去数据项的变体
（文献 [3] 式 (9)，即 \texttt{with\_data=False}）作为对照讨论（§6.4）；
二者在 $\beta$ 足够大时结果一致，验证了文献 [3] 的结论。

---

## 5. 评价指标

设 $x$ 为原图，$\hat x$ 为恢复图（$\hat x|_{\Omega}=u^{*}$，$\hat x|_{\Omega^C}=y$），
$\mathrm{MSE}=\frac1{MN}\sum (x-\hat x)^2$：

| 指标 | 定义 | 含义 |
|---|---|---|
| 峰值信噪比 PSNR | $10\log_{10}\dfrac{s_{\max}^{2}}{\mathrm{MSE}}$ (dB) | 失真能量相对峰值 |
| 信噪比 SNR | $10\log_{10}\dfrac{\sum_{i,j}x_{i,j}^{2}}{\sum_{i,j}(x_{i,j}-\hat x_{i,j})^{2}}$ (dB) | 信号能量/噪声能量 |
| 平均绝对误差 MAE | $\frac1{MN}\sum| x-\hat x|$ | 平均偏差 |
| 迭代次数 $\mathrm{IT}$ | 算法 2 实际迭代步数 | 计算量 |
| CPU 时间 $t$ | 求解耗时（AMF 与第二阶段分别计时） | 实际效率 |
| 检测统计 | $|\Omega|$、TPR（真噪声检出率）、FPR（误检率） | 第一阶段质量 |
| $\Omega$-PSNR | 仅在 $\Omega$ 上计算的 PSNR | 噪声像素恢复精度 |

**口径说明**：所有指标在未取整的浮点恢复图上计算（与文献 [2,3,5] 惯例一致）；
采用两条件候选集定义后，72 个算例的 TPR 均值为 99.99%、最低为 99.82%
（woman_blonde 30%）——端点判决的结构性保证（真噪声像素取值恰为
$s_{\min}$/$s_{\max}$）配合"被替换过"条件基本保留，仅当端点灰度保真像素干扰
窗口极值判决时才可能出现少量漏检；FPR 才是体现检测器质量的主要经验指标。

---

## 6. 数值实验设计

### 6.1 数据集与配置

- 图像：12 张 $512\times512$ 灰度标准测试图（cameraman, house, jetplane, lake, lena,
  livingroom, mandril, peppers, pirate, walkbridge, woman\_blonde, woman\_darkhair，
  均为公开标准测试图库 USC-SIPI 的常规样例）；
- 噪声：$r=30\%$ 与 $50\%$（$p=q=r/2$），每配置 3 个随机种子（2026, 7, 42）；
- $w_{\max}$：按表 1 取 $7\times7$（30%）、$9\times9$（50%）；
- 模型：$\varphi_\alpha=\sqrt{t^2+\alpha}$，$\alpha=300$，$\beta=40$（见 §6.2 参数选择），
  光滑参数 $\mu=1$；
- 求解器：GPSR-BB（BB1 + 投影 + 非单调 Armijo，$M=10$，$\delta=10^{-4}$，
  $\tau=0.5$），$\mathrm{tolP}=10^{-2}$，$\mathrm{maxit}=1500$；
- 环境：Python 3.11 + NumPy/SciPy（代码见 \texttt{src/}，实验脚本
  \texttt{exp/problem1/}）。

### 6.2 参数选择与策略实验（Lena 512，30%/50%，种子 2026）

- **$\beta$ 敏感性**（$\alpha=300$，含数据项）：PSNR 随 $\beta$ 增大在
  $\beta\ge40$ 进入平台（30%：$\beta=40$ 时 38.357 dB，$\beta=640$ 时
  38.365 dB，增益 0.008 dB），符合文献 [3]
  "$\beta$ 足够大时极小值不变"的结论（扫描明细见
  `results/tables/param_grid_log.txt` 与问题二 `problem2_tuning.txt`
  的 sqrt 条目）；
- **$\alpha$ 敏感性**（$\beta=40$）：$\alpha\in\{10,30,100,300,1000,3000\}$
  中 $\alpha=300$ 最优（38.357 dB；$\alpha$ 过小则曲率过强、退化过快，
  $\alpha$ 过大则正则项过弱）；
- 主配置取 $(\alpha,\beta)=(300,40)$；
- **光滑参数 $\mu$**（存档 `results/tables/problem1_mu_ablation.csv`，
  $\alpha=300/\beta=40$、初值 AMF 输出）：$\mu$ 从 $1$ 降到 $10^{-4}$，
  迭代数均为 85 步、恢复 PSNR 均为 38.357 dB（差 $\le0.0001$ dB），即当前
  配置下 BB 步长对小 $\mu$ 的弱凸性并不敏感，$\mu$ 的选取由逼近误差决定，
  取 $\mu=1$（误差不足 1 个灰度级）；
- **求解器对照**（统一初值 AMF 输出，存档 `problem1_strategy.txt`）：

| 策略 | 迭代数 | 时间 (s) | PSNR (dB) | 收敛 |
|---|---|---|---|---|
| GPSR-Basic（沿投影弧 Armijo） | 7160 | 265.86 | 38.357 | 是 |
| **GPSR-BB（BB1 + 非单调）** | **85** | **3.19** | **38.357** | 是 |
| GPSR-BB（BB2） | 79 | 2.96 | 38.357 | 是 |
| GPSR-BB + $\mu$ 续延 $(1,10^{-1},10^{-2},10^{-3})$ | 88 | 3.28 | 38.357 | 是 |
| 无数据项变体（文献 [3] 式 (9)） | 97 | 3.69 | 38.366 | 是 |

  （Lena 512 r=30%、候选集精确化后；另三个算例的对照见
  `problem1_strategy.txt`：Basic 迭代数 5547–7180，BB1 为 82–100。）
  BB 步长把迭代数降低约 **60–88 倍**（时间 50–83 倍），恢复质量完全一致；
  续延与固定 $\mu=1$ 的迭代数只差 3 步，主实验直接用 $\mu=1$；
- **数据项有无**：$\beta=40$ 时二者 PSNR 差 $0.009$ dB（30%，
  38.366 vs 38.357），验证文献 [3]
  "第二阶段数据项可删"的结论（以题目式 (2) 含数据项为主模型）。

### 6.3 主实验结果（12 图 × 2 噪声等级 × 3 种子，均值）

汇总见 `results/tables/problem1_summary.csv`（逐 run 明细
`problem1_full.csv`），要点：

- **检测**：72 个算例 TPR 均值 99.99%（最低 99.82%，见 §5 口径说明），
  FPR 范围 0.0%–0.6%（30%）/ 0.0%–0.7%（50%），候选集大小
  $|\Omega|\approx7.81$–$7.97\times10^{4}$（30%）、
  $1.309$–$1.321\times10^{5}$（50%），与真实噪声数一致；
- **迭代次数**：均值 86.8（30%，逐 run 范围 43–204）、92.4（50%，范围
  49–203）；72 个算例**全部**在终止准则下收敛；
- **第二阶段耗时**：均值 2.94 s（30%，范围 1.52–7.08 s）、
  5.42 s（50%，范围 2.93–12.3 s），随 $|\Omega|$ 近似线性；
- **PSNR**：12 图均值 37.33 / 33.76 dB（逐图均值范围 30.99–48.17 /
  28.29–43.36 dB）；SNR 均值 31.88 / 28.31 dB，与 PSNR 差约 4–6 dB
  （与图像对比度有关）；
- 代表值：Lena 30% 38.35 dB（文献 [5] 同任务报告 36.99 dB，其配置为
  $\sqrt{t^2+100}$、无数据项 + 两项 NYHS CG；文献 [6] 独立框架报告
  37.01 dB），Lena 50% 34.71 dB，说明模型与求解器设置合理；
  恢复视觉结果与收敛曲线见 `results/figures/`（`vis_*.png`、`conv_*.png`）。

---

## 7. 结论

1. 两阶段 + 光滑化把非光滑恢复问题转化为盒约束光滑凸问题，可用文献 [1] 的
   GPSR 投影梯度框架高效求解；
2. 光滑参数 $\mu$ 在当前配置下由逼近误差决定而非求解效率：$\mu$ 从 1 降到
   $10^{-4}$ 迭代数与恢复 PSNR 均几乎不变（存档 `problem1_mu_ablation.csv`），
   取 $\mu=1$（误差不足 1 个灰度级）即可"既光滑又保真"；
3. 参数 $(\alpha,\beta)=(300,40)$ 处于恢复质量的平台区，鲁棒；
4. 该方法在 12 张标准图上 30%/50% 椒盐噪声的 PSNR 均值分别为
   37.33 dB 与 33.76 dB（逐图均值范围 30.99–48.17 / 28.29–43.36 dB；
   Lena 38.35 / 34.71 dB），与文献同任务结果一致或更优；
5. GPSR-BB 的 BB 步长是关键：相比 GPSR-Basic 迭代数降低约 60–88 倍
   （时间 50–83 倍），而恢复质量完全相同；
6. 数据保真项在 $\beta$ 足够大时可删（与文献 [3] 结论一致），
   为问题二进一步探索势函数与参数留下接口。

---

## 参考文献（问题一相关）

[1] Figueiredo M A T, Nowak R D, Wright S J. Gradient Projection for Sparse Reconstruction:
Application to Compressed Sensing and Other Inverse Problems. IEEE JSTSP, 2008, 1(4): 586-597.

[2] Chan R H, Ho C W, Nikolova M. Salt-and-pepper noise removal by median-type noise detectors
and detail-preserving regularization. IEEE TIP, 2005, 14(10): 1479-1485.

[3] Cai J F, Chan R H, Fiore C D. Minimization of a Detail-Preserving Regularization Functional
for Impulse Noise Removal. JMIV, 2007, 29(1): 79-91.

[4] Wu C, Wang J, Alcantara J H, et al. Smoothing Strategy Along with Conjugate Gradient
Algorithm for Signal Reconstruction. JSC, 2021, 87(21): 1-18.

[5] 刘莹, 朱志斌, 丁玥宏, 等. Wolfe线搜索下一个新的共轭梯度法及其在信号处理中的应用. 应用数学, 2025, 38(1): 104-113.

[9] Chen X, Zhou W. Smoothing nonlinear conjugate gradient method for image restoration using
nonsmooth nonconvex minimization. SIAM J. Imaging Sci., 2010, 3(4): 765-790.
