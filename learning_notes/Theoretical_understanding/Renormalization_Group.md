## 1. 公理 / 定义区（Axioms & Definitions）

---
第一部分

* 标度不变性 (Scale Invariance)：如果拿一把尺子去量这个系统，无论把尺子做得多大或多小，测量到的结构复杂性是一样的。因为系统内部没有一个“绝对的米尺”或“基准单位”来衡量现在的大小。
* 自指和尺度的丧失(Self-Reference and Loss of Characteristic Scale)：
  * 特征尺度:如正态分布中的平均值 $\mu$ 或标准差 $\sigma$
  * 自指 (Self-reference) 或 递归 (Recursion):规则在自身的输出上再次应用。系统的局部行为（Micro-behavior）和整体行为（Macro-behavior）遵循相同的逻辑,
    > 这意味着对系统进行缩放（Zoom in/out）时，你无法通过形态来判断你处于什么层级。
  * 数学上的一个强结论：幂律分布 (Power Law Distribution) 是唯一满足标度不变性的函数形式。
    * 数学方程可以写为：$$f(\lambda x) = g(\lambda) f(x)$$
    > 这里 $g(\lambda)$ 是一个与 $x$ 无关的比例因子。
    * 推导：唯一的解形式就是幂函数。$$f(x) = C x^{-k}$$
    > 验证：我们将 $x$ 替换为 $\lambda x$：
    > $$f(\lambda x) = C (\lambda x)^{-k} = \lambda^{-k} (C x^{-k}) = \lambda^{-k} f(x)$$
    > 这里 $\lambda^{-k}$ 就是那个比例因子 $g(\lambda)$。
    * 结论：如果一个系统没有“特征尺度”（如正态分布中的平均值 $\mu$ 或标准差 $\sigma$），它必须服从幂律分布。
* 统计物理中对应一个非常特定的状态，叫做 临界点 (Critical Point)。
  * 远离临界点时： 系统有特征尺度。例如，水分子在常温下的相互作用距离很短，受制于指数衰减 (Exponential Decay, $e^{-x/\xi}$)，这里的 $\xi$ 就是特征尺度。
  * 处于临界点时（相变）： 当水要在临界温度变成气时，分子间的关联长度 $\xi \to \infty$（趋向无穷大）。此时，一个微小的局部扰动（自指的反馈）可以传播到整个系统。
* 自组织临界性 (Self-Organized Criticality, SOC):当一个系统处于临界状态时,在微小扰动的影响下既有可能什么都不发生,也有可能把影响传播到整个系统.

**上述内容可以被形式化为以下公理体系：**
1. 机制：自指/反馈 (Self-reference/Feedback)导致系统在各个层级上重复自身的逻辑。
2. 现象：标度不变性 (Scale Invariance)使系统失去了能够衡量大小的“绝对尺子”。
3. 数学表达：幂律 (Power Law) $\rightarrow$ $P(x) \propto x^{-\alpha}$ 是描述这种“无尺度”状态的唯一数学语言。


---
