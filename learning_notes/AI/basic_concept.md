# AI 学习记录模板（适配你的学习方式）

> 这个模板是为你的学习思维结构设计的：从公理到推演，从历史到当下，从机制到应用。只需要把内容添进去即可。

---

## 0. 本次学习的核心主题（Topic）

* *一句话描述你正在理解的核心概念/问题是什么。*
线性代数（Linear Algebra）、**凸优化（Convex Optimization）和统计学（Statistics）**的第一性原理出发;
概念拆解为三个核心模块：
度量工具：范数（Norms）——我们如何定义“大小”？
约束手段：正则化（Regularization）——我们为何以及如何限制模型？
描述工具：矩（Moments）——我们如何描述数据的形状？


---

* Batch Norm:由于BN使用的是当前Batch的统计量，它总能把数据强行拉回正态分布。
对于一个Batch中的输入数据 $x$，BN的变换如下：
  - $$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$$$y = \gamma \hat{x} + \beta$$
  - 符号解释：$x$: 输入特征向量。
  - $\mu$: 均值（Mean），描述数据的中心位置。
  - $\sigma^2$: 方差（Variance），描述数据的离散程度。
  - $\epsilon$: 极小值（Epsilon），防止分母为0。
  - $\gamma, \beta$: 可学习参数（Learnable Parameters），用于恢复网络的表达能力，防止归一化破坏了特征的分布。
  
* Batch Norm中的分布偏移（Distribution Shift）噩梦:
源于训练时的随机近似（Stochastic Approximation）与推理时的确定性期望（Deterministic Expectation）之间的数学不一致性。
噩梦的根源在于 $\mu$ 和 $\sigma$ (统计量)的来源不同：
训练 (Training):当前Batch的统计量 ($\mu_B, \sigma_B$)局部估计（Local Estimation）。模型实际上学会了利用“当前这批数据”的特性来归一化自己。
推理 (Inference):全局移动平均 ($\mu_{running}, \sigma_{running}$)全局期望（Global Expectation）。通常使用训练过程中所有Batch统计量的滑动平均值。
表现形式:训练时的 $\mu_B$ 和 推理时的 $\mu_{running}$ 出现巨大的鸿沟。
  - 独立同分布（I.I.D.）假设的破裂
    - BN强烈依赖于一个假设：每一个Mini-batch都是总体数据集的无偏采样。如果这个假设不成立，$\mu_{running}$ 就无法代表真实的测试数据。
  - 实例归一化与群体归一化的混淆
    - 在训练时，因为使用的是Batch统计量，BN不仅归一化了特征，实际上还引入了样本间的相互依赖。即：样本 A 的输出结果，取决于同在一个Batch里的样本 B、C、D 是什么（因为它们共同决定了 $\mu_B$）。而在推理时，这种依赖消失了。
  - 训练集与测试集的域偏移 (Domain Shift),这是最直观的分布偏移。
    - 如果训练集是ImageNet（自然图像），测试集是素描画。训练集的 $\mu_{running}$ 是自然图像的统计特征。测试数据进来时，被强行减去了自然图像的均值。
    - 结果：特征空间发生扭曲，神经网络后续的权重矩阵处理的是错误的数值范围。
* 解决方案的演进（理解原理）
为了解决这个噩梦，后续的研究试图切断对“Batch统计量”的依赖，转向**样本内（Instance-level）**的统计。

Layer Normalization (LN) / Group Normalization (GN)：

原理：不再跨样本（Batch维度）计算均值方差，而是在单个样本内部（Channel维度）计算。

解决：彻底消除了训练和推理的不一致性。无论Batch size是多少，无论分布怎么变，对单个样本的处理逻辑是恒定的。
---

## 1. 公理 / 定义区（Axioms & Definitions）

* 数学或概念公理：
* 范数 (Norms) :向量空间的“尺子”:选用不同的范数 (norm) —— 比如 L₁, L₂, L∞ ……其实是在定义不同的“世界
  > 在数学上，范数是将向量映射到非负实数的函数，本质是定义在这个空间中的“长度”或“大小”的单位。设向量 $\mathbf{x} = [x_1, x_2, ..., x_n]$。
  > p-范数 (Lp norm) 随着 𝑝 从 1 → 2 → … → ∞ 变化。
    - p=1：L₁-范数 → 单位球是菱形 (diamond),xn绝对值相加
    - p=2：L₂-范数 → 单位球是圆 (circle),xn平方和开根
    - p→∞：L∞-范数 → 单位球渐趋正方形 (square),max(∣x∣,∣y∣)≤1
* 正则化(regularization):约束优化的本质
  > 为什么要加正则化？
  -  第一性原理：为了解决病态问题（Ill-posed Problem）和过拟合（Overfitting）。
  -  在损失函数 $J(\mathbf{w})$ 后面加上一项 $\lambda \|\mathbf{w}\|$，本质是在**带约束的优化问题（Constrained Optimization）**中引入拉格朗日乘子（Lagrange Multiplier）。
  - 
* “矩”这个词在 BN（Batch Normalization）里，其实不是深奥的新概念，只是一个统计学老词被借来用了。
  - 在数学里，**矩（moment）**用来描述一个分布“长什么样”。可以把它当成：对随机变量的“形状特征”的压缩描述。
  - 一阶矩（first moment）:严格说是一阶中心矩之前的原始矩，对应的就是𝐸[𝑋],也就是均值（mean）。这个分布整体“站在什么位置”。
  - 二阶矩（second moment）:如果直接算𝐸[𝑋2],叫二阶原始矩；但在 BN 里用的是二阶中心矩：𝐸[(𝑋−𝜇)2],这就是方差（variance）。数据“散得有多开”。
  - 在更一般的概率论里：一阶矩 → 位置;二阶矩 → 尺度;三阶矩 → 偏度（skewness，左右歪不歪）;四阶矩 → 峰度（kurtosis，尖不尖）
  - BN 只用到了前两阶，刚好够把分布“拉直 + 拉齐”。
 
  - 
---
Attention(Q,K,V)=softmax(QKTdk)V
Scores = Softmax( (Q · K^T) / sqrt(d_k) )
Output = Scores · V Q\K\V: 在Transformer中，Q（Query）、K（Key）、V（Value）都不是原始输入，而是由 输入序列 X [batch, seq_len, dim_model],通过三个不同的线性层（全连接层）投影得到的。 Q = X · W_Q K = X · K_W V = X · W_V 这里的 W_Q, W_K, W_V 都是可学习的权重矩阵。 经过线性变换后，输出的 Q, K, V 自然保持了这个 [batch, seq_len, dim_model] 的基本形状。 为了做矩阵乘法，需要将K转置 Q 的形状：[batch, seq_len, dim_model] K 的转置形状：[batch, dim_model, seq_len] 它们相乘的结果 Scores = Q · K^T 的形状就是：[batch, seq_len, seq_len]。 这个结果矩阵的每一行 i 就代表了序列中第 i 个token（作为Query）与序列中所有token（作为Key）的注意力分数。 sqrt(d_k) 和 d_k
