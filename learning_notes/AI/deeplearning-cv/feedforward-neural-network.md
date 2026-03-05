
## 0. 前馈神经网络（Feedforward Neural Network FNN）

* *一种最早、最基本、也是最常见的人工神经网络 结构。它的核心特点是：信息在网络中单向、无反馈地流动，从输入层经过隐藏层（如果有的话），最终到达输出层。网络中不存在环路或回路，信息不会返回到之前的节点。*


## 1.理解 transformer注意力机制(计算/技术细节,理论解释用图论视角)

### 第一步**token → embedding**

> 核心假设:**语义可能本来就是一种高维几何结构。**语言只是它的投影。

Transformer 里所谓 **token → embedding**，本质上是一种**离散对象到连续向量空间的映射**。
一串输入字符进入模型时变成 **one-hot 向量****embedding**

先从最原始的表示开始。
假设词表（vocabulary）大小是 (V)。
每个 token 其实只是一个 **整数索引**：

[
t \in {0,1,2,...,V-1}
]

例如：

```
"cat" → 381
"dog" → 912
```

神经网络不能直接处理“符号”。它处理的是 **向量**。
于是第一步通常是把这个整数变成 **one-hot 向量**：

[
x \in \mathbb{R}^{V}
]

形式是

[
x_i =
\begin{cases}
1 & i = t \
0 & i \ne t
\end{cases}
]

例如 (V=5)：

```
token = 2
x = [0,0,1,0,0]
```

这一步只是 **离散符号编码（symbol encoding）**，没有学习，没有语义。

---

然后进入真正的 embedding。

Transformer 会有一个 **embedding matrix**：

[
E \in \mathbb{R}^{V \times d}
]

其中：

* (V)：词表大小
* (d)：embedding 维度（例如 512、768、4096）

embedding 的计算非常直接：

[
e = xE
]

因为 (x) 是 one-hot，这个矩阵乘法会简化为：

[
e = E_{t}
]

也就是说：

**embedding 就是取矩阵的第 (t) 行。**

如果写成代码，本质就是：

```python
embedding = E[token_id]
```

所以从计算上看：

**embedding layer = 一个可学习的查找表（lookup table）。**

---

那理论基础在哪里？

这里其实有三个层面的思想。

---

第一层：**线性代数视角**

one-hot 向量可以看作标准基：

[
e_1, e_2, ..., e_V
]

embedding matrix (E) 定义了一个线性映射：

[
\mathbb{R}^{V} \rightarrow \mathbb{R}^{d}
]

也就是把 **离散基向量**映射到一个 **低维连续空间**。

因此：

[
E e_i = v_i
]

其中 (v_i) 是 token (i) 的 embedding。

换句话说：

**embedding 就是给每个离散符号分配一个向量。**

---

第二层：**分布式表示（Distributed Representation）**

这是早期神经网络语言模型的核心思想，来自：

Yoshua Bengio
2003 年论文 *A Neural Probabilistic Language Model*

传统 NLP 用的是 **离散特征**：

```
dog = 一个符号
cat = 一个符号
```

彼此完全独立。

问题：
模型无法表达 **相似性结构**。

分布式表示的思想是：

> 用向量空间表达语义结构。

如果

```
dog → v_dog
cat → v_cat
car → v_car
```

训练后通常会出现：

[
|v_{dog}-v_{cat}| < |v_{dog}-v_{car}|
]

这种几何结构不是手工设计的，而是通过训练自动形成。

---

第三层：**统计语言模型的压缩**

从信息论角度看，embedding 在做一种 **低维压缩**。

如果不用 embedding：

输入维度 = 词表大小 (V)

例如：

```
V = 50,000
```

那第一层参数会爆炸。

embedding 把它压缩到：

```
d = 768
```

这相当于学习一个：

[
V \rightarrow d
]

的低维表示。

某种意义上它类似：

* PCA（主成分分析）
* 矩阵分解
* 潜变量模型

只是这里是 **通过任务损失函数学习出来的**。

---

第四层（更有意思的一点）：
embedding 其实和 **softmax 输出层**有数学联系。

在很多语言模型中：

输出概率：

[
P(w|h) = softmax(W h)
]

其中：

[
W \in \mathbb{R}^{V \times d}
]

很多模型会直接 **共享参数**：

[
W = E
]

也就是说：

**输入 embedding 和输出分类器使用同一个矩阵。**

这背后有个漂亮的解释：

embedding 向量其实在学习

> token 在概率空间中的坐标。

---

所以把整个过程压缩成一句非常“物理化”的描述：

**token embedding 是一个可学习的离散→连续映射，本质是给每个符号分配一个向量坐标，使语言统计结构在向量空间中变得线性可操作。**

---

还有一个很多人第一次意识到会觉得很怪的事实：

Transformer **并不知道 token 是“词”**。

在数学上它看到的只是：

```
整数 → 向量 → 矩阵运算
```



