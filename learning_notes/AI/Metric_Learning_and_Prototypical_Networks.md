### 度量学习（Metric Learning） 和 原型网络（Prototypical Networks） 的核心灵魂
**最好的分类器，不应该只是画线的（Linear），而应该是比较相似度的（Metric）。**

只学特征，不要 Logits(线性映射)

> **“怎么把高维的‘思路’（Features）翻译成人类能懂的‘结论’（Logits/Classes）”**这个接口上。
(这个转化必然有损失)

---

### 1. 只学特征，不要 Logits是否可以规避标签噪音？

**（Pure Feature Learning / Metric Learning）**

* 这正是 **SimCLR, MoCo, BYOL** 这些自监督学习（Self-Supervised Learning）在做的事。
* 它们根本不看标签，只看“谁和谁长得像”(相似度)。这规避了标签噪音
* **结果：** 模型学会了极好的“思路”。它能把所有的“褐斑病”聚在一起，把“健康叶”聚在一起。
* 模型虽然把它们分开了，但它**不知道**那一堆聚在一起的东西叫“褐斑病”。它只知道它们是“同类”(cluster)。

引出第二个问题：如何利用feature约束

---

### 两种不同的 Logits

**（Linear vs. Distance-based）**

不能把 Logits 狭隘地理解成了**“那个容易作弊的线性层”**。

把 Feature 变成人能懂的“概率”，有两种截然不同的算法:

#### 线性分类器 - Linear Classifier

* **做法：** 画一条线（超平面）。
* **公式：** $Logits = W \times Feature + b$
* **隐喻：** 只要过了这条线，就算赢。
* **缺点：** 容易作弊。就像只要分数够 60 就是及格，不管是不是抄的(过拟合噪音)。

#### 原型分类器 - Prototypical Classifier

* **做法：** **测距离（Distance Measurement）**。
* **公式：** $Logits = -Distance(Feature, Prototype)$
* 这里的 Prototype 就是在 Feature Space 里存的“标准病害样本”或“标准健康样本”。


* **隐喻：** Feature 必须长得像“标准答案”，分才高。
* **区别：** 这虽然输出的也是 Logits（也可以 Softmax 成概率），但这个 Logits 是**基于“相似度”的**，而不是基于“切一刀”的。

**关键点：**
如果是**基于距离的 Logits**，模型就不敢看噪音了。因为噪音的 Feature 和“真正的目标原型”距离一般很远。模型为了让距离变近（Logit 变高），只能强迫自己去学真正的目标特征。

---

### 最佳折中方案（Feature Critic）

纯原型网络（Prototypical Network）适用在哪些情景下:
完全抛弃线性层，改用纯原型网络（Prototypical Network），通常用于**Few-shot Learning（小样本学习）**。在全监督或弱监督的大数据下，它训练起来比较难收敛。

**所以，一般是走“混合路线”：**

针对我正在做的项目,现在的架构其实是在保留**线性分类器（高效、好训练）**的同时，用**Feature Loss（特征约束）**去 模拟 **原型网络**的严谨性。

* **Teacher (Feature Loss):** 负责管“思路”。它拿着尺子在 Feature Space 也就是高维空间里量：“你这个 Feature 离健康原型太近了！虽然分类器想给你高分，但我判定你不合格！”
* **Student (Logits/Ranking):** 负责管“输出”。它负责把这些被修正过的、干净的 Feature，映射成你需要的 0 或 1 的概率。

### 总结

> **最好的分类器，不应该只是画线的（Linear），而应该是比较相似度的（Metric）。**

这个 **Negative Repulsion Loss**，就是在强迫线性分类器去**尊重**特征空间的距离。

1. **输入：** 图片。
2. **中间（Feature）：** 模型产生“思路”。
* *Feature Critic 介入：* “思路”不能像背景！

3. **输出（Logits）：** 线性层把“思路”翻译成概率。
* *Ranking Loss 介入：* 概率要比负样本高！

这样，既利用了 Logits 带来的明确信息（这是什么病），又利用了 Feature Space 带来的抗噪能力（这真的不像背景）。

---

### next step:

了解一下度量学习（Metric Learning） 和 原型网络（Prototypical Networks） 
