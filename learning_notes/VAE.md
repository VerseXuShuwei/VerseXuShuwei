# VAE(Variational Autoencoder) - 灵魂7问:

## 1. 它解决的问题是什么

**核心问题**：我们有一堆高维数据（比如图片），想学会两件事——理解它（提取本质结构）和创造它（生成新样本）。
传统的自编码器能压缩，但它的隐空间是散乱的，从里面随便采一个点，解码出来大概率是垃圾。GAN 能生成，但无法反过来把一张图编码回隐空间。
VAE 的野心是同时做到这两件事：学到一个结构化的、连续的、可采样的隐空间，既能编码也能解码，而且隐空间里相邻的点对应语义相邻的数据。
用一句话说：VAE 解决的是"如何学一个既能推断又能生成的概率模型，同时让隐空间有好的几何结构"。

---

## 2. 定义：从直觉到理论母语

直觉层。 想象一个画家,看过几万幅人脸，脑子里自然形成了一些"旋钮"：肤色、脸型、表情、年龄…… 给这位画家一组旋钮值，就能画出一张脸。
VAE 就是用神经网络自动发现这些旋钮（z\mathbf{z}z），同时学会两个映射：看到一张脸→推测旋钮值（encoder），给定旋钮值→画出一张脸（decoder）。

**图论视角:**
把 VAE 画成一个有向图模型（贝叶斯网络）：z → x

只有两个节点，一条边。
z 是隐变量（latent，不可观测），x 是观测变量（observed）。箭头方向表示生成因果：z 产生 x。

训练时还需要一条反向推断路径：

生成方向（generative）:   z  ──→  x      由 p_θ(x|z) 描述

推断方向（inference）:    x  ──→  z      由 q_φ(z|x) 近似

> 这两条路径构成了 VAE 的核心对偶结构。

从图论角度看，VAE 在同一张图上同时学习正向因果（生成）和逆向推断（识别），这就是它和普通 AE 的本质区别——它不是在学一个确定性映射，而是在学两个概率分布之间的关系。

**理论母语**（概率图模型）
VAE 属于 深度隐变量模型（Deep Latent Variable Model），用神经网络参数化一个经典的贝叶斯推断问题。
它的理论根基是变分推断（Variational Inference），**核心思想**是：既然真正的后验p(z∣x) 算不出来，就用一个可以算的分布 qϕ(z∣x)去逼近它。

---

## 3. 最小数学形式 + 符号详解

* **生成模型（Generative Model）**：

pθ(x)=∫pθ(x∣z) p(z) dz

**符号拆解**

- x (observed data): 观测数据
  
*一个数据样本，如一张 28×28 的图，展平成 784 维向量*

- z (latent variable): 隐变量
  
*低维隐空间中的一个点，比如 mathbb{R}^{20}中的一个向量*

- p(z) (prior):先验
  
*我们对 z 的先验信念，通常选 N(0,I)——标准正态*

- pθ(x∣z)(likelihood / decoder): 似然 / 解码器
  
*给定旋钮值 z，生成 x 的概率。θ 是 decoder 神经网络的参数*

- pθ(x) (marginal likelihood / evidence): 边际似然 / 证据
  
*对所有可能的 z 积分后得到的 x 的总概率。这个积分**intractable（不可解析计算）**，是一切问题的起源*

- ∫⋅ dz (marginalization):边际化
  
*把隐变量"积掉"，遍历所有可能的 z 值*


**推断模型（Inference Model）**：

qϕ(z∣x)=N(z;μϕ(x),diag(σϕ^2(x)))

**符号拆解**

- qϕ(z∣x) (approximate posterior / encoder): 近似后验 / 编码器
  
*看到 x 后，猜测 z 长什么样。ϕ 是 encoder 网络的参数*

- μϕ(x) (mean network): 均值网络
  
*encoder 输出的 z 的均值向量*

- σϕ^2(x) (variance network): 方差网络
  
*encoder 输出的 z 的方差向量（对角协方差）*

- diag(⋅) (diagonal covariance): 对角协方差

*假设 z 的各维度条件独立——这是简化假设*


* **ELBO（优化目标）**：
  
L(θ,ϕ;x)=Eqϕ(z∣x)[log⁡pθ(x∣z)]⏟reconstruction term 重建项−DKL(qϕ(z∣x)∥p(z))⏟regularization term 正则项


\mathcal{L}(\theta, \phi; \mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{reconstruction term 重建项}} - \underbrace{D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{regularization term 正则项}}


L(θ,ϕ;x) = reconstruction term 重建项Eqϕ​(z∣x)​[logpθ​(x∣z)]​​ − regularization term 正则项DKL​(qϕ​(z∣x)∥p(z))
​​
**符号拆解**

- L (ELBO - Evidence Lower BOund): 证据下界

*log⁡pθ(x) 的下界，最大化它来间接最大化数据的对数似然*

- Eqϕ[⋅] (expectation under q): 在 q 下的期望

*从 encoder 采样 z，然后算 [⋅] 里面东西的平均*

- log⁡pθ(x∣z) (log-likelihood): 对数似然

*decoder 重建 x 的好坏。越大 = 重建越精确*

- DKL(⋅∥⋅) (KL divergence): KL 散度

*衡量 qϕ(z∣x) 和 p(z) 之间的"距离"。越小 = encoder 的输出越接近标准正态先验*


* **重参数化技巧（Reparameterization Trick）**：

z=μϕ(x)+σϕ(x)⊙ϵ,ϵ∼N(0,I)

\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})

z=μϕ​(x)+σϕ​(x)⊙ϵ,ϵ∼N(0,I)

*这个技巧把"从 qϕ​ 采样"这个不可微操作，变成了"从固定分布采样 + 可微变换"，让梯度可以流过 μ 和 σ 回传到 encoder。*

**符号拆解**

- ϵ (noise sample): 噪声样本

*从标准正态分布采的随机向量*

- ⊙ (element-wise product): 逐元素乘法

*每个维度分别乘*

---

## 4. 和哪些对象相连

用图论的方式画连接关系：

```
                    ┌──── AE（自编码器）
                    │     去掉概率，去掉KL → 就是确定性的AE
                    │
        ┌───────── VAE ─────────┐
        │           │           │
    变分推断(VI)   概率图模型   深度学习
        │           │           │
        │      贝叶斯网络    神经网络
        │                  (encoder/decoder)
        │
   ┌────┴─────┐
   │          │
 EM算法    MCMC
(VAE是VI的   (另一种近似
 神经网络版)  推断方法)

```
下游/变体：
VAE ──→ β-VAE（加强 disentanglement）
    ──→ VQ-VAE（离散隐空间，通向 DALL-E）
    ──→ Latent Diffusion（SD 的 VAE encoder/decoder）
    ──→ CVAE（条件生成）
    ──→ VAE-GAN（混合模型）
    
在 Stable Diffusion 语境里：VAE 是压缩器。
它把 512×512×3 的图压缩到 64×64×464 的隐空间，扩散过程在这个隐空间里做，最后 VAE decoder 解码回像素空间。

---

## 5. 在哪个时间尺度上起作用

这里"时间"理解为计算过程的时间线：

* 训练时（hours~days）：encoder 和 decoder 联合训练，每个 batch 执行"编码→采样→解码→算 loss→反传"。

* 单次前向传播 ：给一个 x ，encoder 算出 μ,σ 采样 z，decoder 生成 x^。

* 生成时 ：从 p(z)=N(0,I) 采样，过 decoder，得到新样本。无需 encoder。

* 在 Latent Diffusion 中：VAE 只在最前和最后各用一次——编码一次、解码一次。中间的扩散过程可能走几十步甚至上百步迭代，但那不是 VAE 的事。

---

## 6. 在整个系统里放哪一层

```
┌─────────────────────────────────────────────┐
│  应用层 (Application)                        │
│  图像生成 / 异常检测 / 数据增强 / 药物发现     │
├─────────────────────────────────────────────┤
│  生成模型层 (Generative Model)               │
│  VAE / GAN / Diffusion / Flow               │  ← VAE 在这里
├─────────────────────────────────────────────┤
│  表征学习层 (Representation Learning)        │
│  VAE的encoder输出 = 可用的隐表征              │  ← VAE的encoder也在这里
├─────────────────────────────────────────────┤
│  神经网络层 (Neural Network)                 │
│  MLP / CNN / Transformer 作为encoder/decoder │
├─────────────────────────────────────────────┤
│  优化层 (Optimization)                       │
│  SGD / Adam + 重参数化技巧                    │
├─────────────────────────────────────────────┤
│  概率论层 (Probability Theory)               │
│  变分推断 / KL散度 / ELBO                     │
└─────────────────────────────────────────────┘
```

VAE 有意思的地方在于它横跨多层：底层依赖概率论和变分推断的理论，中层用神经网络做参数化，上层直接就是一个生成模型。

## 7. 最小验证：MNIST VAE

一个最小可跑的 VAE，在 MNIST 上训练，然后我们看重建效果和从隐空间采样生成新手写数字。

[实验脚本](learning_notes/VAE.py)
