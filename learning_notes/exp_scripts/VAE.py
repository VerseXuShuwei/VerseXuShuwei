
"""
最小验证: VAE on MNIST
Minimal Verification: Variational Autoencoder on MNIST

运行方式 / How to run:
    pip install torch torchvision matplotlib
    python minimal_vae_mnist.py

会自动下载 MNIST, 训练 10 个 epoch, 然后输出:
  1. reconstruction.png  — 原图 vs 重建图 (encoding → decoding)
  2. generation.png      — 从 N(0,I) 采样, 纯解码生成的新数字
  3. latent_space.png    — 2D 隐空间可视化 (如果 latent_dim=2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────
# 超参数 / Hyperparameters
# ──────────────────────────────────────────────
LATENT_DIM = 2       # 隐空间维度 (设为2方便可视化; 实际常用 20~128)
                      # latent dimension (2 for visualization; typically 20~128)
HIDDEN_DIM = 512     # 中间层维度 / hidden layer dimension
INPUT_DIM = 784      # 28*28, MNIST 图像展平 / flattened MNIST image
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# 模型定义 / Model Definition
# ──────────────────────────────────────────────
class VAE(nn.Module):
    """
    图论视角 / Graph perspective:
        Encoder: x → (μ, σ)     推断方向 inference direction
        Sample:  z = μ + σ⊙ε    重参数化 reparameterization
        Decoder: z → x̂          生成方向 generative direction
    """
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()

        # ---- Encoder: q_φ(z|x) ----
        # 从观测 x 推断隐变量 z 的分布参数
        # Infer distribution parameters of z from observation x
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 784 → 512
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # → μ (均值 mean)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)   # → log(σ²) (对数方差 log-variance)
        # 为什么输出 log(σ²) 而不是 σ²?
        # Why log(σ²) instead of σ²?
        #   1. log(σ²) ∈ (-∞, +∞), 无需约束为正 / no positivity constraint
        #   2. 数值更稳定 / numerically more stable
        #   3. KL 散度公式里本来就有 log / KL formula naturally uses log

        # ---- Decoder: p_θ(x|z) ----
        # 从隐变量 z 生成观测 x
        # Generate observation x from latent z
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),   # 2 → 512
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),    # 512 → 784
            nn.Sigmoid(),  # 输出 ∈ [0,1], 当作像素值的伯努利参数
                           # output ∈ [0,1], treated as Bernoulli parameter for pixel values
        )

    def encode(self, x):
        """
        x → q_φ(z|x) = N(μ, diag(σ²))
        输入一张图, 输出描述 z 分布的 μ 和 log(σ²)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧 / Reparameterization Trick:
            z = μ + σ ⊙ ε,   ε ~ N(0, I)

        为什么需要这个技巧?
        Why do we need this trick?
            "从分布采样" 这个操作不可微 → 梯度无法回传到 encoder
            "Sampling from a distribution" is not differentiable → gradients can't flow to encoder

            重参数化把随机性转移到 ε 上, z 成为 μ 和 σ 的确定性函数
            Reparameterization shifts randomness to ε, making z a deterministic function of μ and σ
            → 梯度可以通过 z 流向 μ 和 σ → 流向 encoder 参数 φ
            → gradients can flow through z to μ and σ → to encoder params φ
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²)) = √(σ²)
        eps = torch.randn_like(std)     # ε ~ N(0, I)
        return mu + std * eps           # z = μ + σ ⊙ ε

    def decode(self, z):
        """z → p_θ(x|z): 解码, 从隐空间回到观测空间"""
        return self.decoder(z)

    def forward(self, x):
        """完整的前向传播: encode → reparameterize → decode"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# ──────────────────────────────────────────────
# 损失函数 / Loss Function = -ELBO
# ──────────────────────────────────────────────
def loss_function(x_recon, x, mu, logvar):
    """
    ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
    我们最小化 -ELBO, 所以:
    loss = -E_q[log p(x|z)] + KL(q(z|x) || p(z))
         = reconstruction_loss + kl_loss

    重建项 / Reconstruction term:
        用二值交叉熵 (BCE), 因为 decoder 输出 Sigmoid ∈ [0,1]
        Binary cross-entropy since decoder outputs Sigmoid ∈ [0,1]
        直觉: 重建得越像原图, 这项越小
        Intuition: better reconstruction → smaller loss

    KL 项 / KL term:
        KL(N(μ, σ²) || N(0, I)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        这是高斯对高斯 KL 的解析解 (closed-form)
        直觉: encoder 输出的分布越接近标准正态, 这项越小
        Intuition: encoder distribution closer to N(0,I) → smaller loss
        作用: 防止隐空间"坍缩"成离散点, 保证隐空间的连续性和可采样性
        Purpose: prevent latent space "collapse" into discrete points,
                 ensure continuity and sampleability
    """
    # 重建损失: 逐像素 BCE, 求和后取 batch 平均
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL 散度: 解析公式
    # -0.5 * Σ_j (1 + log(σ_j²) - μ_j² - σ_j²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (recon_loss + kl_loss) / x.size(0)  # 除以 batch size 取平均


# ──────────────────────────────────────────────
# 训练循环 / Training Loop
# ──────────────────────────────────────────────
def train():
    # 数据加载
    transform = transforms.ToTensor()  # [0,255] → [0,1]
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = VAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 训练
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.view(-1, INPUT_DIM).to(DEVICE)  # 展平 flatten

            x_recon, mu, logvar = model(batch_x)
            loss = loss_function(x_recon, batch_x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.4f}")

    return model, test_loader, test_data


# ──────────────────────────────────────────────
# 可视化 / Visualization
# ──────────────────────────────────────────────
def visualize_reconstruction(model, test_loader):
    """对比原图和重建图 / Compare originals vs reconstructions"""
    model.eval()
    with torch.no_grad():
        batch_x, _ = next(iter(test_loader))
        batch_x = batch_x.view(-1, INPUT_DIM).to(DEVICE)
        x_recon, _, _ = model(batch_x)

    n = 10
    fig, axes = plt.subplots(2, n, figsize=(15, 3))
    for i in range(n):
        # 原图 / Original
        axes[0, i].imshow(batch_x[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        # 重建 / Reconstruction
        axes[1, i].imshow(x_recon[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

    plt.tight_layout()
    plt.savefig('reconstruction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 保存重建对比图 → reconstruction.png")


def visualize_generation(model):
    """从 p(z)=N(0,I) 采样, 纯生成 / Sample from prior and generate"""
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, LATENT_DIM).to(DEVICE)  # 采样 64 个 z
        generated = model.decode(z)

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i in range(64):
        axes[i//8, i%8].imshow(generated[i].cpu().view(28, 28), cmap='gray')
        axes[i//8, i%8].axis('off')

    plt.suptitle('Generated from z ~ N(0, I)', fontsize=14)
    plt.tight_layout()
    plt.savefig('generation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 保存生成图 → generation.png")


def visualize_latent_space(model, test_data):
    """
    2D 隐空间可视化 (仅当 LATENT_DIM=2 时有意义)
    Visualize 2D latent space (only meaningful when LATENT_DIM=2)

    这张图是验证 VAE 是否 work 的最直观证据:
    - 如果不同数字聚成不同的簇 → encoder 学会了有意义的表征
    - 如果簇之间连续过渡 → 隐空间结构良好, 可以插值
    - 如果整体接近圆形 → KL 正则在起作用, 拉向 N(0,I)
    """
    if LATENT_DIM != 2:
        print("⚠ 隐空间可视化需要 LATENT_DIM=2, 跳过")
        return

    model.eval()
    zs, labels = [], []
    loader = DataLoader(test_data, batch_size=512, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            x = x.view(-1, INPUT_DIM).to(DEVICE)
            mu, _ = model.encode(x)  # 用 μ 作为确定性编码
            zs.append(mu.cpu())
            labels.append(y)

    zs = torch.cat(zs).numpy()
    labels = torch.cat(labels).numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(zs[:, 0], zs[:, 1], c=labels, cmap='tab10',
                          s=1, alpha=0.5)
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel('z₁')
    plt.ylabel('z₂')
    plt.title('Latent Space (colored by digit label)')
    plt.savefig('latent_space.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 保存隐空间图 → latent_space.png")


# ──────────────────────────────────────────────
# 主函数 / Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Latent dim: {LATENT_DIM}")
    print(f"Training VAE on MNIST for {EPOCHS} epochs...\n")

    model, test_loader, test_data = train()

    print("\n生成可视化 / Generating visualizations...")
    visualize_reconstruction(model, test_loader)
    visualize_generation(model)
    visualize_latent_space(model, test_data)

    print("\n✅ 完成! 查看三张图:")
    print("   reconstruction.png  — 重建效果")
    print("   generation.png      — 随机生成")
    print("   latent_space.png    — 隐空间结构")
