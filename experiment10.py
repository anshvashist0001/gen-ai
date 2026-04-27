import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

latent = 64
bs = 128

G = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(), nn.Linear(256, 784), nn.Tanh())
D = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())

opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)
bce = nn.BCELoss()
mse = nn.MSELoss()

adversarial_losses = []
feature_matching_losses = []
wasserstein_losses = []
epochs = 50

for epoch in range(epochs):
    real = torch.randn(bs, 784)
    z = torch.randn(bs, latent)
    fake = G(z)

    real_labels = torch.ones(bs, 1)
    fake_labels = torch.zeros(bs, 1)

    adv_loss_d = bce(D(real), real_labels) + bce(D(fake.detach()), fake_labels)
    opt_D.zero_grad()
    adv_loss_d.backward()
    opt_D.step()

    adv_loss_g = bce(D(fake), real_labels)
    adversarial_losses.append(adv_loss_g.item())

    real_feat = D[0](real)
    fake_feat = D[0](fake)
    fm_loss = mse(fake_feat.mean(0), real_feat.mean(0).detach())
    feature_matching_losses.append(fm_loss.item())

    w_loss = -(D(real).mean() - D(fake.detach()).mean())
    wasserstein_losses.append(w_loss.item())

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}  Adversarial: {adv_loss_g.item():.4f}  Feature Matching: {fm_loss.item():.4f}  Wasserstein: {w_loss.item():.4f}")

plt.figure(figsize=(10, 4))
plt.plot(adversarial_losses, label="Adversarial Loss")
plt.plot(feature_matching_losses, label="Feature Matching Loss")
plt.plot(wasserstein_losses, label="Wasserstein Loss")
plt.legend()
plt.title("GAN Loss Functions Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
