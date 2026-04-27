import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)

latent = 20

encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 400), nn.ReLU())
fc_mu = nn.Linear(400, latent)
fc_var = nn.Linear(400, latent)
decoder = nn.Sequential(nn.Linear(latent, 400), nn.ReLU(), nn.Linear(400, 784), nn.Sigmoid())

params = list(encoder.parameters()) + list(fc_mu.parameters()) + list(fc_var.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)

losses = []

for epoch in range(15):
    for x, _ in loader:
        h = encoder(x)
        mu = fc_mu(h)
        log_var = fc_var(h)

        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)

        x_recon = decoder(z)
        recon_loss = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = (recon_loss + kl_loss) / x.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses.append(loss.item())
    print(f"Epoch {epoch+1}/15  Loss: {loss.item():.4f}")

with torch.no_grad():
    z = torch.randn(16, latent)
    gen = decoder(z).view(16, 1, 28, 28)

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    axes[0, i].imshow(gen[i].squeeze(), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(gen[i+8].squeeze(), cmap='gray')
    axes[1, i].axis('off')

plt.suptitle("VAE Generated Images")
plt.tight_layout()
plt.show()

plt.plot(losses)
plt.title("VAE Training Loss")
plt.xlabel("Epoch")
plt.show()
