import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)

latent = 64

G = nn.Sequential(
    nn.Linear(latent, 256), nn.ReLU(),
    nn.Linear(256, 512), nn.ReLU(),
    nn.Linear(512, 784), nn.Tanh()
)

D = nn.Sequential(
    nn.Linear(784, 512), nn.LeakyReLU(0.2),
    nn.Linear(512, 256), nn.LeakyReLU(0.2),
    nn.Linear(256, 1), nn.Sigmoid()
)

opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)
loss_fn = nn.BCELoss()

g_losses = []
d_losses = []

for epoch in range(20):
    for real, _ in loader:
        real = real.view(-1, 784)
        bs = real.size(0)

        real_labels = torch.ones(bs, 1)
        fake_labels = torch.zeros(bs, 1)

        z = torch.randn(bs, latent)
        fake = G(z).detach()

        loss_D = loss_fn(D(real), real_labels) + loss_fn(D(fake), fake_labels)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        z = torch.randn(bs, latent)
        loss_G = loss_fn(D(G(z)), real_labels)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    g_losses.append(loss_G.item())
    d_losses.append(loss_D.item())
    print(f"Epoch {epoch+1}/20  G Loss: {loss_G.item():.4f}  D Loss: {loss_D.item():.4f}")

plt.plot(g_losses, label="Generator")
plt.plot(d_losses, label="Discriminator")
plt.legend()
plt.title("GAN Training Losses")
plt.xlabel("Epoch")
plt.show()
