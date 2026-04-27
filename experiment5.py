import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

N = 1000
torch.manual_seed(42)

x = torch.randn(N, 2)
y1 = torch.cat([torch.randn(N//2, 2) + 2, torch.randn(N//2, 2) - 2])

class FlowLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        shift = self.net(x1)
        scale = torch.exp(self.log_scale)
        z2 = x2 * scale + shift
        log_det = self.log_scale.expand(x.size(0))
        return torch.cat([x1, z2], dim=1), log_det

    def inverse(self, z):
        z1, z2 = z[:, 0:1], z[:, 1:2]
        shift = self.net(z1)
        scale = torch.exp(self.log_scale)
        x2 = (z2 - shift) / scale
        return torch.cat([z1, x2], dim=1)

class NormalizingFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FlowLayer() for _ in range(4)])

    def forward(self, x):
        log_det_total = torch.zeros(x.size(0))
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total += log_det
        return x, log_det_total

    def sample(self, n):
        z = torch.randn(n, 2)
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

model = NormalizingFlow()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
prior = torch.distributions.Normal(0, 1)

losses = []
for epoch in range(300):
    z, log_det = model(y1)
    log_prob = prior.log_prob(z).sum(1)
    loss = -(log_prob + log_det).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/300  Loss: {loss.item():.4f}")

with torch.no_grad():
    samples = model.sample(500).numpy()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(y1[:, 0], y1[:, 1], s=5)
plt.title("Real Data")

plt.subplot(1, 3, 2)
plt.scatter(samples[:, 0], samples[:, 1], s=5, color='orange')
plt.title("Generated Samples")

plt.subplot(1, 3, 3)
plt.plot(losses)
plt.title("Training Loss")

plt.tight_layout()
plt.show()
