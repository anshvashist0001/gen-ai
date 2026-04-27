import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

seq_len = 20
vocab_size = 10
embed_dim = 32
hidden_dim = 64
n_samples = 2000

data = torch.randint(0, vocab_size, (n_samples, seq_len))
x = data[:, :-1]
y = data[:, 1:]

class AutoregressiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out)

model = AutoregressiveModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

losses = []
for epoch in range(30):
    pred = model(x)
    loss = loss_fn(pred.reshape(-1, vocab_size), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/30  Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    start = torch.randint(0, vocab_size, (1, 1))
    generated = [start.item()]
    inp = start
    h = None
    for _ in range(seq_len - 1):
        out, h = model.lstm(model.embed(inp), h)
        logits = model.fc(out[:, -1, :])
        next_token = logits.argmax(1).unsqueeze(0)
        generated.append(next_token.item())
        inp = next_token

print("\nGenerated sequence:", generated)

plt.plot(losses)
plt.title("Autoregressive Model (LSTM) Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
