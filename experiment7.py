import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

transform = transforms.ToTensor()
full_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_size = int(0.85 * len(full_data))
val_size = len(full_data) - train_size
train_data, val_data = random_split(full_data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(8):
    model.train()
    for x, y in train_loader:
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_correct = 0
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x)
            val_loss += loss_fn(out, y).item()
            val_correct += (out.argmax(1) == y).sum().item()

    print(f"Epoch {epoch+1}  Val Loss: {val_loss/len(val_loader):.4f}  Val Acc: {val_correct/len(val_data)*100:.2f}%")

test_correct = 0
with torch.no_grad():
    for x, y in test_loader:
        test_correct += (model(x).argmax(1) == y).sum().item()

print(f"\nFinal Test Accuracy: {test_correct / len(test_data) * 100:.2f}%")
