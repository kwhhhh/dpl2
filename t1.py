import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

# 1. 超参数 -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BATCH_SIZE   = 256
EPOCHS       = 3
LR           = 1e-3
DOWNLOAD     = True        # 第一次运行后改成 False 可省流量

# 2. 数据 --------------------------------------------------------------
# 只用一个非常简单的 ToTensor() 把 HWC PIL Image -> CHW tensor [0,1]
transform = T.Compose([T.ToTensor()])

train_set = torchvision.datasets.MNIST(
        root="./data", train=True,  download=DOWNLOAD, transform=transform)
test_set  = torchvision.datasets.MNIST(
        root="./data", train=False, download=DOWNLOAD, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=2, pin_memory=True)

# 3. 模型 --------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_features=28*28, hidden=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),               # (B,1,28,28) -> (B,784)
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
print(model)

# 4. 损失 & 优化器 ------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 5. 训练 --------------------------------------------------------------
def train_one_epoch(epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=correct/total)
    return running_loss/total, correct/total

# 6. 测试 --------------------------------------------------------------
@torch.no_grad()
def test():
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    acc = correct / total
    print(f"Test accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# 7. 开跑 --------------------------------------------------------------
if __name__ == "__main__":
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(epoch)
        print(f"Epoch {epoch+1} | loss {train_loss:.4f} | acc {train_acc:.4f}")
    test()