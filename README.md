# rithik-roshan
A curated collection of projects, experiments, and learning resources from my journey as an AI &amp; Machine Learning Engineering student. Includes implementations of core algorithms, deep learning models, data analysis pipelines, and real-world applications built using Python, TensorFlow, PyTorch, scikit-learn, and more
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Load dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
    transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)

# Simple Neural Net
class Net(nn.Module):
    def __init__(self): 
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x): return self.fc(x)

# Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(1):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()

print("✅ Model trained on MNIST!")
