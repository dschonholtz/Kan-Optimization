from efficient_kan import KAN
from efficient_kan.kanvolution import Kanv2d

# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# plotting imports
import time
import json
import matplotlib.pyplot as plt

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 16)  # First linear layer
        self.relu = nn.ReLU()            # ReLU activation
        self.fc2 = nn.Linear(16, 10)     # Second linear layer (output layer)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load MNIST
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
valset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# Define model
model = KAN([28 * 28, 8, 10]) #SimpleMLP() # kanv2d?

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
model.to(device)
# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# Define loss
criterion = nn.CrossEntropyLoss()
num_epochs = 5
start_time = time.time()
for epoch in range(num_epochs):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 28 * 28).to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(-1, 28 * 28).to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )


    results_file = 'training_results.json'
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    title = "KANvolution kernel of 3 stride 1"
    if title not in results:
        results[title] = []

    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time} seconds")
    results[title].append({
        'epoch': epoch + 1,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'training_time': total_training_time
    })
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)


training_times = [r['training_time'] for r in results[title]]
val_losses = [r['val_loss'] for r in results[title]]
val_accuracies = [r['val_accuracy'] for r in results[title]]
epochs = [r['epoch'] for r in results[title]]

plt.figure(figsize=(15, 5))

# Plot Validation Loss
plt.subplot(1, 3, 1)
for title in results:
    epochs = [r['epoch'] for r in results[title]]
    val_losses = [r['val_loss'] for r in results[title]]
    plt.plot(epochs, val_losses, label=title)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Over Epochs')
plt.legend()

# Plot Validation Accuracy
plt.subplot(1, 3, 2)
for title in results:
    epochs = [r['epoch'] for r in results[title]]
    val_accuracies = [r['val_accuracy'] for r in results[title]]
    plt.plot(epochs, val_accuracies, label=title)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.legend()

# Plot Training Time
plt.subplot(1, 3, 3)
for title in results:
    epochs = [r['epoch'] for r in results[title]]
    training_times = [r['training_time'] for r in results[title]]
    plt.plot(epochs, training_times, label=title)
plt.xlabel('Epoch')
plt.ylabel('Time (s)')
plt.title('Training Time Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()