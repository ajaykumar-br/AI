import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = Path('./data')
MODEL_DIR = Path('./models')

# Create directories
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

print(f"Using device: {DEVICE}")

# ============================================================================
# DATA LOADING
# ============================================================================
# TODO: Define data augmentation transforms
# Hint: CIFAR-10 images are 32×32. Consider:
# - Normalization (CIFAR-10 mean/std)
# - Random crops, flips, rotations for training
# - Basic normalization for testing

train_transform = transforms.Compose([
    # ADD AUGMENTATIONS HERE
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 dataset mean
        std=[0.2470, 0.2435, 0.2616]    # CIFAR-10 dataset std
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

# Load datasets
train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# TODO: Design a CNN using Phase 2 knowledge
# Target: Input (batch, 3, 32, 32) → Output (batch, 10) for 10 CIFAR-10 classes
#
# Suggested structure:
# - Start with a few Conv blocks (Conv → ReLU → MaxPool)
# - Gradually increase channels (3 → 32 → 64 → 128)
# - End with flattening and 1-2 fully connected layers
#
# Questions to guide your design:
# Q1: How many Conv layers? (2-5 typically work)
# Q2: What kernel sizes? (3×3 is standard)
# Q3: When/how much to downsample? (2-3 stride-2 layers)
# Q4: How many output channels per layer?

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Block 1: Input (batch, 3, 32, 32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # → (batch, 32, 16, 16)

        # Block 2
        # TODO: Add Conv2d, ReLU, MaxPool2d (or more blocks)
        # Suggested: Conv(32→64), ReLU, MaxPool → (batch, 64, 8, 8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3 (optional)
        # TODO: Add another block if desired

        # Flatten and FC layers
        # TODO: Calculate flattened size after all Conv/Pool layers
        # Hint: After conv blocks above, you should have (batch, 64, 8, 8) = 64*8*8 = 4096
        self.flatten = nn.Flatten()

        # TODO: Add fully connected layers
        # Suggestion: fc1 (flattened_size → 128) → ReLU → fc2 (128 → 10)
        self.fc1 = nn.Linear(4096, 128)  # ADJUST IF YOUR CONV OUTPUTS DIFFER
        self.fc2 = nn.Linear(128, 10)  # 10 CIFAR-10 classes

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2 (TODO)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Block 3 (TODO)

        # Flatten and FC
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)  # ReLU before final FC
        x = self.fc2(x)

        return x

# Instantiate model
model = SimpleCNN().to(DEVICE)
print(f"\nModel architecture:")
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# TRAINING SETUP
# ============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Optional: Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Progress
        if (batch_idx + 1) % 50 == 0:
            print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == '__main__':
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Print stats
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Step scheduler
        scheduler.step()

    # ============================================================================
    # SAVE MODEL
    # ============================================================================
    torch.save(model.state_dict(), MODEL_DIR / 'cifar10_cnn.pth')
    print(f"\nModel saved to {MODEL_DIR / 'cifar10_cnn.pth'}")

    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(test_losses, label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].legend()
    axes[0].grid()

    # Accuracy plot
    axes[1].plot(train_accs, label='Train Accuracy')
    axes[1].plot(test_accs, label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.savefig('cifar10_training_curves.png')
    print("Training curves saved to cifar10_training_curves.png")
    plt.show()

    print(f"\nFinal Test Accuracy: {test_accs[-1]:.2f}%")