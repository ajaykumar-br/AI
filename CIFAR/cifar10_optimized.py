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
# DATA LOADING WITH ENHANCED AUGMENTATION
# ============================================================================
# OPTIMIZATION 1: More aggressive data augmentation
# This creates more diverse training samples, forcing the model to learn
# more robust features instead of memorizing specific training images
train_transform = transforms.Compose([
    # Original augmentations would go here - now we add MORE:
    transforms.RandomCrop(32, padding=4),           # Random crop with padding
    transforms.RandomHorizontalFlip(p=0.5),         # Random horizontal flip (50% chance)
    transforms.RandomRotation(degrees=15),          # Random rotation up to ±15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Random color variations
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 dataset mean
        std=[0.2470, 0.2435, 0.2616]    # CIFAR-10 dataset std
    )
])

# Test set: minimal augmentation (only normalization)
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

# num_workers=0 to avoid multiprocessing issues on Windows
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# ============================================================================
# MODEL ARCHITECTURE WITH BATCH NORM & DROPOUT
# ============================================================================
# OPTIMIZATION 2: Batch Normalization
# Normalizes layer inputs, stabilizes training, and acts as a regularizer
# Applied after Conv layers (before ReLU for best practice)
#
# OPTIMIZATION 3: Dropout
# Randomly disables neurons during training (prevents co-adaptation)
# - After conv blocks: 0.5 dropout (aggressive, conv already has spatial redundancy)
# - After fc1: 0.3 dropout (moderate, to preserve some capacity)
# - NO dropout in last layer (predictions need to be deterministic at test time)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Block 1: Input (batch, 3, 32, 32)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)           # ← OPTIMIZATION 2: BatchNorm after conv1
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)       # ← OPTIMIZATION 3: Dropout after relu1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # → (batch, 32, 16, 16)

        # Block 2: Conv(32→64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)           # ← OPTIMIZATION 2: BatchNorm after conv2
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)       # ← OPTIMIZATION 3: Dropout after relu2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # → (batch, 64, 8, 8)

        # Flatten and FC layers
        self.flatten = nn.Flatten()

        # FC layers with dropout between them
        self.fc1 = nn.Linear(4096, 128)
        self.dropout_fc = nn.Dropout(p=0.2)     # ← OPTIMIZATION 3: Moderate dropout in FC
        self.fc2 = nn.Linear(128, 10)  # 10 CIFAR-10 classes (NO dropout here - final output)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)                         # ← Apply BatchNorm
        x = self.relu1(x)
        x = self.dropout1(x)                    # ← Apply Dropout during training
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)                         # ← Apply BatchNorm
        x = self.relu2(x)
        x = self.dropout2(x)                    # ← Apply Dropout during training
        x = self.pool2(x)

        # Flatten and FC
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout_fc(x)                  # ← Apply Dropout between FC layers
        x = self.fc2(x)

        return x

# Instantiate model
model = SimpleCNN().to(DEVICE)
print(f"\nModel architecture:")
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# TRAINING SETUP WITH L2 REGULARIZATION
# ============================================================================
criterion = nn.CrossEntropyLoss()

# OPTIMIZATION 4: L2 Regularization via weight_decay
# weight_decay applies L2 penalty to all parameters: Loss = CrossEntropy + weight_decay * ||weights||^2
# This penalizes large weights, forcing simpler models that generalize better
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5)

# Optional: Learning rate scheduler (reduce LR when learning plateaus)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()  # Important: enables dropout and batch norm training behavior
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        # Note: weight_decay is applied automatically by optimizer

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
    model.eval()  # Important: disables dropout, uses running stats for batch norm
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

if __name__ == "__main__":
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    print("\n" + "="*60)
    print("Starting training with regularization optimizations...")
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
    torch.save(model.state_dict(), MODEL_DIR / 'cifar10_cnn_optimized.pth')
    print(f"\nModel saved to {MODEL_DIR / 'cifar10_cnn_optimized.pth'}")

    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(test_losses, label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss over Epochs (Optimized)')
    axes[0].legend()
    axes[0].grid()

    # Accuracy plot
    axes[1].plot(train_accs, label='Train Accuracy')
    axes[1].plot(test_accs, label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy over Epochs (Optimized)')
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.savefig('cifar10_training_curves_optimized.png')
    print("Training curves saved to cifar10_training_curves_optimized.png")
    plt.show()

    print(f"\n{'='*60}")
    print(f"FINAL TEST ACCURACY: {test_accs[-1]:.2f}%")
    print(f"Gap between train/test (overfitting): {train_accs[-1] - test_accs[-1]:.2f}%")
    print(f"{'='*60}")