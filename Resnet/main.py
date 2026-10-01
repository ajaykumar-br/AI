"""
ResNet50 Transfer Learning: Compare 3 Strategies
1. Feature Extraction (freeze early layers)
2. Fine-tuning with Uniform Learning Rate
3. Fine-tuning with Discriminative Learning Rates
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
DATA_DIR = Path('./data')
MODEL_DIR = Path('./models')

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

print(f"Using device: {DEVICE}\n")

# ============================================================================
# DATA LOADING
# ============================================================================
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

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
# APPROACH 1: FEATURE EXTRACTION (Freeze all except final layer)
# ============================================================================
print("="*70)
print("APPROACH 1: FEATURE EXTRACTION (Freeze early layers)")
print("="*70)

model1 = models.resnet50(pretrained=True)
in_features = model1.fc.in_features
model1.fc = nn.Linear(in_features, 10)
model1 = model1.to(DEVICE)

# FREEZE all layers except final fc
for param in model1.parameters():
    param.requires_grad = False

# UNFREEZE only the final layer
for param in model1.fc.parameters():
    param.requires_grad = True

print(f"Trainable parameters: {sum(p.numel() for p in model1.parameters() if p.requires_grad)}")
print(f"Frozen parameters: {sum(p.numel() for p in model1.parameters() if not p.requires_grad)}\n")

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam([p for p in model1.parameters() if p.requires_grad], lr=LEARNING_RATE)
scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=5, gamma=0.1)

train_losses1, train_accs1 = [], []
test_losses1, test_accs1 = [], []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss, train_acc = train_epoch(model1, train_loader, criterion, optimizer1, DEVICE)
    train_losses1.append(train_loss)
    train_accs1.append(train_acc)

    test_loss, test_acc = evaluate(model1, test_loader, criterion, DEVICE)
    test_losses1.append(test_loss)
    test_accs1.append(test_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n")
    scheduler1.step()

final_acc1 = test_accs1[-1]
final_gap1 = train_accs1[-1] - test_accs1[-1]
print(f"✓ APPROACH 1 - Final Test Accuracy: {final_acc1:.2f}%, Gap: {final_gap1:.2f}%\n")

# ============================================================================
# APPROACH 2: FINE-TUNING WITH UNIFORM LEARNING RATE (current approach)
# ============================================================================
print("="*70)
print("APPROACH 2: FINE-TUNING (Uniform LR = 0.0001)")
print("="*70)

model2 = models.resnet50(pretrained=True)
model2.fc = nn.Linear(in_features, 10)
model2 = model2.to(DEVICE)

# Keep all layers trainable (default)
print(f"Trainable parameters: {sum(p.numel() for p in model2.parameters() if p.requires_grad)}\n")

optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE * 0.1)
scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.1)

train_losses2, train_accs2 = [], []
test_losses2, test_accs2 = [], []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss, train_acc = train_epoch(model2, train_loader, criterion, optimizer2, DEVICE)
    train_losses2.append(train_loss)
    train_accs2.append(train_acc)

    test_loss, test_acc = evaluate(model2, test_loader, criterion, DEVICE)
    test_losses2.append(test_loss)
    test_accs2.append(test_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n")
    scheduler2.step()

final_acc2 = test_accs2[-1]
final_gap2 = train_accs2[-1] - test_accs2[-1]
print(f"✓ APPROACH 2 - Final Test Accuracy: {final_acc2:.2f}%, Gap: {final_gap2:.2f}%\n")

# ============================================================================
# APPROACH 3: FINE-TUNING WITH DISCRIMINATIVE LEARNING RATES
# ============================================================================
print("="*70)
print("APPROACH 3: FINE-TUNING (Discriminative LRs)")
print("="*70)
print("Early layers: 0.00001 (preserve pre-trained knowledge)")
print("Mid layers: increasing LR")
print("Late layers + FC: 0.001 (adapt to CIFAR-10)\n")

model3 = models.resnet50(pretrained=True)
model3.fc = nn.Linear(in_features, 10)
model3 = model3.to(DEVICE)

# DISCRIMINATIVE LEARNING RATES
# Different LRs for different layer groups
param_groups = [
    {'params': model3.layer1.parameters(), 'lr': 1e-5},      # Very early: freeze-like
    {'params': model3.layer2.parameters(), 'lr': 5e-5},      # Early
    {'params': model3.layer3.parameters(), 'lr': 1e-4},      # Middle
    {'params': model3.layer4.parameters(), 'lr': 5e-4},      # Late (higher LR)
    {'params': model3.fc.parameters(), 'lr': LEARNING_RATE}  # Final layer: highest LR
]

print(f"Trainable parameters: {sum(p.numel() for p in model3.parameters() if p.requires_grad)}")
print(f"Using 5 parameter groups with different learning rates\n")

optimizer3 = optim.Adam(param_groups)
scheduler3 = optim.lr_scheduler.StepLR(optimizer3, step_size=5, gamma=0.1)

train_losses3, train_accs3 = [], []
test_losses3, test_accs3 = [], []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss, train_acc = train_epoch(model3, train_loader, criterion, optimizer3, DEVICE)
    train_losses3.append(train_loss)
    train_accs3.append(train_acc)

    test_loss, test_acc = evaluate(model3, test_loader, criterion, DEVICE)
    test_losses3.append(test_loss)
    test_accs3.append(test_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n")
    scheduler3.step()

final_acc3 = test_accs3[-1]
final_gap3 = train_accs3[-1] - test_accs3[-1]
print(f"✓ APPROACH 3 - Final Test Accuracy: {final_acc3:.2f}%, Gap: {final_gap3:.2f}%\n")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n" + "="*70)
print("COMPARISON OF ALL THREE APPROACHES")
print("="*70)
print(f"\n{'Metric':<30} {'Approach 1 (FE)':<20} {'Approach 2 (Uniform)':<20} {'Approach 3 (Discrim.)':<20}")
print("-"*90)
print(f"{'Final Test Accuracy':<30} {final_acc1:>18.2f}% {final_acc2:>18.2f}% {final_acc3:>18.2f}%")
print(f"{'Train-Test Gap':<30} {final_gap1:>18.2f}% {final_gap2:>18.2f}% {final_gap3:>18.2f}%")
print(f"{'Final Train Accuracy':<30} {train_accs1[-1]:>18.2f}% {train_accs2[-1]:>18.2f}% {train_accs3[-1]:>18.2f}%")

# ============================================================================
# VISUALIZATION: SIDE-BY-SIDE COMPARISON
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Approach 1
axes[0, 0].plot(train_losses1, label='Train', linewidth=2)
axes[0, 0].plot(test_losses1, label='Test', linewidth=2)
axes[0, 0].set_title('Approach 1: Feature Extraction - Loss', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid()

axes[1, 0].plot(train_accs1, label='Train', linewidth=2)
axes[1, 0].plot(test_accs1, label='Test', linewidth=2)
axes[1, 0].set_title('Approach 1: Feature Extraction - Accuracy', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy (%)')
axes[1, 0].legend()
axes[1, 0].grid()

# Approach 2
axes[0, 1].plot(train_losses2, label='Train', linewidth=2)
axes[0, 1].plot(test_losses2, label='Test', linewidth=2)
axes[0, 1].set_title('Approach 2: Uniform LR - Loss', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid()

axes[1, 1].plot(train_accs2, label='Train', linewidth=2)
axes[1, 1].plot(test_accs2, label='Test', linewidth=2)
axes[1, 1].set_title('Approach 2: Uniform LR - Accuracy', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].legend()
axes[1, 1].grid()

# Approach 3
axes[0, 2].plot(train_losses3, label='Train', linewidth=2)
axes[0, 2].plot(test_losses3, label='Test', linewidth=2)
axes[0, 2].set_title('Approach 3: Discriminative LR - Loss', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].legend()
axes[0, 2].grid()

axes[1, 2].plot(train_accs3, label='Train', linewidth=2)
axes[1, 2].plot(test_accs3, label='Test', linewidth=2)
axes[1, 2].set_title('Approach 3: Discriminative LR - Accuracy', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Accuracy (%)')
axes[1, 2].legend()
axes[1, 2].grid()

plt.tight_layout()
plt.savefig('resnet_comparison_all_approaches.png', dpi=100, bbox_inches='tight')
print("Comparison chart saved to resnet_comparison_all_approaches.png")
plt.show()

# ============================================================================
# KEY INSIGHTS
# ============================================================================
print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("\nApproach 1 (Feature Extraction):")
print(f"  - Fastest training (only final layer updates)")
print(f"  - Test Accuracy: {final_acc1:.2f}%")
print(f"  - Best if: Limited computational budget OR small dataset")

print("\nApproach 2 (Uniform Learning Rate):")
print(f"  - All layers trainable with same (lower) learning rate")
print(f"  - Test Accuracy: {final_acc2:.2f}%")
print(f"  - Best if: Good balance between speed and adaptation")

print("\nApproach 3 (Discriminative Learning Rates):")
print(f"  - Early layers learn slowly, late layers learn fast")
print(f"  - Test Accuracy: {final_acc3:.2f}%")
print(f"  - Best if: Maximum performance needed")
print("\n" + "="*70)