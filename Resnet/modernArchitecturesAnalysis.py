"""
Project 3: Modern Architectures Analysis
Explore VGG, ResNet, and MobileNet architectures
- Understand design choices (depth, skip connections, efficiency)
- Compare models (parameters, FLOPs, accuracy, inference speed)
- Learn why different architectures exist
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time
from pathlib import Path
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
DATA_DIR = Path('./data')

print(f"Using device: {DEVICE}\n")

# ============================================================================
# DATA LOADING
# ============================================================================
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================================
# ARCHITECTURE 1: VGG16
# ============================================================================
print("="*70)
print("ARCHITECTURE 1: VGG16 (Very Deep Convolutional Networks)")
print("="*70)
print("\nKey Ideas:")
print("  • Simplicity: Only 3×3 convolutions (vs 7×7, 5×5 in older models)")
print("  • Depth over Width: 16 layers deep (vs fewer wider layers)")
print("  • Uniform architecture: Repeating Conv blocks followed by MaxPool")
print("  • Problem: Very deep → vanishing gradients, lots of parameters")
print()

# TODO: Understand VGG architecture
# VGG16 consists of:
# - 5 blocks of Conv layers (each block has multiple 3×3 convs)
# - Each block followed by MaxPool (2×2) to reduce spatial dims
# - Then 3 fully connected layers (4096, 4096, num_classes)
#
# Why 3×3 kernels? Two 3×3 convs = same receptive field as one 5×5,
# but fewer parameters: 2*(3*3) vs 1*(5*5) = 18 vs 25
#
# Blocks progression: 64 → 128 → 256 → 512 → 512 channels

vgg16 = models.vgg16(pretrained=True)
print("VGG16 Architecture:")
print(vgg16)

total_params_vgg = sum(p.numel() for p in vgg16.parameters())
print(f"\nTotal Parameters: {total_params_vgg:,}")
print(f"Trainable Parameters: {sum(p.numel() for p in vgg16.parameters() if p.requires_grad):,}")

# TODO: Analyze VGG layers
# Count conv layers vs FC layers
conv_params = sum(p.numel() for p in vgg16.features.parameters())
fc_params = sum(p.numel() for p in vgg16.classifier.parameters())
print(f"\nConv Layers: {conv_params:,} params ({conv_params/total_params_vgg*100:.1f}%)")
print(f"FC Layers: {fc_params:,} params ({fc_params/total_params_vgg*100:.1f}%)")
print("^ Note: Most params in FC layers! (inefficient)")

# ============================================================================
# ARCHITECTURE 2: ResNet50
# ============================================================================
print("\n" + "="*70)
print("ARCHITECTURE 2: ResNet50 (Residual Networks)")
print("="*70)
print("\nKey Ideas:")
print("  • Skip Connections: Add input to output (identity shortcut)")
print("  • Solves vanishing gradient: Can train very deep (50+ layers)")
print("  • Bottleneck blocks: 1×1 reduce → 3×3 compute → 1×1 expand")
print("  • Efficiency: Fewer parameters than VGG despite being deeper")
print()

# TODO: Understand ResNet architecture
# ResNet50 consists of:
# - Conv layer (7×7) + MaxPool
# - 4 residual blocks (layer1, layer2, layer3, layer4)
# - Each block has bottleneck structure: Conv(1×1) → Conv(3×3) → Conv(1×1)
#
# Skip connections allow gradients to flow directly, enabling deep networks
# Formula: y = F(x) + x (where F is the residual function)
#
# Bottleneck reduces computation: 64→64 (reduce) → 64→64 (compute) → 64→256 (expand)
# vs standard: 256×256 convolution

resnet50 = models.resnet50(pretrained=True)
print("ResNet50 Architecture (simplified view):")
print(f"  - Input: 3×224×224")
print(f"  - Conv1 (7×7, stride=2): 3→64")
print(f"  - MaxPool: 64×112×112 → 64×56×56")
print(f"  - Layer1 (×3 blocks): 64→256 channels")
print(f"  - Layer2 (×4 blocks): 256→512 channels")
print(f"  - Layer3 (×6 blocks): 512→1024 channels")
print(f"  - Layer4 (×3 blocks): 1024→2048 channels")
print(f"  - AvgPool + FC: 2048 → num_classes")
print()

total_params_resnet = sum(p.numel() for p in resnet50.parameters())
print(f"Total Parameters: {total_params_resnet:,}")

# TODO: Compare bottleneck efficiency
# A bottleneck block (in, mid, out):
# Params = (1×1×in×mid) + (3×3×mid×mid) + (1×1×mid×out) + biases
# Example with in=256, mid=64, out=256:
# = (256×64) + (3×3×64×64) + (64×256) = 16.4K + 36.9K + 16.4K = ~70K params
# vs standard Conv(256, 256, 3×3) = 256×256×3×3 = 589K params
# Reduction: 589K / 70K ≈ 8.4x fewer parameters!

params_bottleneck_example = (1*1*256*64) + (3*3*64*64) + (1*1*64*256)
params_standard_example = 256*256*3*3
reduction_ratio = params_standard_example / params_bottleneck_example
print(f"\nBottleneck Efficiency Example:")
print(f"  Standard 3×3 Conv(256→256): {params_standard_example:,} params")
print(f"  Bottleneck (256→64→256): {params_bottleneck_example:,} params")
print(f"  Reduction: {reduction_ratio:.1f}x fewer parameters!")

# ============================================================================
# ARCHITECTURE 3: MobileNetV2
# ============================================================================
print("\n" + "="*70)
print("ARCHITECTURE 3: MobileNetV2 (Mobile & Efficient Networks)")
print("="*70)
print("\nKey Ideas:")
print("  • Mobile-first: Designed for phones, edge devices, IoT")
print("  • Depthwise Separable Convolutions: Split into spatial & channel operations")
print("  • Inverted Bottlenecks: Expand first, then compress (opposite of ResNet)")
print("  • Low latency & low memory: Trade accuracy for speed/efficiency")
print()

# TODO: Understand MobileNetV2 architecture
# MobileNetV2 consists of:
# - Linear Bottleneck blocks (inverted residual blocks)
# - Structure: Conv(1×1 expand) → DW Conv(3×3 depthwise) → Conv(1×1 compress)
# - Uses Depthwise Separable Convolution to reduce params
#
# Depthwise Separable = Depthwise (spatial) + Pointwise (channel)
# Depthwise: Apply one kernel per input channel (in_channels kernels)
# Pointwise: 1×1 conv to mix channels
# Reduction: Standard(in×out×3×3) vs DW+PW(in + in×out)
#
# Example: 32 input channels, 64 output channels, 3×3 kernel
# Standard: 32×64×3×3 = 18.4K params
# DW+PW: (32×1×3×3) + (32×64×1×1) = 288 + 2.048K = 2.3K params
# Reduction: 8x fewer parameters!

mobilenet = models.mobilenet_v2(pretrained=True)
print("MobileNetV2 Architecture (simplified view):")
print(f"  - Input: 3×224×224")
print(f"  - Conv1 (3×3, stride=2): 3→32")
print(f"  - 17 Inverted Bottleneck layers (expanding then compressing)")
print(f"  - Channel progression: 16→24→32→64→96→160→320")
print(f"  - Conv Last (1×1): 320→1280")
print(f"  - AvgPool + Linear: 1280 → num_classes")
print()

total_params_mobile = sum(p.numel() for p in mobilenet.parameters())
print(f"Total Parameters: {total_params_mobile:,}")

# TODO: Compare depthwise separable efficiency
params_dw_pw = (32*1*3*3) + (32*64*1*1)
params_standard_mobile = 32*64*3*3
reduction_ratio_mobile = params_standard_mobile / params_dw_pw
print(f"\nDepthwise Separable Efficiency Example:")
print(f"  Standard Conv(32→64, 3×3): {params_standard_mobile:,} params")
print(f"  Depthwise Separable: {params_dw_pw:,} params")
print(f"  Reduction: {reduction_ratio_mobile:.1f}x fewer parameters!")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n" + "="*70)
print("ARCHITECTURE COMPARISON")
print("="*70)

vgg16.fc = nn.Linear(4096, 10)
resnet50.fc = nn.Linear(2048, 10)
mobilenet.classifier[-1] = nn.Linear(1280, 10)

vgg16 = vgg16.to(DEVICE)
resnet50 = resnet50.to(DEVICE)
mobilenet = mobilenet.to(DEVICE)

def count_flops_approx(model, input_shape=(1, 3, 224, 224)):
    """Rough estimate of FLOPs (not precise, for comparison only)"""
    total_params = sum(p.numel() for p in model.parameters())
    # Rough heuristic: 1 forward pass ≈ 2x params FLOPs
    return total_params * 2 / 1e9  # in GigaFLOPs

def measure_inference_time(model, test_loader, device, num_batches=10):
    """Measure average inference time"""
    model.eval()
    times = []

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= num_batches:
                break

            data = data.to(device)

            start = time.time()
            _ = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)

    return sum(times) / len(times)

print("\n{:<20} {:<15} {:<15} {:<15}".format("Model", "Parameters", "FLOPs (est.)", "Inference (ms)"))
print("-" * 65)

# VGG16
vgg_params = sum(p.numel() for p in vgg16.parameters())
vgg_flops = count_flops_approx(vgg16)
print(f"Measuring VGG16 inference time...")
vgg_time = measure_inference_time(vgg16, test_loader, DEVICE) * 1000
print(f"{'VGG16':<20} {vgg_params:>13,} {vgg_flops:>13.2f}G {vgg_time:>13.2f}ms")

# ResNet50
resnet_params = sum(p.numel() for p in resnet50.parameters())
resnet_flops = count_flops_approx(resnet50)
print(f"Measuring ResNet50 inference time...")
resnet_time = measure_inference_time(resnet50, test_loader, DEVICE) * 1000
print(f"{'ResNet50':<20} {resnet_params:>13,} {resnet_flops:>13.2f}G {resnet_time:>13.2f}ms")

# MobileNetV2
mobile_params = sum(p.numel() for p in mobilenet.parameters())
mobile_flops = count_flops_approx(mobilenet)
print(f"Measuring MobileNetV2 inference time...")
mobile_time = measure_inference_time(mobilenet, test_loader, DEVICE) * 1000
print(f"{'MobileNetV2':<20} {mobile_params:>13,} {mobile_flops:>13.2f}G {mobile_time:>13.2f}ms")

# ============================================================================
# KEY LEARNINGS
# ============================================================================
print("\n" + "="*70)
print("KEY ARCHITECTURAL INSIGHTS")
print("="*70)

print("""
1. VGG16 - The Simple Deep Approach
   ✓ Very simple, easy to understand
   ✗ Massive parameter count (138M)
   ✗ Most params in FC layers (inefficient)
   ✗ Slow inference
   → Historical significance: Showed that depth matters

2. ResNet50 - The Game Changer
   ✓ Skip connections solve vanishing gradient
   ✓ Can train very deep (50+ layers) successfully
   ✓ Bottleneck design is parameter-efficient
   ✓ Good balance of accuracy vs efficiency
   ✗ Still requires significant compute
   → Modern standard: Used in many applications

3. MobileNetV2 - The Efficient Design
   ✓ 1/50th the parameters of VGG (3.5M vs 138M)
   ✓ Depthwise separable convolutions are clever
   ✓ Inverted bottleneck design (expand-compress)
   ✓ Fast inference (suitable for mobile)
   ✗ Slightly lower accuracy on complex tasks
   → Real-world deployment: Mobile phones, IoT, edge devices

DESIGN PRINCIPLE EVOLUTION:
  VGG (2014): Go deeper with simple ops
    ↓
  ResNet (2015): Use skip connections to go even deeper
    ↓
  MobileNet (2017): Optimize for efficiency (depthwise separable)
    ↓
  Vision Transformer (2020): Abandon convolutions entirely (later study)
""")

print("\n" + "="*70)
print("YOUR ANALYSIS TASKS")
print("="*70)
print("""
TODO 1: Architecture Comparison
  - Fill the comparison table above
  - Which model is best for: accuracy? speed? mobile? large scale?

TODO 2: Understand Key Innovations
  - Skip connections in ResNet: Why do they solve vanishing gradient?
  - Bottleneck blocks in ResNet: Why is 1×1→3×3→1×1 efficient?
  - Depthwise separable in MobileNet: How does splitting convolutions reduce params?

TODO 3: Depth vs Width Trade-off
  - VGG went deep with many channels (16 layers, 512 channels)
  - ResNet went even deeper with bottleneck efficiency (50 layers, bottleneck design)
  - What's the relationship between depth and expressiveness?

TODO 4: Modern Trends
  - Why did MobileNet become important after ResNet?
  - What markets/devices drove the need for efficiency?
  - How do Vision Transformers (ViT) change this paradigm? (Study next)
""")

# ============================================================================
# VISUALIZATION: Parameter Distribution
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# VGG16
vgg_fc = fc_params
vgg_conv = conv_params
axes[0].bar(['Conv Layers', 'FC Layers'], [vgg_conv/1e6, vgg_fc/1e6], color=['#3498db', '#e74c3c'])
axes[0].set_ylabel('Parameters (Millions)')
axes[0].set_title('VGG16: Heavy FC Layers')
axes[0].set_ylim(0, 140)
for i, v in enumerate([vgg_conv/1e6, vgg_fc/1e6]):
    axes[0].text(i, v + 2, f'{v:.1f}M', ha='center', fontweight='bold')

# ResNet50
resnet_conv = sum(p.numel() for p in resnet50.conv1.parameters()) + \
              sum(p.numel() for p in resnet50.layer1.parameters()) + \
              sum(p.numel() for p in resnet50.layer2.parameters()) + \
              sum(p.numel() for p in resnet50.layer3.parameters()) + \
              sum(p.numel() for p in resnet50.layer4.parameters())
resnet_fc = sum(p.numel() for p in resnet50.fc.parameters())
axes[1].bar(['Conv Layers', 'FC Layers'], [resnet_conv/1e6, resnet_fc/1e6], color=['#3498db', '#e74c3c'])
axes[1].set_ylabel('Parameters (Millions)')
axes[1].set_title('ResNet50: Balanced Design')
axes[1].set_ylim(0, 140)
for i, v in enumerate([resnet_conv/1e6, resnet_fc/1e6]):
    axes[1].text(i, v + 2, f'{v:.1f}M', ha='center', fontweight='bold')

# MobileNetV2
mobile_features = sum(p.numel() for p in mobilenet.features.parameters())
mobile_classifier = sum(p.numel() for p in mobilenet.classifier.parameters())
axes[2].bar(['Features', 'Classifier'], [mobile_features/1e6, mobile_classifier/1e6], color=['#3498db', '#e74c3c'])
axes[2].set_ylabel('Parameters (Millions)')
axes[2].set_title('MobileNetV2: Lightweight Design')
axes[2].set_ylim(0, 140)
for i, v in enumerate([mobile_features/1e6, mobile_classifier/1e6]):
    axes[2].text(i, v + 2, f'{v:.1f}M', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('architecture_comparison_params.png', dpi=100, bbox_inches='tight')
print("Parameter distribution chart saved to architecture_comparison_params.png")
plt.show()

print("\n✓ Analysis complete! Review the comparison table and TODOs above.")