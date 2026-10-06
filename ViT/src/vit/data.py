import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(data_dir: str, batch_size: int, num_workers: int = 0):
    # ViTs have no built-in locality bias, so from-scratch training on 50k images
    # leans heavily on augmentation. Crop + flip is the minimum.
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        transforms.RandomErasing(p=0.25),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
