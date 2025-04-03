import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.lenet import LeNet5
from models.resnet18 import ResNet18
from datasets import get_mnist_loaders, get_cifar10_loaders

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["lenet", "resnet18"])
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and test data
    if args.model == "lenet":
        model = LeNet5(num_classes=10)
        model.load_state_dict(torch.load("lenet_mnist_noaug.pth"))
        model.eval().to(device)
        _, test_loader = get_mnist_loaders(batch_size=100, num_workers=2, pin_memory=True)

    elif args.model == "resnet18":
        model = ResNet18(num_classes=10)
        model.load_state_dict(torch.load("resnet18_cifar10_aug.pth"))
        model.eval().to(device)
        _, test_loader = get_cifar10_loaders(batch_size=100, num_workers=2, pin_memory=True)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"\n✅ Clean Test Accuracy: {acc:.2f}%")
