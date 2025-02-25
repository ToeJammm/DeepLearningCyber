import torch
import torchvision
import torchvision.transforms as transforms

def get_mnist_loaders(batch_size=64):
    print("Grabbing MNIST loaders")
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.RandomRotation(degrees=70),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ]),
        download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1325,), std=(0.3105,))
        ]),
        download=True
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_cifar10_loaders(batch_size=64):
    print("Grabbing Cifar-10 loaders")
    transform_train = transforms.Compose([
    transforms.Resize(64),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  # Randomly crop to 80-100% of original
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),  # Small rotations to increase variability
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader