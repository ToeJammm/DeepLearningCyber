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

import torchvision.transforms as transforms

def get_cifar10_loaders(batch_size=128, num_workers=4, pin_memory=True):
    print("Grabbing Cifar-10 loaders")
    
    transform_train = transforms.Compose([
        transforms.Resize(128),  # Upscale to 128x128 for ResNet
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
        transforms.RandomRotation(15),  # Rotate images by ±15 degrees
        transforms.RandomCrop(128, padding=8),  # Randomly crop 128x128 with padding
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add color variation
        transforms.RandomGrayscale(p=0.1),  # Occasionally turn images grayscale
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # Normalize
    ])

    transform_test = transforms.Compose([
        transforms.Resize(128), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, test_loader
