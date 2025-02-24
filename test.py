import torch
from torchvision import datasets, transforms
from datasets import get_cifar10_loaders, get_mnist_loaders
import matplotlib.pyplot as plt
import numpy as np


all_models = list_models()
classification_models = list_models(module=torchvision.models)

# cifar10_train, cifar10_test = get_cifar10_loaders()

## shows cifar pictures

# for images, labels in cifar10_train:
#     print("Batch of images shape:", images.shape)  # (batch_size, 3, 32, 32)
#     print("Batch of labels shape:", labels.shape)  # (batch_size,)
#     print("Labels:", labels.numpy())  # Print actual class labels
#     break  # Stop after one batch


# # Get one batch of images and labels
# images, labels = next(iter(cifar10_train))

# # Convert tensor images to numpy for visualization
# images = images.numpy().transpose((0, 2, 3, 1))  # Change shape from (batch, C, H, W) to (batch, H, W, C)

# # Define class labels for CIFAR-10
# cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
#                     'dog', 'frog', 'horse', 'ship', 'truck']

# # Plot some images
# fig, axes = plt.subplots(3, 5, figsize=(10, 6))  # Create a 3x5 grid of images
# axes = axes.flatten()

# for i in range(15):  # Show first 15 images
#     img = images[i]
#     label = labels[i].item()
    
#     axes[i].imshow(img)
#     axes[i].set_title(cifar10_classes[label])
#     axes[i].axis("off")  # Hide axis

# plt.tight_layout()
# plt.show()

# # Load MNIST data
# mnist_train, mnist_test = get_mnist_loaders()

# # Print one batch of training data
# for images, labels in mnist_train:
#     print("Batch of images shape:", images.shape)  # (batch_size, 1, 32, 32)
#     print("Batch of labels shape:", labels.shape)  # (batch_size,)
#     print("Labels:", labels.numpy())  # Print actual class labels
#     break  # Stop after one batch

# # Get one batch of images and labels
# images, labels = next(iter(mnist_train))

# # Convert tensor images to numpy for visualization
# images = images.numpy().squeeze(1)  # Remove the single color channel (batch, 1, H, W) -> (batch, H, W)

# # Plot some images
# fig, axes = plt.subplots(3, 5, figsize=(10, 6))  # Create a 3x5 grid of images
# axes = axes.flatten()

# for i in range(15):  # Show first 15 images
#     img = images[i]
#     label = labels[i].item()
    
#     axes[i].imshow(img, cmap="gray")  # MNIST images are grayscale
#     axes[i].set_title(f"Label: {label}")
#     axes[i].axis("off")  # Hide axis

# plt.tight_layout()
# plt.show()