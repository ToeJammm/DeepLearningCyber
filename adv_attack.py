import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.lenet import LeNet5
from models.resnet18 import ResNet18
from datasets import get_mnist_loaders, get_cifar10_loaders
import matplotlib.pyplot as plt



def PGD(x, y, model, loss_fn, niter, epsilon, stepsize, randinit=True):
    x_adv = x.clone().detach().to(x.device)
    if randinit:
        x_adv += torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    x_adv.requires_grad = True

    for _ in range(niter):
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)
        model.zero_grad()
        loss.backward()

        grad = x_adv.grad.data
        x_adv = x_adv + stepsize * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)  # Project to L_inf ball
        x_adv = torch.clamp(x_adv, 0, 1).detach()
        x_adv.requires_grad = True
        # print(f"Mean grad: {grad.abs().mean().item():.6f}")

    return x_adv


def plot_accuracy(x_vals, acc_vals, xlabel, title):
    plt.figure()
    plt.plot(x_vals, acc_vals, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel("Adversarial Accuracy (%)")
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["lenet", "resnet18"])
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    


    # Load model
    if args.model == "lenet":
        model = LeNet5(num_classes=10)
        model.load_state_dict(torch.load("lenet_mnist_advTrain.pth"))
        model.eval().to(device)
        _, test_loader = get_mnist_loaders(batch_size=100, num_workers=2, pin_memory=True)
        epsilon = 0.3
    elif args.model == "resnet18":
        model = ResNet18(num_classes=10)
        model.load_state_dict(torch.load("resnet18_cifar10_advTrain.pth"))
        model.eval().to(device)
        _, test_loader = get_cifar10_loaders(batch_size=100, num_workers=2, pin_memory=True)
        epsilon = 0.03

    stepsize = 2/255
    loss_fn = nn.CrossEntropyLoss()

    # Define all the iteration values to test
    iteration_list = [1, 2, 3, 4, 5, 10, 20, 30, 40, 80]
    epsilon_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    niter_acc = []
    epsilon_acc = []

    print("\n=== PGD Hyperparameter Sweep ===")
    print(f"Model:   {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Epsilon: {epsilon}")
    # print(f"Niter: {5}")
    print(f"Stepsize: {stepsize}")
    print("===============================\n")

    for niter in iteration_list:
        correct = 0
        total = 0

        print(f">>> Running PGD with niter = {niter}")
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            adv_images = PGD(images, labels, model, loss_fn,
                             niter=niter, epsilon=epsilon,
                             stepsize=stepsize, randinit=True)

            with torch.no_grad():
                outputs = model(adv_images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        adv_acc = 100 * correct / total
        
        niter_acc.append(adv_acc)
        print(f"Adversarial Accuracy (niter={niter}): {adv_acc:.2f}%\n")
        
    plot_accuracy(iteration_list, niter_acc, "Iterations", "Accuracy")



#python adv_attack.py --model lenet --dataset mnist
#python adv_attack.py --model resnet18 --dataset cifar10
