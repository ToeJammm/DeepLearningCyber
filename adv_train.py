
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datasets import get_mnist_loaders, get_cifar10_loaders
from models.lenet import LeNet5
from models.vgg import VGG16
from models.resnet18 import ResNet18
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from adv_attack import PGD


def train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    model.to(device)
    model.train()

    scaler = torch.amp.GradScaler() if device.type == "cuda" else None
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    epochs_list = []
    adv_accuracies = []

    for epoch in range(num_epochs):
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            model.eval()
            adv_images = PGD(images, labels, model, criterion,
                             niter=5, epsilon=0.03,
                             stepsize=2/255, randinit=True)

            optimizer.zero_grad()
            if device.type == "cuda":
                with torch.amp.autocast(device_type=device_type):
                    outputs_adv = model(adv_images)
                    outputs_clean = model(images)
                loss = 0.5 * criterion(outputs_clean, labels) + 0.5 * criterion(outputs_adv, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: 
                outputs_adv = model(adv_images)
                outputs_clean = model(images)
                loss = 0.5 * criterion(outputs_clean, labels) + 0.5 * criterion(outputs_adv, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs_clean, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Compute training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # Evaluate on the test set every epoch
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        adv_loss, adv_accuracy = evaluate_adv(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        adv_accuracies.append(adv_accuracy)
        epochs_list.append(epoch + 1)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

        # Step the scheduler using test loss
        scheduler.step()

    return train_losses, test_losses, train_accuracies, test_accuracies, adv_accuracies, epochs_list



def evaluate(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    test_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate_adv(model, test_loader, criterion, device):
    model.eval()
    adv_correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            adv_images = PGD(images, labels, model, criterion, niter=5, epsilon=0.03, stepsize=2/255, randinit=False)
            outputs = model(adv_images)
            _, predicted = torch.max(outputs, 1)
            adv_correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * adv_correct / total


def plot_metrics(epochs, train_values, test_values, ylabel, title):
    plt.figure()
    plt.plot(epochs, train_values, label="Train", marker='o')
    plt.plot(epochs, test_values, label="Test", marker='s')
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train a neural network model")
    parser.add_argument("--model", type=str, required=True, choices=["lenet", "vgg16", "resnet18"], help="Choose a model")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10"], help="Choose dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")



    if args.dataset == "mnist":
        train_loader, test_loader = get_mnist_loaders(batch_size=64, num_workers=4, pin_memory=True)
    elif args.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(batch_size=64, num_workers=4, pin_memory=True)


    if args.model == "lenet":
        model = LeNet5(num_classes=10)
    elif args.model == "vgg16":
        model = VGG16(num_classes=10)
    elif args.model == "resnet18":
        model = ResNet18(num_classes=10)

    

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)
    
    criterion = nn.CrossEntropyLoss() 
    # optimizer = optim.SGD(model.parameters(), lr=0.01, #good for Resnet18
    #                   momentum=0.9, weight_decay=5e-4)


    # Train the model
    train_losses, test_losses, train_accuracies, test_accuracies, epochs_list, adv_accuracies = train(
        model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs=60 
    )

    plot_metrics(epochs_list, train_accuracies, test_accuracies, "Accuracy (%)", f"{args.model} Accuracy Over Epochs")
    # plot_metrics(epochs_list, train_losses, test_losses, "Loss", f"{args.model} Loss Over Epochs")
    plot_metrics(epochs_list, test_accuracies, adv_accuracies, "Accuracy (%)", f"{args.model} Adversarial Accuracy Over Epochs")


    save_model = input("Do you want to save the trained model? (y/n): ").strip().lower()
    if save_model == 'y':
        model_path = f"{args.model}_{args.dataset}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        print("Model not saved.")

if __name__ == "__main__":
    main()




