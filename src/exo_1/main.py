import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from MLP_Architecture import MLPClassier

LEARNING_RATE = 0.01
BATCH_SIZE = 128
EPOCH_NUMBER = 20


def main():
    # Display model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLPClassier().to(device)
    print(f"model: {model}")

    # Display total parameters
    model_total_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {model_total_params}")

    # Display total trainable parameters
    model_total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"trainable: {model_total_trainable_params}")

    # Loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]
    )
    target_transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda y: torch.zeros(10, dtype=torch.float).scatter_(
                    0, torch.tensor(y), value=1
                )
            )
        ]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Training loop
    for epoch in range(EPOCH_NUMBER):
        running_loss = 0.0
        accuracy = 0.0
        print("started epoch")
        for i, data in enumerate(trainloader, 0):
            print("looping")
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            labels = labels.argmax(dim=1)
            outputs = outputs.argmax(dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            corrects = outputs == labels
            accuracy += corrects.sum().float() / float(labels.size(0))

            running_loss += loss.item()

        print("[%d, %5d] accuracy: %.3f" % (epoch + 1, i + 1, accuracy / i))
        print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / i))
