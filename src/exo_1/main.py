import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

import torchvision
import torchvision.transforms as transforms

from .MLP_Architecture import MLPClassier
from .other_tools import get_model_information
from .train_process import train_model
from .test_process import test_model

LEARNING_RATE = 0.01
BATCH_SIZE = 128
EPOCH_NUMBER = 20
INPUT_SHAPE = 784


def main():
    # Display model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLPClassier(INPUT_SHAPE).to(device)

    get_model_information(model)

    # Loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Data Transformation
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

    # Train and validation set
    train_set = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    train_set_shape = list(train_set.data.shape)
    train_set, validation_set = random_split(
        train_set,
        (
            int(train_set_shape[0] * 0.9),
            train_set_shape[0] - int(train_set_shape[0] * 0.9),
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=False
    )

    # Test set
    test_set = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=True
    )

    # Training
    model = train_model(EPOCH_NUMBER,train_loader,validation_loader,model,optimizer,loss_function, device)
    test_model(test_loader, model, loss_function, device)