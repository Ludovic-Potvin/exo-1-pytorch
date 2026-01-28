import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

import torchvision
import torchvision.transforms as transforms

from .CNN_Architecture import CNNClassifier
from .other_tools import get_model_information
from .train_process import train_model
from .test_process import test_model

LEARNING_RATE = 0.005
BATCH_SIZE = 16
EPOCH_NUMBER = 20
INPUT_SHAPE = 784
DATASET_PATH = "./data/Images"
SEED = 42


def main():
    # Transformer
    transform = transforms.Compose(
        [transforms.Resize((240, 240)), transforms.ToTensor()]
    )
    target_transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda y: torch.zeros(7, dtype=torch.float).scatter_(
                    0, torch.tensor(y), value=1
                )
            )
        ]
    )

    # Dataset
    dataset = torchvision.datasets.ImageFolder(
        root=DATASET_PATH,
        transform=transform,
        target_transform=target_transform,
    )

    classes = dataset.classes
    class_number = len(list(dataset.classes))

    generator1 = torch.Generator().manual_seed(SEED)
    train_set, validation_set, test_set = random_split(
        dataset, [0.7, 0.15, 0.15], generator=generator1
    )

    # Loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=True
    )

    # Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    images, labels = next(iter(train_loader))
    image_shape = list(images.data.shape)
    image_channel = image_shape[1]

    model = CNNClassifier(image_channel, class_number).to(device)

    get_model_information(model)

    # Loss function
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Training
    model = train_model(
        EPOCH_NUMBER,
        train_loader,
        validation_loader,
        model,
        optimizer,
        loss_function,
        device,
    )
    test_model(test_loader, model, loss_function, device, classes)