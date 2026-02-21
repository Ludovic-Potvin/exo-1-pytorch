import torch
import os
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

import torchvision
import torchvision.transforms as transforms

from .CNN_Architecture import CNNClassifier
from .other_tools import get_model_information
from .train_process import train_model
from .test_process import test_model

from .patch_embeding.vit import ViT

LEARNING_RATE = 0.005
BATCH_SIZE = 8
EPOCH_NUMBER = 3
INPUT_SHAPE = 784
DATASET_PATH = "./data/Images"
RESULT_PATH = "./results"
SEED = 42


def main():
    # Transformer
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
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

    #model = CNNClassifier(image_channel, class_number).to(device)
    model = ViT(num_classes=7).to(device)

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
    
    # Write to file
    now = datetime.now()
    my_folder_name = now.strftime("%Y-%m-%d_%H" + "h" + "%M" + "min" + "%S" +
    "sec")
    os.makedirs(os.path.join(RESULT_PATH, my_folder_name))
    print("\nResult folder created")
    txt_file = open(os.path.join(os.path.join(RESULT_PATH, my_folder_name),
    "Results.txt"), "a")
    txt_file.write("Hyperparameters\n")
    txt_file.write(f"Epoch number: {EPOCH_NUMBER}\n")
    txt_file.write(f"Batch size: {BATCH_SIZE}\n")
    txt_file.write(f"Learning rate: {LEARNING_RATE}\n")