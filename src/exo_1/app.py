import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

def main():
    # Display model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLPClassier().to(device)
    print(f'model: {model}')

    # Display total parameters
    model_total_params = sum(p.numel() for p in model.parameters())
    print(f'total parameters: {model_total_params}')

    # Display total trainable parameters
    model_total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f'trainable: {model_total_trainable_params}')

    # Loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    # Training dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
    download=True, transform=transform)

class MLPClassier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.Sigmoid()

        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.Sigmoid()

        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x