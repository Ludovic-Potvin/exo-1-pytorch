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
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    # Target
    target_transform = transforms.Compose([
        transforms.Lambda(
            lambda y: torch.zeros(10, dtype=torch.float)\
                .scatter_(0, torch.tensor(y), value=1)
        )
    ])

    # Training dataset
    print("before dataset...")
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
    download=True, transform=transform, target_transform=target_transform)
    
    # Train loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    # Training loop
    for epoch in range(20):
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

            corrects = (outputs == labels)
            accuracy += corrects.sum().float() / float(labels.size(0))

            running_loss += loss.item()

        print('[%d, %5d] accuracy: %.3f' % (epoch + 1, i + 1, accuracy /
        i))
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss /
        i))


class MLPClassier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 784)
        self.act1 = nn.Tanh()

        self.hidden2 = nn.Linear(784, 500)
        self.act2 = nn.Tanh()

        self.output = nn.Linear(500, 10)
        self.act_output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x