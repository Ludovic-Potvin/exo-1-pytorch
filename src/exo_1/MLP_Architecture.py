import torch.nn as nn


class MLPClassier(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.hidden1 = nn.Linear(input_shape, 784)
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
