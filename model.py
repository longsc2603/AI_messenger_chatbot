import torch.nn as nn


class MyNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(MyNeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        logits = self.layer1(x)
        logits = self.relu(logits)
        logits = self.layer2(logits)
        logits = self.relu(logits)
        logits = self.layer3(logits)
        logits = self.relu(logits)
        return logits
