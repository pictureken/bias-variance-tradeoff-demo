from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out
