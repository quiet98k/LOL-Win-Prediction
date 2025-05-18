import torch.nn as nn

class LoLNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(LoLNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)