import torch.nn as nn

class HealthScoreNN(nn.Module):
    def __init__(self, input_size):
        super(HealthScoreNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.fc(x)