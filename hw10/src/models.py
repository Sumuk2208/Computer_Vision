from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)  # Single linear layer

    def forward(self, x):
        # Flatten the input if it's not already flat (e.g., for images)
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch
        return self.linear(x)
