import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels, output_size, hidden_size, dropout=0.3):
        super(CNN, self).__init__()
        
        # Convolutional layers to extract spatial features
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, 29, 46)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (16, 14, 23)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 14, 23)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 7, 11)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 7, 11)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (64, 3, 5)
        )
        
        flattened_size = 64  # Replace with the actual flattened size
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )
    def forward(self, x):
        # Input shape: (batch_size, seq_length, channels, height, width)
        batch_size, seq_length, channels, height, width = x.size()

        # Reshape input for CNN: (batch_size * seq_length, channels, height, width)
        x = x.view(batch_size * seq_length, channels, height, width)

        # Pass through CNN layers
        x = self.cnn(x)  # Output shape after CNN: (batch_size * seq_length, 64, 1, 1)

        # Flatten the output: (batch_size * seq_length, 64)
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = self.fc_layers(x)  # Output shape: (batch_size * seq_length, output_size)

        # Reshape back to (batch_size, seq_length, output_size)
        x = x.view(batch_size, seq_length, -1)

        # Aggregate over the sequence dimension (e.g., take the mean or last time step)
        x = x.mean(dim=1)  # Output shape: (batch_size, output_size)

        return x