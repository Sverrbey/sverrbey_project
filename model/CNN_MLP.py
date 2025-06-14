import torch
import torch.nn as nn

class CNN_MLP(nn.Module):
    def __init__(self, input_channels, height, width, hidden_size, output_size, num_layers=1, dropout=0.3):
        super(CNN_MLP, self).__init__()
        # CNN for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Reduce spatial size for efficiency
            nn.Flatten()  # Output shape: (batch, 32*4*4)
        )
        cnn_output_size = 32 * 4 * 4

        # MLP layers: ensure final output size matches fc (output_size)
        self.mlp = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x: (batch, channels, height, width)
        batch_size, channels, height, width = x.size()
        # Merge batch for CNN processing
        x = x.view(batch_size, channels, height, width)
        x = self.cnn(x)  # (batch_size, cnn_output_size)
        
        # Restore sequence dimension
        x = x.view(batch_size, -1)  # (batch_size, cnn_output_size)
        
        # Pass through the MLP layers
        out = self.mlp(x)  

        return out