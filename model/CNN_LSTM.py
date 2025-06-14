import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, height, width, hidden_size, output_size, num_layers=1, dropout=0.3):
        super(CNN_LSTM, self).__init__()
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

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.size()
        # Merge batch and seq_len for CNN processing
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.cnn(x)  # (batch_size * seq_len, cnn_output_size)
        # Restore sequence dimension
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, cnn_output_size)
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        out = lstm_out[:, -1, :]    # Last time step
        out = self.fc(out)          # (batch_size, output_size)
        return out