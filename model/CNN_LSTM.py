import torch
import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_channels, height, width, hidden_size, output_size, num_layers=1, dropout=0.3):
        super(CNN_LSTM_Model, self).__init__()
        
        # Flatten spatial dimensions (height * width) into a single feature vector
        self.input_size = input_channels * height * width

        # Convolutional layers to extract spatial features
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, H, W)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (16, H/2, W/2)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (32, H/4, W/4)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,  # Flattened input size
            hidden_size=hidden_size,    # Number of hidden units
            num_layers=num_layers,      # Number of LSTM layers
            batch_first=True,           # Input shape: (batch_size, seq_length, input_size)
            dropout=dropout,             # Dropout for regularization
            bidirectional=True  # Enable bidirectional LSTM
        )
        
        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        
        # Reshape input for CNN: (batch_size * seq_length, channels, height, width)
        x = x.view(batch_size * seq_length, channels, height, width)
        
        # Pass through CNN
        x = self.cnn(x)  # Output: (batch_size * seq_length, 32, H/4, W/4)
        
        # Flatten spatial dimensions: (batch_size * seq_length, 32 * H/4 * W/4)
        x = x.view(x.size(0), -1)
        
        # Reshape for LSTM: (batch_size, seq_length, 32 * H/4 * W/4)
        x = x.view(batch_size, seq_length, -1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # Output: (batch_size, seq_length, hidden_size)
        
        # Take the output of the last time step
        out = self.fc(lstm_out[:, -1, :])  # Output: (batch_size, output_size)
        
        return out