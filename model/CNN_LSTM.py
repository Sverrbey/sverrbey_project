import torch
import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size, lstm_shape, num_layers=1, dropout=0.3):
        super(CNN_LSTM_Model, self).__init__()
        
        # Convolutional layers to extract spatial features
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, H, W)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (16, H/2, W/2)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (32, H/4, W/4)
        )
        
        # LSTM layers to model temporal dependencies
        self.lstm = nn.LSTM(32 * (lstm_shape[1] // 4) * (lstm_shape[0] // 4), hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_size, output_size)
    
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