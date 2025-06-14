import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_channels, height, width, hidden_size, output_size, num_layers=1, dropout=0.3):
        super(BiLSTM, self).__init__()
        
        # Flatten spatial dimensions (height * width) into a single feature vector
        self.input_size = input_channels * height * width
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size  = self.input_size, # Flattened input size
            hidden_size = hidden_size,     # Number of hidden units
            num_layers  = num_layers,      # Number of LSTM layers
            batch_first = True,            # Input shape: (batch_size, seq_length, input_size)
            dropout     = dropout,         # Dropout for regularization
            bidirectional = True           # Enable bidirectional LSTM
        )
        
        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_size*2, output_size)

        
    def forward(self, x):
        # Input shape: (batch_size, seq_length, channels, height, width)
        batch_size, seq_length, channels, height, width = x.size()

        # Flatten spatial dimensions (channels, height, width) into a single feature vector
        x = x.view(batch_size, seq_length, -1)  # Shape: (batch_size, seq_length, input_size)

        batch_size, seq_length, _ = x.size()
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # Output shape: (batch_size, seq_length, hidden_size)
        
        # Take the output of the last time step
        out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through the fully connected layer
        out = self.fc(out)  # Shape: (batch_size, output_size)
        
        return out