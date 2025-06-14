import torch
import torch.nn as nn

# Decoupled LSTM model with MLP layers for additional processing
class Decoupled_LSTM(nn.Module):
    def __init__(self, input_channels, height, width, hidden_size, output_size, num_layers=2, dropout=0.3, mlp_hidden=256):
        super(Decoupled_LSTM, self).__init__()

        input_size  = input_channels * height * width
        
        self.lstm   = nn.LSTM(input_size,
                                hidden_size,
                                num_layers,
                                batch_first=True,
                                dropout=dropout)
        
        # MLP layers: ensure final output size matches fc (output_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, output_size)
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input shape: (batch_size, seq_length, channels, height, width)
        batch_size, seq_length, channels, height, width = x.size()
        
        # Flatten spatial dimensions (channels, height, width) into a single feature vector
        x = x.view(batch_size, seq_length, -1)  # Shape: (batch_size, seq_length, input_size)
        
        # Pass through LSTM
        lstm_out, (h,_) = self.lstm(x)  # Output shape: (batch_size, seq_length, hidden_size)
        
        
        # Pass through the MLP layers
        external_out = self.mlp(out) 
         
        # Take the output of the last time step
        out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through the fully connected layer
        out = self.fc(out)  # Shape: (batch_size, output_size)

        return out + external_out