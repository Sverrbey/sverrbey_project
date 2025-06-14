import torch
import torch.nn as nn

# Decoupled LSTM model with MLP layers for additional processing
class Double_LSTM(nn.Module):
    def __init__(self, input_channels, height, width, hidden_size, output_size, num_layers=2, dropout=0.3):
        super(Double_LSTM, self).__init__()

        input_size  = input_channels * height * width
        
        self.lstm   = nn.LSTM(input_size,
                                hidden_size,
                                num_layers,
                                batch_first=True,
                                dropout=dropout)
        
        # MLP layers: ensure final output size matches fc (output_size)
        self.gradient = nn.LSTM(hidden_size,
                                128,
                                2,
                                batch_first=True,
                                dropout=dropout)
        
        self.fc_grad = nn.Linear(128, output_size)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input shape: (batch_size, seq_length, channels, height, width)
        batch_size, seq_length, _, _, _ = x.size()
        
        # Flatten spatial dimensions (channels, height, width) into a single feature vector
        x = x.view(batch_size, seq_length, -1)  # Shape: (batch_size, seq_length, input_size)
        
        # Pass through LSTM
        lstm_out, (_,_) = self.lstm(x)  # Output shape: (batch_size, seq_length, hidden_size)
        
        # Pass through second LSTM for gradient computation
        gradient_out, (_,_) = self.gradient(lstm_out)  # Output shape: (batch_size, seq_length, hidden_size)

        # Take the output of the last time step
        out             = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        gradient_out    = gradient_out[:, -1, :]  # Get the last time step output from the gradient LSTM

        # Pass through the fully connected layer
        out             = self.fc(out)  # Shape: (batch_size, output_size)
        external_out    = self.fc_grad(gradient_out)  # Use the last time step output from the gradient LSTM

        return out + external_out