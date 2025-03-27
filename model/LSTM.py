import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
       # Using nn.Sequential to create non-linear containers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Multiply by 2 for bidirectional
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
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional
        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc_layers(out[:, -1, :])
        return out.view(-1, 1, 2)  # Ensure the output shape matches the target shape

class MaskedMSELoss(nn.Module):
    def __init__(self, ignore_value=-99.0):
        super(MaskedMSELoss, self).__init__()
        self.ignore_value = ignore_value

    def forward(self, predictions, targets):
        # Create a mask for valid values (where targets are not equal to the ignore value)
        mask = targets != self.ignore_value  # Shape: (batch_size, num_features)

        # Apply the mask to the predictions and targets
        masked_predictions = predictions[mask]
        masked_targets = targets[mask]

        # Compute the Mean Squared Error only on valid values
        loss = nn.functional.mse_loss(masked_predictions, masked_targets, reduction='mean')
        return loss