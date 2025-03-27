import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
7

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # Multiply by 2 for bidirectional
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional
        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out.view(-1, 1, 2)  # Ensure the output shape matches the target shape

class MaskedMSELoss(nn.Module):
    def __init__(self, ignore_value=-99):
        super(MaskedMSELoss, self).__init__()
        self.ignore_value = ignore_value
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        mask = targets != self.ignore_value
        loss = self.mse_loss(inputs, targets)
        loss = loss * mask.float()  # Apply the mask
        return loss.sum() / mask.sum()  # Normalize by the number of valid elements