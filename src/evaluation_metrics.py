import numpy as np

# Evaluation metrics
def NSE_formula(y_true, y_pred):
    # Calculate NSE in the original scale
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator if denominator != 0 else 0)

def RMSE_formula(y_true, y_pred):
    # Calculate RMSE in the original scale
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

def MAE_formula(y_true, y_pred):
    # Calculate MAE in the original scale
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def MAPE_formula(y_true, y_pred):
    # Calculate MAPE in the original scale
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def R_squared_formula(y_true, y_pred):
    # Calculate R-squared in the original scale
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot if ss_tot != 0 else 0)

def KGE_formula(y_true, y_pred):
    # Calculate KGE in the original scale
    mean_y_true = np.mean(y_true)
    mean_y_pred = np.mean(y_pred)
    std_y_true = np.std(y_true)
    std_y_pred = np.std(y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    kge = correlation * (std_y_pred / std_y_true) * (mean_y_pred / mean_y_true)
    return kge

def Pearson_formula(y_true, y_pred):
    # Calculate Pearson correlation coefficient
    return np.corrcoef(y_true, y_pred)[0, 1]
