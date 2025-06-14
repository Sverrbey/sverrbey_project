import numpy as np

def nan_filter(y_true, y_pred):
    """Return y_true and y_pred with only valid (non-nan, non-inf) pairs."""
    mask = (
        ~np.isnan(y_true) & ~np.isnan(y_pred) &
        ~np.isinf(y_true) & ~np.isinf(y_pred)
    )
    return y_true[mask], y_pred[mask]

def NSE_formula(y_true, y_pred):
    y_true, y_pred = nan_filter(y_true, y_pred)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator if denominator != 0 else 0)

def RMSE_formula(y_true, y_pred):
    y_true, y_pred = nan_filter(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

def MAE_formula(y_true, y_pred):
    y_true, y_pred = nan_filter(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def MAPE_formula(y_true, y_pred):
    y_true, y_pred = nan_filter(y_true, y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def R_squared_formula(y_true, y_pred):
    y_true, y_pred = nan_filter(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot if ss_tot != 0 else 0)

def KGE_formula(y_true, y_pred):
    y_true, y_pred = nan_filter(y_true, y_pred)
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    mean_y_true = np.mean(y_true)
    mean_y_pred = np.mean(y_pred)
    std_y_true = np.std(y_true)
    std_y_pred = np.std(y_pred)
    if std_y_true == 0 or mean_y_true == 0:
        return np.nan
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    kge = correlation * (std_y_pred / std_y_true) * (mean_y_pred / mean_y_true)
    return kge

def Pearson_formula(y_true, y_pred):
    y_true, y_pred = nan_filter(y_true, y_pred)
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    return np.corrcoef(y_true, y_pred)[0, 1]