import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# Define model class (needed for loading)
class ResidualBlock(nn.Module):
    def __init__(self, n_units, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(n_units, n_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(n_units, n_units)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out += residual
        out = self.relu(out)
        return out

class NanohelixTorchModel(nn.Module):
    def __init__(self, input_dim, n_layers, n_units, dropout_rate=0.1):
        super(NanohelixTorchModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, n_units)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList([
            ResidualBlock(n_units, dropout_rate) for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(n_units, 1)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(out)
        for block in self.blocks:
            out = block(out)
        out = self.output_layer(out)
        return out


def predict_g_factor(params_dict):
    """
    Predict the g-factor for a nanohelix structure given a dictionary of the 4 basic parameters.

    Parameters:
    -----------
    params_dict : dict
        Dictionary containing the 4 basic parameters:
        - 'pitch': The pitch of the helix
        - 'fiber_radius': The radius of the fiber
        - 'n_turns': Number of turns in the helix
        - 'helix_radius': Radius of the helix

    Returns:
    --------
    result_dict : dict
        Dictionary containing:
        - 'g_factor': Predicted g-factor
        - All input parameters
        - All derived parameters
    """
    # Extract parameters from dictionary
    pitch = params_dict["pitch"]
    fiber_radius = params_dict["fiber_radius"]
    n_turns = params_dict["n_turns"]
    helix_radius = params_dict["helix_radius"]
    
    # Optional parameters with defaults
    # Default values chosen based on training data analysis:
    # wavelength=550.0 (middle of visible spectrum)
    # x_y=0 (default class)
    # direction=0 (default class)
    x_y = params_dict.get("x_y", 0)
    direction = params_dict.get("direction", 0)
    wavelength = params_dict.get("wavelength", 550.0)

    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define function to get relative path
    def get_path(filename):
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filename)

    # Check if model and scalers exist
    model_path = get_path("models/nanohelix_torch_model.pth")
    scaler_X_path = get_path("models/nanohelix_scaler_X.pkl")
    scaler_y_path = get_path("models/nanohelix_scaler_y.pkl")

    if not all(os.path.exists(p) for p in [model_path, scaler_X_path, scaler_y_path]):
        raise FileNotFoundError("Model files not found. Please train the model first.")

    # Load scalers
    import joblib
    import pandas as pd
    import numpy as np
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    
    # Load model metadata and state
    model_data = torch.load(model_path, map_location=device)
    model = NanohelixTorchModel(
        model_data['input_dim'],
        model_data['n_layers'],
        model_data['n_units'],
        model_data['dropout_rate']
    )
    model.load_state_dict(model_data['state_dict'])
    model.to(device)
    model.eval()

    # Create a DataFrame with the basic parameters
    data = pd.DataFrame({
        "pitch": [pitch],
        "fiber_radius": [fiber_radius],
        "n_turns": [n_turns],
        "helix_radius": [helix_radius]
    })

    # Prepare complete feature set matching train.py
    data_enriched = compute_nanohelix_parameters(data)
    data_enriched["x_y"] = x_y
    data_enriched["direction"] = direction
    data_enriched["wavelength"] = wavelength

    # Ensure columns match scaler's requirement
    X = data_enriched[scaler_X.feature_names_in_]

    # Scale features
    X_scaled = scaler_X.transform(X)

    # Make prediction
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        y_pred_scaled = model(X_tensor).cpu().numpy().ravel()
    
    g_factor = float(scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0][0])

    # Create result dictionary
    result_dict = {
        "g_factor": g_factor,
        **params_dict,
        **{k: v for k, v in data_enriched.iloc[0].to_dict().items() if k not in params_dict}
    }

    return result_dict


def compute_nanohelix_parameters(df):
    # Create a copy to avoid modifying the original
    df_enriched = df.copy()

    # Calculate derived parameters using vectorized operations
    pitch = df_enriched["pitch"]
    fiber_radius = df_enriched["fiber_radius"]
    n_turns = df_enriched["n_turns"]
    helix_radius = df_enriched["helix_radius"]

    # Calculate turn length
    turn_length = np.sqrt((2 * np.pi * helix_radius) ** 2 + pitch**2)

    # Derived parameters as they appear in the CSV
    df_enriched["total_length"] = (turn_length * n_turns).astype(float)
    df_enriched["height"] = (pitch * n_turns).astype(float)
    df_enriched["curl"] = (helix_radius / (helix_radius**2 + (pitch / (2 * np.pi)) ** 2)).astype(float)
    df_enriched["angle"] = (np.arctan2(pitch, 2 * np.pi * helix_radius)).astype(float)
    
    # torsion (extra feature)
    df_enriched["torsion"] = ((pitch / (2 * np.pi)) / (helix_radius**2 + (pitch / (2 * np.pi)) ** 2)).astype(float)
    
    total_fiber_length = turn_length * n_turns * (1 + (2 * np.pi * fiber_radius) / turn_length)
    df_enriched["total_fiber_length"] = total_fiber_length.astype(float)
    
    # Volume and Mass
    V = np.pi * fiber_radius**2 * total_fiber_length
    df_enriched["V"] = V.astype(float)
    df_enriched["mass"] = V.astype(float)

    return df_enriched
