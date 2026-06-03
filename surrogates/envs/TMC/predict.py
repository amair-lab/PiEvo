import os
import logging
import pandas as pd
import subprocess
import sys
from typing import List, Optional, Dict, Any

# Ensure project root is in path for absolute imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

logger = logging.getLogger(__name__)

# Global variables
dataset_df = None

def load_tmc_dataset():
    """Download (if needed) and load the TMC dataset."""
    global dataset_df
    
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    csv_filename = "ground_truth_fitness_values.csv"
    csv_path = os.path.join(data_dir, csv_filename)
    url = "https://zenodo.org/records/14328055/files/ground_truth_fitness_values.csv"

    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        if not os.path.exists(csv_path):
            logger.info(f"TMC dataset not found at {csv_path}. Downloading...")
            # Use curl to download
            subprocess.run(['curl', '-L', '-o', csv_path, url], check=True)
            logger.info("Download complete.")
        
        logger.info(f"Loading TMC dataset from {csv_path}...")
        dataset_df = pd.read_csv(csv_path)
        logger.info(f"TMC dataset loaded. Shape: {dataset_df.shape}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load TMC dataset: {e}")
        return False

# Surrogate model globals
surrogate_model = None
surrogate_processor = None
surrogate_scaler = None

def load_surrogate():
    """Lazy load the surrogate model and associated tools."""
    global surrogate_model, surrogate_processor, surrogate_scaler
    if surrogate_model is not None:
        return True
        
    try:
        import torch
        import joblib
        try:
            from .src.data_processor import TMCDataProcessor
            from .src.model import PDComplexPredictor
        except (ImportError, ValueError):
            from src.data_processor import TMCDataProcessor
            from src.model import PDComplexPredictor
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", "best_tmc_model.pth")
        scaler_path = os.path.join(current_dir, "models", "target_scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.warning("Surrogate model files not found. Fallback disabled.")
            return False
            
        surrogate_processor = TMCDataProcessor()
        surrogate_scaler = joblib.load(scaler_path)
        
        # Determine input dim
        input_dim = surrogate_processor.total_features
        surrogate_model = PDComplexPredictor(input_dim)
        
        # Load state dict
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device)
        surrogate_model.load_state_dict(state_dict)
        surrogate_model.to(device)
        surrogate_model.eval()
        
        logger.info("TMC Surrogate model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load TMC surrogate model: {e}")
        return False

def find_tmc_value(tmc_string: str, property_name: str = "polarisability") -> Dict[str, Any]:
    """
    Find the property value for a given TMC string.
    Prioritizes predicting using the surrogate model, falls back to ground truth CSV.
    Includes strict validation for structure, ligand pool, and charge balance.
    """
    global dataset_df
    
    # 1. Basic Format Validation
    # Format must be Pd_$L1_$L2_$L3_$L4
    parts = tmc_string.split("_")
    if len(parts) != 5 or parts[0] != "Pd":
        return {
            "input": {"tmc": tmc_string, "property": property_name},
            "output": 0.0,
            "success": False,
            "error_msg": f"Invalid format: '{tmc_string}'. Must strictly follow 'Pd_$L1_$L2_$L3_$L4' format with exactly 4 ligands."
        }

    # 2. Ligand Pool & Charge Balance Validation
    # Load metadata for validation
    try:
        from .src.ligands import LIGAND_POOL
    except (ImportError, ValueError):
        from src.ligands import LIGAND_POOL

    # Attempt to lazy load surrogate
    load_surrogate()

    ligand_ids = parts[1:]
    total_ligand_charge = 0
    missing_ligands = []
    
    for lid in ligand_ids:
        if lid not in LIGAND_POOL:
            missing_ligands.append(lid)
        else:
            total_ligand_charge += LIGAND_POOL[lid]['charge']

    if missing_ligands:
        return {
            "input": {"tmc": tmc_string, "property": property_name},
            "output": 0.0,
            "success": False,
            "error_msg": f"Invalid Ligands: {', '.join(missing_ligands)} not found in the valid ligand pool. Do NOT hallucinate names."
        }

    # Charge balance check: Pd(II) requires total ligand charge of -2 for a neutral complex.
    if total_ligand_charge != -2:
        return {
            "input": {"tmc": tmc_string, "property": property_name},
            "output": 0.0,
            "success": False,
            "error_msg": (
                f"Charge Mismatch: Total ligand charge is {total_ligand_charge}, but MUST be -2 "
                f"to neutralize the Pd(II) center ($Pd^{{2+}}$). "
                "Combine Neutral (0) and Anionic (-1) ligands appropriately."
            )
        }

    # 3. Try Predicting with Surrogate Model (if available)
    # Features are already validated at this point.
    if surrogate_model is not None and surrogate_processor is not None:
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            features = surrogate_processor.featurize_tmc(tmc_string)
            features_t = torch.FloatTensor(features).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_scaled = surrogate_model(features_t).cpu().item()
            
            # Unscale
            y_mean = surrogate_scaler["mean"]
            y_std = surrogate_scaler["std"]
            val = pred_scaled * y_std + y_mean
            
            return {
                "input": {"tmc": tmc_string, "property": property_name},
                "output": float(val) if val else 0.0,
                "success": True,
                "method": "predicting",
                "error_msg": ""
            }
        except Exception as e:
            logger.error(f"Surrogate prediction failed: {e}")
            # Fall through to lookup

    # 4. Fallback to Lookup in Dataset
    if dataset_df is None:
        load_tmc_dataset()

    val = None
    if dataset_df is not None:
        ligs = parts[1:]
        rotations = [ligs[i:] + ligs[:i] for i in range(4)]
        for rot in rotations:
            temp_df = dataset_df[
                (dataset_df["lig1"] == rot[0]) &
                (dataset_df["lig2"] == rot[1]) &
                (dataset_df["lig3"] == rot[2]) &
                (dataset_df["lig4"] == rot[3])
            ]
            if not temp_df.empty:
                if property_name in temp_df.columns:
                    val = float(temp_df[property_name].mean())
                    break
    
    if val is not None:
        return {
            "input": {"tmc": tmc_string, "property": property_name},
            "output": float(val),
            "success": True,
            "method": "lookup",
            "error_msg": ""
        }

    return {
        "input": {"tmc": tmc_string, "property": property_name},
        "output": 0.0,
        "success": False,
        "error_msg": "Structure passed validation but prediction failed and entry not found in database."
    }

def find_tmc_value_from_smiles(tmc_smiles: str, property_name: str = "polarisability") -> Dict[str, Any]:
    """
    Find the property value for a given TMC string using SMILES for ligands.
    Converts SMILES to internal IDs and then calls find_tmc_value.
    """
    parts = tmc_smiles.split("_")
    if len(parts) != 5 or parts[0] != "Pd":
        return {
            "input": {"tmc": tmc_smiles, "property": property_name},
            "output": 0.0,
            "success": False,
            "error_msg": f"Invalid format: '{tmc_smiles}'. Must strictly follow 'Pd_$S1_$S2_$S3_$S4' format with SMILES for ligands."
        }
    
    try:
        from .src.ligands import LIGAND_POOL
    except (ImportError, ValueError):
        from src.ligands import LIGAND_POOL
    
    # Create reverse mapping (lazy/on-the-fly for now, can be cached if needed)
    smiles_to_id = {v['smiles']: k for k, v in LIGAND_POOL.items()}
    
    mapped_ids = []
    missing_smiles = []
    for s in parts[1:]:
        if s in smiles_to_id:
            mapped_ids.append(smiles_to_id[s])
        else:
            missing_smiles.append(s)
            
    if missing_smiles:
        return {
            "input": {"tmc": tmc_smiles, "property": property_name},
            "output": 0.0,
            "success": False,
            "error_msg": f"Invalid SMILES: {', '.join(missing_smiles)} not found in the valid ligand pool. Must choose from the given pool."
        }
    
    tmc_id_string = f"Pd_{'_'.join(mapped_ids)}"
    result = find_tmc_value(tmc_id_string, property_name)
    
    # Overwrite input in result to show original SMILES query
    result["input"]["tmc"] = tmc_smiles
    return result
