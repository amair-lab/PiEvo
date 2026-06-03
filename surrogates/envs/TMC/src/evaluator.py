import torch
import pandas as pd
import numpy as np
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset
from .model import PDComplexPredictor
from .data_processor import TMCDataProcessor

class TMCEvaluator:
    def __init__(self, model_path, scaler_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TMCDataProcessor()
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler_data = pickle.load(f)
        
        # Initialize and load model
        input_dim = self.processor.total_features
        self.model = PDComplexPredictor(input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Pre-calculate ligand features for speed
        self.ligand_features = {
            l_id: self.processor.get_ligand_features(l_id) 
            for l_id in self.processor.ligands
        }

    def predict_batch(self, df):
        """Highly optimized featurization and prediction."""
        # Convert ligand columns to indices or directly to features
        # Columns: lig1, lig2, lig3, lig4
        batch_size = len(df)
        feat_dim = self.processor.feat_per_ligand
        
        X = np.zeros((batch_size, self.processor.total_features), dtype=np.float32)
        
        for i in range(4):
            lig_col = f"lig{i+1}"
            # Map ligand IDs to pre-calculated features
            # Use .values for faster iteration
            col_data = df[lig_col].values
            for j in range(batch_size):
                lid = col_data[j]
                if lid in self.ligand_features:
                    X[j, i*feat_dim : (i+1)*feat_dim] = self.ligand_features[lid]
        
        X_tensor = torch.from_numpy(X).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()
            
        # Unscale predictions
        preds_unscaled = preds * self.scaler_data['std'] + self.scaler_data['mean']
        return preds_unscaled

    def evaluate_full_dataset(self, data_path, output_path, batch_size=50000):
        """Process the 1.3M dataset in chunks and save results."""
        print(f"Starting evaluation on {data_path}...")
        
        # If output exists, we can skip or overwrite. Here we overwrite.
        if os.path.exists(output_path):
            os.remove(output_path)
            
        header = True
        total_processed = 0
        
        # Read in chunks
        reader = pd.read_csv(data_path, chunksize=batch_size)
        
        for chunk in reader:
            preds = self.predict_batch(chunk)
            
            # Prepare result chunk
            res = pd.DataFrame({
                'id': chunk['id'],
                'true_polarisability': chunk['polarisability'],
                'pred_polarisability': preds
            })
            
            # Append to CSV
            res.to_csv(output_path, mode='a', index=False, header=header)
            header = False
            
            total_processed += len(chunk)
            if total_processed % (batch_size * 2) == 0:
                print(f"Processed {total_processed} rows...")
                
        print(f"Evaluation complete. Results saved to {output_path}")
        return output_path
