import numpy as np
import pandas as pd
from .ligands import LIGAND_POOL

class TMCDataProcessor:
    def __init__(self):
        self.ligands = sorted(list(LIGAND_POOL.keys()))
        self.ligand_to_idx = {l: i for i, l in enumerate(self.ligands)}
        self.atom_types = sorted(list(set(l["atom"] for l in LIGAND_POOL.values())))
        self.atom_to_idx = {a: i for i, a in enumerate(self.atom_types)}
        
        self.num_ligands = len(self.ligands)
        self.num_atoms = len(self.atom_types)
        self.feat_per_ligand = self.num_ligands + self.num_atoms + 1
        self.total_features = self.feat_per_ligand * 4

    def get_ligand_features(self, ligand_id):
        if ligand_id not in LIGAND_POOL:
            # Handle unknown ligands with zeros
            return np.zeros(self.feat_per_ligand)
            
        data = LIGAND_POOL[ligand_id]
        
        # One-hot ligand ID
        l_feat = np.zeros(self.num_ligands)
        l_feat[self.ligand_to_idx[ligand_id]] = 1.0
        
        # One-hot atom
        a_feat = np.zeros(self.num_atoms)
        a_feat[self.atom_to_idx[data["atom"]]] = 1.0
        
        # Charge
        c_feat = np.array([float(data["charge"])])
        
        return np.concatenate([l_feat, a_feat, c_feat])

    def featurize_tmc(self, tmc_string):
        """Convert Pd_L1_L2_L3_L4 string to feature vector."""
        parts = tmc_string.split("_")
        if len(parts) != 5 or parts[0] != "Pd":
            return np.zeros(self.total_features)
            
        ligs = parts[1:]
        feats = []
        for l in ligs:
            feats.append(self.get_ligand_features(l))
            
        return np.concatenate(feats)

    def get_all_rotations(self, tmc_string):
        """Generate all 4 cyclic rotations of a TMC string."""
        parts = tmc_string.split("_")
        if len(parts) != 5 or parts[0] != "Pd":
            return [tmc_string]
            
        ligs = parts[1:]
        rotations = []
        for i in range(4):
            rot = ligs[i:] + ligs[:i]
            rotations.append(f"Pd_{'_'.join(rot)}")
        return rotations

    def prepare_data(self, df, augment=False):
        """
        Prepare features and targets from a DataFrame.
        Expected columns: 'lig1', 'lig2', 'lig3', 'lig4', 'polarisability'
        """
        X = []
        y = []
        
        # We assume the input DF has polarisability
        for _, row in df.iterrows():
            tmc = f"Pd_{row['lig1']}_{row['lig2']}_{row['lig3']}_{row['lig4']}"
            val = float(row['polarisability'])
            
            if augment:
                # Add all 4 rotations with the same target value
                for rot_tmc in self.get_all_rotations(tmc):
                    X.append(self.featurize_tmc(rot_tmc))
                    y.append(val)
            else:
                X.append(self.featurize_tmc(tmc))
                y.append(val)
                
        return np.array(X), np.array(y)
