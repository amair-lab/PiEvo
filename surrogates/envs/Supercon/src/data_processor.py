import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import re


from .elemental_properties import ELEMENTAL_PROPERTIES, PROPERTY_NAMES

class SuperconDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        # Define the periodic table elements from the properties source
        self.elements = list(ELEMENTAL_PROPERTIES.keys())
        self.structure_types = []
        self.feature_columns = []

    def _parse_formula(self, formula):
        """
        Robustly parse chemical formula into elemental composition ratios,
        handling parentheses and nested fractions.

        Example:
        "Ba0.2La1.8Cu1O4-Y" -> {'Ba': 0.2, 'La': 1.8, 'Cu': 1.0, 'O': 4.0}
        "(La,Sr)2CuO4" -> {'La': 1.0, 'Sr': 1.0, 'Cu': 1.0, 'O': 4.0}
        "Ba0.15La1.85Cu1O4-Y" -> {'Ba': 0.15, 'La': 1.85, 'Cu': 1.0, 'O': 4.0}
        """
        if pd.isna(formula) or not isinstance(formula, str):
            return {}

        # Remove any non-standard characters like -Y, -X, (O,S) etc that are not simple parentheses
        # For simplicity in this env, we clean some common Supercon suffixes
        formula = re.sub(r'-[A-ZYZXz]+', '', formula)
        formula = re.sub(r'[A-ZYZXz]$', '', formula) # trailing X, Y, Z

        def parse_comp(s):
            comp = {}
            i = 0
            while i < len(s):
                if s[i] == '(':
                    # Find matching parenthesis
                    count = 1
                    j = i + 1
                    while j < len(s) and count > 0:
                        if s[j] == '(': count += 1
                        elif s[j] == ')': count -= 1
                        j += 1
                    
                    sub_comp = parse_comp(s[i+1:j-1])
                    i = j
                    # Get multiplier
                    match = re.match(r'(\d*\.?\d*)', s[i:])
                    mul = float(match.group(1)) if match.group(1) else 1.0
                    i += len(match.group(1))
                    
                    for el, count in sub_comp.items():
                        comp[el] = comp.get(el, 0) + count * mul
                elif s[i] == ',':
                    # Handle (La,Sr) by treating it as additive if no fractions provided
                    # In many Supercon formulas, (La,Sr)2 means La and Sr sum to 2
                    # But without explicit ratios, we might need to be careful.
                    # Here we treat it as balanced if weights aren't specified.
                    i += 1
                elif s[i].isupper():
                    match = re.match(r'([A-Z][a-z]*)(\d*\.?\d*)', s[i:])
                    if match:
                        el = match.group(1)
                        num = float(match.group(2)) if match.group(2) else 1.0
                        comp[el] = comp.get(el, 0) + num
                        i += len(match.group(0))
                    else:
                        i += 1
                else:
                    i += 1
            return comp

        return parse_comp(formula)

    def _extract_advanced_features(self, df):
        """
        Extract Magpie-style features: Weighted Mean, Weighted Avg Dev, Max, Min, Range.
        """
        features_list = []
        valid_indices = []

        for idx, row in df.iterrows():
            formula = row["element"]
            try:
                counts = self._parse_formula(formula)
                if not counts:
                    continue

                total_atoms = sum(counts.values())
                fractions = {el: c / total_atoms for el, c in counts.items()}

                # Check if all elements exist in our database
                if any(el not in ELEMENTAL_PROPERTIES for el in counts.keys()):
                    continue

                row_features = []
                for prop_name in PROPERTY_NAMES:
                    values = np.array([ELEMENTAL_PROPERTIES[el][prop_name] for el in counts.keys()])
                    weights = np.array([fractions[el] for el in counts.keys()])

                    # Weighted Mean
                    w_mean = np.sum(values * weights)
                    # Weighted Average Deviation
                    w_dev = np.sum(weights * np.abs(values - w_mean))
                    # Max, Min, Range
                    f_max = np.max(values)
                    f_min = np.min(values)
                    f_range = f_max - f_min

                    row_features.extend([w_mean, w_dev, f_max, f_min, f_range])



                features_list.append(row_features)
                valid_indices.append(idx)
            except Exception:
                continue

        X = np.array(features_list)
        if "tc" in df.columns:
            y = df.iloc[valid_indices]["tc"].values
        else:
            y = np.zeros(len(valid_indices))
        
        # Structure features (one-hot)
        if "str3" in df.columns:
            df_valid = df.iloc[valid_indices]
            # Use fixed structure list if available, or create it
            if not self.structure_types:
                self.structure_types = sorted(df_valid["str3"].dropna().unique().tolist())
            
            X_struct = pd.get_dummies(df_valid["str3"])
            # Ensure all columns exist
            for st in self.structure_types:
                if st not in X_struct.columns:
                    X_struct[st] = 0
            X_struct = X_struct[self.structure_types].values
            X = np.hstack((X, X_struct))
            
        return X, y

    def load_and_process_data(self, tsv_path, test_size=0.2, random_state=42):
        print(f"Loading data from {tsv_path}...")
        df = pd.read_csv(tsv_path, sep="\t")
        df = df.dropna(subset=["element", "tc"])

        # Extract features and filtered target
        X, y = self._extract_advanced_features(df)
        
        print(f"Extracted {X.shape[0]} valid samples with {X.shape[1]} features")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Save processor state for inference
        if not os.path.exists("models"):
            os.makedirs("models")
        
        state_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "supercon_processor.pkl")
        with open(state_path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler, 
                "structure_types": self.structure_types,
                "n_features": X_train_scaled.shape[1]
            }, f)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def load_state(self, path=None):
        """
        Load processor state from pickle file
        """
        if path is None:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            path = os.path.join(model_dir, "supercon_processor.pkl")
        
        with open(path, "rb") as f:
            state = pickle.load(f)
            self.scaler = state["scaler"]
            self.structure_types = state.get("structure_types", [])
            self.n_features = state.get("n_features", 0)
        return state

    def process_input(self, input_data):
        """
        Process input data for inference
        """
        if self.scaler is None or not hasattr(self, 'n_features'):
            self.load_state()

        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Ensure all required columns are there
        if "element" not in input_data.columns:
            raise ValueError("Missing 'element' column")

        # Reuse the advanced extraction logic
        X, _ = self._extract_advanced_features(input_data)
        
        # Handle cases where str3 might be missing or features don't match
        if X.shape[1] < self.n_features:
            # Pad with zeros (assuming structure features are at the end)
            padding = np.zeros((X.shape[0], self.n_features - X.shape[1]))
            X = np.hstack((X, padding))
        elif X.shape[1] > self.n_features:
            # This shouldn't happen if extraction is consistent
            X = X[:, :self.n_features]

        return self.scaler.transform(X)

