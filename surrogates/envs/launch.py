import os
import sys
from flask import Flask, request, jsonify

# Ensure envs directory is in path for absolute imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from ChEMBL35.predict import MoleculePredictor
from Supercon.predict import Inference as SuperconInference
from Nanohelix.predict import predict_g_factor
from TMC.predict import find_tmc_value_from_smiles

app = Flask(__name__)

# Initialize models globally so they don't reload on every request
chembl_model_path = os.path.join(CURRENT_DIR, "ChEMBL35", "models", "best_r2_model.pt")
chembl_predictor = MoleculePredictor(model_path=chembl_model_path)

supercon_model_path = os.path.join(CURRENT_DIR, "Supercon", "models", "best_supercon_model.pth")
supercon_inference = SuperconInference(model_path=supercon_model_path, input_size=470, hidden_size=128)


@app.route("/prediction_chembl", methods=["POST"])
def prediction_chembl():
    data = request.json
    smiles = data.get("smiles")
    if not smiles:
        return jsonify({"error": "smiles is required"}), 400
    try:
        prediction = chembl_predictor.predict(smiles)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/prediction_supercon", methods=["POST"])
def prediction_supercon():
    data = request.json
    element = data.get("element")
    str3 = data.get("str3")
    if not element:
        return jsonify({"error": "element is required"}), 400
    try:
        prediction = supercon_inference.predict_tc_from_formula(element, str3)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/prediction_nanohelix", methods=["POST"])
def prediction_nanohelix():
    data = request.json
    required_keys = ["pitch", "fiber_radius", "n_turns", "helix_radius"]
    for k in required_keys:
        if k not in data:
            return jsonify({"error": f"{k} is required"}), 400
    try:
        result = predict_g_factor(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/prediction_tmc_smiles", methods=["POST"])
def prediction_tmc_smiles():
    data = request.json
    tmc = data.get("tmc")
    property_name = data.get("property", "polarisability")
    if not tmc:
        return jsonify({"error": "tmc is required"}), 400
    try:
        result = find_tmc_value_from_smiles(tmc, property_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    print("Starting Unified Surrogate Server on http://127.0.0.1:12600")
    app.run(host="127.0.0.1", port=12600)


if __name__ == "__main__":
    main()
