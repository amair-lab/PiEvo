import requests
import json
import sys


BASE_URL = "http://127.0.0.1:12600"


def test_endpoint(name, path, data):
    print(f"\n[SYSTEM] Testing {name} prediction...")
    url = f"{BASE_URL}{path}"
    try:
        session = requests.Session()
        session.trust_env = False
        
        response = session.post(
            url,
            data=json.dumps(data),
            headers={"Content-type": "application/json"},
            timeout=60
        )
        print(f"Status Code: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=4))
            res_json = response.json()
        except:
            print(f"Raw response: {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {name} server at {url}. Is it running? (check ip and port)")
    except Exception as e:
        print(f"❌ Error testing {name}: {e}")





# ChEMBL 
test_endpoint(
    "ChEMBL", 
    "/prediction_chembl", 
    {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O"
    }
)


# Supercon 
test_endpoint(
    "Superconductor", 
    "/prediction_supercon", 
    {
        "element": "Ba0.2La1.8Cu1O4-Y", 
        "str3": "T"
    }
)

# Nanohelix
test_endpoint(
    "Nanohelix", 
    "/prediction_nanohelix", 
    {
        "pitch": 200.0, 
        "fiber_radius": 20.0, 
        "n_turns": 3.0, 
        "helix_radius": 40.0,
        "wavelength": 460.0, 
        "x_y": 2, 
        "direction": 0
    }
)


# TMC
test_endpoint(
    "TMC", 
    "/prediction_tmc_smiles", 
    {
        "tmc": "Pd_c1ccccn1_CP(C)C_S1(=O)(=O)[N-]C(=O)c2c1cccc2_[N-]=[N+]=[N-]",
        "property": "polarisability"
    }
)

