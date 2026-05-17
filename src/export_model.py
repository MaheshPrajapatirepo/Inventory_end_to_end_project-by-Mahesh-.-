import pickle
import os
 
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def save_model(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Saved → {path}")

def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Loaded ← {path}")
    return model
