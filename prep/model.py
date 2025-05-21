import pickle
import os

model_dir = os.path.dirname("./prep/models/")

def save_model(model, filename="arima_model.pkl"):
    dir = os.path.join(model_dir, filename)
    with open(dir, "wb") as f:
        pickle.dump(model, f)

def load_model(filename="arima_model.pkl"):
    dir = os.path.join(model_dir, filename)
    with open(dir, "rb") as f:
        model = pickle.load(f)
    return model