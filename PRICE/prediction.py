import torch
import torch.nn as nn
import pandas as pd
import joblib
import argparse

#Correr con : python prediction.py --model audi.pth


# Definir la red neuronal igual que en el entrenamiento
class PricePredictor(nn.Module):
    def __init__(self, input_dim):
        super(PricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)

def load_model(model_path, scaler_path, columns_path, device="cpu"):
    # Cargar scaler y columnas
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(columns_path)

    # Crear modelo
    input_dim = len(feature_columns)
    model = PricePredictor(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, scaler, feature_columns

def predict_price(car_data, model, scaler, feature_columns, device="cpu"):
    # Convertir diccionario a DataFrame
    df_input = pd.DataFrame([car_data])

    # Mapeos como en entrenamiento
    df_input["fuel_type"] = df_input["fuel_type"].map({"Gasoline": 0, "Diesel": 1, "Electric": 2})
    df_input["transmission"] = df_input["transmission"].map({"A/T": 0, "M/T": 1, "Automatic":0, "Manual":1, "Dual shift":0})

    # Reordenar columnas
    df_input = df_input[feature_columns]

    # Normalizar
    X_input = scaler.transform(df_input)

    # Convertir a tensor
    X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)

    # Inferencia
    with torch.no_grad():
        pred_price = model(X_tensor).cpu().numpy()[0][0]

    return pred_price

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predecir precio de coche usado con modelo entrenado.")
    parser.add_argument("--model", type=str, required=True, help="Ruta al archivo .pth (ej: chevrolet.pth)")
    parser.add_argument("--scaler", type=str, default="scaler.save", help="Ruta al scaler guardado")
    parser.add_argument("--columns", type=str, default="feature_columns.pkl", help="Ruta al archivo de columnas")
    args = parser.parse_args()

    # === EJEMPLO DE COCHE ===
    nuevo_coche = {
        "transmission": "A/T",   # "Manual", "Automatic", "M/T", "A/T", "Dual shift"
        "model_year": 2021,
        "kilometraje": 90000,
        "fuel_type": "Gasoline",    # "Gasoline", "Diesel", "Electric"
        "accident(1 yes /0 no)": 0,
        "HP": 150,
        "L": 1.48
    }

    # Cargar modelo
    model, scaler, feature_columns = load_model(args.model, args.scaler, args.columns)

    # Hacer predicción
    precio_estimado = predict_price(nuevo_coche, model, scaler, feature_columns)

    print(f"\n=== Predicción ===")
    print(f"Precio estimado para el coche introducido: {precio_estimado:,.2f} €")
