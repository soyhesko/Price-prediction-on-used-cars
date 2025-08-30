import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch 
import torch.nn as nn
import torch.optim as optim
import joblib
import os

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler

#Leer las hojas del excel y unirlas en una sola variable 
def read_excel_sheets(path_excel):
    #Lee las hojas
    df= pd.read_excel(path_excel, sheet_name="AUDI")
    #Une las hojas en un solo DataFrame
    return df
#Ruta del archivo Excel
file_path="data/used_cars.xlsx"
#Leer el archivo Excel
df= read_excel_sheets(file_path)
#Eliminamos las columnas que no son necesarias para que compile más rápido
columnas_necesarias=["transmission","model_year","kilometraje","fuel_type","accident(1 yes /0 no)","precio","HP","L"]
df=df[columnas_necesarias]
#Convertir en número el tipo de motor
df["fuel_type"] = df["fuel_type"].map({"Gasoline": 0, "Diesel": 1, "Electric": 2})
#Convertir transmisión (0 automática / 1 manual)
df["transmission"] = df["transmission"].map({"A/T": 0, "M/T": 1,"Automatic":0,"Manual":1,"Dual shift":0})
#Todas las columnas en números
df=df.apply(pd.to_numeric, errors="coerce").fillna(0)
#Valores que faltan en 0
df=df[(df["HP"]!=0)&(df["L"]!=0)]
#Definimos x(caracteristicas) e y (precio)
X = df.drop(columns=["precio"])
Y = df["precio"]
#Testeando el resultado 
print("Primeras filas de X:")
print(X.head())
print("Primeras filas de Y:")
print(Y.head())
#Entrenamiento, dividir datos (80 % entrenamiento y 20 % prueba)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1, random_state=42)
#Normalizar las características (media=0, desviación=1)
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#Convertir a tensores para Pytorch
X_train_tensor=torch.tensor(X_train,dtype=torch.float32)
Y_train_tensor=torch.tensor(Y_train.values,dtype=torch.float32).view(-1, 1)
X_test_tensor=torch.tensor(X_test,dtype=torch.float32)
Y_test_tensor=torch.tensor(Y_test.values,dtype=torch.float32).view(-1, 1)

#Guardamos el orden de las columnas
feature_columns=X.columns.tolist() if hasattr(X,"columns") else None
#Split interno
X_train_final, X_val, Y_train_final, Y_val= train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
#Convertir a tensores pyTorch
X_train_tensor=torch.tensor(X_train_final,dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train_final.values if hasattr(Y_train_final, "values") else Y_train_final,dtype=torch.float32).view(-1, 1)                        
X_val_tensor=torch.tensor(X_val,dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val.values if hasattr(Y_val, "values") else Y_val,dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test.values if hasattr(Y_test, "values") else Y_test,dtype=torch.float32).view(-1, 1)        
                                               
#Creamos el batch size y DataLoaders
batch_size=64
train_ds=TensorDataset(X_train_tensor, Y_train_tensor)
val_ds=TensorDataset(X_val_tensor,Y_val_tensor)
test_ds=TensorDataset(X_test_tensor,Y_test_tensor)
train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False)

print(f"Train samples:{len(train_ds)}, Val samples:{len(val_ds)}, Test samples:{len(test_ds)}")
#Elegimos el dispositivo (GPU preferiblemente)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando device:", device)
#Crear el modelo de red neuronal
# Crear el modelo de red neuronal con más capas
class PricePredictor(nn.Module):
    def __init__(self):
        super(PricePredictor, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)  # Capa 1: más neuronas
        self.fc2 = nn.Linear(128, 64)                # Capa 2
        self.fc3 = nn.Linear(64, 32)                 # Capa 3
        self.fc4 = nn.Linear(32, 16)                 # Capa 4
        self.fc5 = nn.Linear(16, 1)                  # Salida: precio

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # En la última capa no usamos ReLU para regresión
        return x
    
input_dim=X_train_tensor.shape[1]
model=PricePredictor().to(device)
#Definimos la función pérdida y optimizador
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.005)

print(model)

#Training + bucle de entrenamiento
n_epochs=30000
best_val_loss=float("inf")
save_path="audi.pth"
patience=30000
counter=0

for epoch in range(1,n_epochs+1):
    #Entrenamiento
    model.train()
    train_losses=[]
    for xb,yb in train_loader:
        xb=xb.to(device)
        yb=yb.to(device)

        optimizer.zero_grad()
        preds=model(xb)
        loss=criterion(preds,yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    #Validación
    model.eval()
    val_losses=[]
    with torch.no_grad():
        for xb,yb in val_loader:
            xb=xb.to(device)
            yb=yb.to(device)
            preds=model(xb)
            val_losses.append(criterion(preds,yb).item())

    avg_train=float(np.mean(train_losses))
    avg_val=float(np.mean(val_losses))

    print(f"Epoch {epoch}: Train loss: {avg_train:.4f}, Val loss: {avg_val:.4f}")

    #Early stopping
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), save_path)
        counter = 0
    else:
        counter += 1
    if epoch % 1 == 0:
        print(f"Epoch {epoch} - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
    if counter >= patience:
        print(f"Early stopping: no mejora en {patience} epochs.")
        break

#Evaluación final sobre el set de test y guardamos
    #Cargamos el modelo mejor guardado 

model.load_state_dict(torch.load(save_path,map_location=device))
model.eval()

#Predecir sobre test set
preds=[]
trues=[]
with torch.no_grad():
    for xb, yb in test_loader:
        xb=xb.to(device)
        out=model(xb).cpu().numpy().reshape(-1)
        preds.extend(out.tolist())
        trues.extend(yb.numpy().reshape(-1).tolist())
#Métricas
mae=mean_absolute_error(trues,preds)
rmse=mean_squared_error(trues,preds,squared=False)
r2=r2_score(trues,preds)

print("\Evaluación en TEST")
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
# Mostrar algunos ejemplos
print("\nEjemplos (predicho vs real):")
for p, t in list(zip(preds, trues))[:10]:
    print(f"{p:.0f}  \tvs\t {t:.0f}")

# Guardar scaler y lista de columnas para la inferencia posterior
joblib.dump(scaler, "scaler.save")
if feature_columns is not None:
    joblib.dump(feature_columns, "feature_columns.pkl")
print("\nModel, scaler y columnas guardadas: ", os.path.abspath(save_path), "scaler.save, feature_columns.pkl")