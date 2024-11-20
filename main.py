from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Carregue os dados
df = pd.read_csv('laptop_prices.csv')

# Prepare os dados
df = pd.get_dummies(df, drop_first=True)
X = df.drop(['Price_euros'], axis=1)
y = df['Price_euros']

# Divida os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treine o modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Defina o modelo de entrada para a API
class LaptopInput(BaseModel):
    Product: str
    Ram: int
    OS: str

# Crie a instância do FastAPI
app = FastAPI()

@app.post("/predict")
def predict_laptop_price(data: LaptopInput):
    # Transforme as variáveis de entrada para o formato do modelo
    input_data = pd.DataFrame([data.dict()])
    input_data = pd.get_dummies(input_data, drop_first=True)
    
    # Alinhe as colunas do input com as do modelo
    missing_cols = set(X.columns) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0
    input_data = input_data[X.columns]

    # Normalização
    input_data = scaler.transform(input_data)
    
    # Previsão
    prediction = rf_model.predict(input_data)
    return {"predicted_price": prediction[0]}

