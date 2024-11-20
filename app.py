import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
from flask import Flask, request, jsonify
import numpy as np

# Carregar o dataset
dataset_path = "laptop_prices.csv"
df = pd.read_csv(dataset_path)

# Identificar colunas categóricas e numéricas
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Preenchendo valores ausentes
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# Codificação das variáveis categóricas
df_encoded = pd.get_dummies(df, drop_first=True)

# Separando variáveis independentes e dependentes
X = df_encoded.drop("Price_euros", axis=1)
y = df_encoded["Price_euros"]

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelos
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train, y_train)

# Predições
y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_rf = random_forest_model.predict(X_test)

# Avaliação
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Exibindo os resultados
print("Regressão Linear:")
print("  MAE:", mae_linear)
print("  MSE:", mse_linear)
print("  R²:", r2_linear)

print("\nRandom Forest:")
print("  MAE:", mae_rf)
print("  MSE:", mse_rf)
print("  R²:", r2_rf)

if r2_rf > r2_linear:
    print("\nO modelo Random Forest teve um desempenho melhor.")
else:
    print("\nO modelo de Regressão Linear teve um desempenho melhor.")

# Salvando o modelo, scaler e colunas do conjunto de treino
dump({'model': random_forest_model, 'scaler': scaler, 'X_train_columns': X.columns}, 'model_and_scaler.joblib')

# API Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obter dados da requisição
        data = request.json

        # Validar dados de entrada
        required_fields = ["Product", "TypeName", "Ram", "OS"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Campo {field} é obrigatório.")

        # Organizar os dados conforme esperado
        input_data = {
            "Company": "Unknown",
            "Product": data.get("Product", "Unknown"),
            "TypeName": data.get("TypeName", "Unknown"),
            "Inches": 0,
            "Ram": data.get("Ram", 0),
            "OS": data.get("OS", "Unknown"),
            "Weight": 0,
            "Screen": "Unknown",
            "ScreenW": 0,
            "ScreenH": 0,
            "Touchscreen": "Unknown",
            "IPSpanel": "Unknown",
            "RetinaDisplay": "Unknown",
            "CPU_company": "Unknown",
            "CPU_freq": 0,
            "CPU_model": "Unknown",
            "PrimaryStorage": 0,
            "SecondaryStorage": 0,
            "PrimaryStorageType": "Unknown",
            "SecondaryStorageType": "Unknown",
            "GPU_company": "Unknown",
            "GPU_model": "Unknown"
        }

        # Criar DataFrame para aplicar o mesmo pré-processamento do treino
        input_df = pd.DataFrame([input_data])

        # Codificar variáveis categóricas da mesma forma que no treinamento
        input_df_encoded = pd.get_dummies(input_df, drop_first=True)

        # Carregar o modelo e scaler
        model_data = load('model_and_scaler.joblib')
        random_forest_model = model_data['model']
        scaler = model_data['scaler']
        X_train_columns = model_data['X_train_columns']

        # Garantir que a entrada tenha as mesmas colunas que o conjunto de treino
        input_df_encoded = input_df_encoded.reindex(columns=X_train_columns, fill_value=0)

        # Verifique o número de colunas após o reindex
        if input_df_encoded.shape[1] != X_train_columns.shape[0]:
            raise ValueError(f"Mismatch in columns: expected {X_train_columns.shape[0]} columns, but got {input_df_encoded.shape[1]} columns.")

        # Normalizar os dados
        input_data_scaled = scaler.transform(input_df_encoded)

        # Fazer a previsão
        prediction = random_forest_model.predict(input_data_scaled)

        return jsonify({"predicted_price": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)