import pandas as pd
from joblib import dump, load
from flask import Flask, request, jsonify, render_template

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler

# # Carregar o dataset e treinar o modelo
# dataset_path = "laptop_prices.csv"  # Certifique-se de usar o caminho correto
# df = pd.read_csv(dataset_path)

# # Preenchendo valores ausentes
# df = df.dropna()  # Remove qualquer linha com valores nulos

# # Codificação de variáveis categóricas para números
# categorical_cols = df.select_dtypes(include=['object']).columns
# df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# # Separando variáveis independentes e dependentes
# X = df_encoded.drop("Price_euros", axis=1)
# y = df_encoded["Price_euros"]

# # Divisão dos dados em treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Normalização
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Treinamento do modelo Random Forest
# random_forest_model = RandomForestRegressor(random_state=42)
# random_forest_model.fit(X_train_scaled, y_train)

# # Avaliação do modelo
# y_pred_rf = random_forest_model.predict(X_test_scaled)
# mae_rf = mean_absolute_error(y_test, y_pred_rf)
# mse_rf = mean_squared_error(y_test, y_pred_rf)
# r2_rf = r2_score(y_test, y_pred_rf)

# print("\nRandom Forest:")
# print("  MAE:", mae_rf)
# print("  MSE:", mse_rf)
# print("  R²:", r2_rf)

# # Salvando o modelo, scaler e colunas do conjunto de treino
# dump({'model': random_forest_model, 'scaler': scaler, 'X_train_columns': X.columns}, 'model_and_scaler.joblib')

# API Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

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

        # Carregar o modelo e scaler
        model_data = load('model_and_scaler.joblib')
        random_forest_model = model_data['model']
        scaler = model_data['scaler']
        X_train_columns = model_data['X_train_columns']

        # Criar DataFrame para entrada
        input_data = pd.DataFrame([data])

        # Codificar variáveis categóricas com `get_dummies`
        input_data_encoded = pd.get_dummies(input_data)

        # Reindexar para garantir compatibilidade com as colunas do modelo
        input_data_encoded = input_data_encoded.reindex(columns=X_train_columns, fill_value=0)

        # Normalizar os dados
        input_data_scaled = scaler.transform(input_data_encoded)

        # Fazer a previsão
        prediction = random_forest_model.predict(input_data_scaled)

        return jsonify({"predicted_price": round(prediction[0], 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
