import pickle
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Carregar o modelo e o scaler
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Ajuste para seus dados de entrada
    input_data = np.array([[
        data['Product'], 
        data['Ram'], 
        data['OS']
    ]])

    scaled_input = scaler.transform(input_data)
    prediction = rf_model.predict(scaled_input)
    
    return jsonify({"predicted_price": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
