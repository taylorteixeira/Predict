# Predição de Preços de Laptops

Este projeto utiliza modelos de Machine Learning para prever o preço de laptops com base em um conjunto de dados.  O modelo treinado e o scaler são salvos para uso futuro.

## Dados

O dataset utilizado é `laptop_prices.csv`.  Ele contém informações sobre vários laptops, incluindo características como especificações técnicas e preço.  Os valores ausentes foram tratados preenchendo com a mediana para colunas numéricas e com "Unknown" para colunas categóricas.

## Pré-processamento

1. **Tratamento de Dados Ausentes:**  Valores ausentes em colunas numéricas foram preenchidos com a mediana da coluna, e em colunas categóricas com a string "Unknown".
2. **Codificação One-Hot:** Variáveis categóricas foram codificadas usando One-Hot Encoding com `pd.get_dummies(df, drop_first=True)`.
3. **Escalonamento:** As features numéricas foram escalonadas utilizando `StandardScaler` da biblioteca scikit-learn antes de serem alimentadas aos modelos.


## Modelos

Dois modelos de regressão foram treinados e comparados:

* **Regressão Linear:** Um modelo de regressão linear simples foi utilizado como linha de base.
* **Random Forest:** Um modelo de Random Forest foi treinado para comparação, visando um melhor desempenho.


## Métricas de Avaliação

Os modelos foram avaliados usando as seguintes métricas:

* **Mean Absolute Error (MAE):** A média do valor absoluto dos erros.
* **Mean Squared Error (MSE):** A média do quadrado dos erros.
* **R-squared (R²):** Um coeficiente de determinação que indica a proporção da variância da variável dependente que é previsível a partir da variável independente.


## Resultados

Os resultados obtidos mostram que o modelo **Random Forest** obteve um desempenho significativamente melhor que a Regressão Linear, conforme demonstrado pelos valores de MAE, MSE e R².

* **Regressão Linear:**  MAE muito alto, MSE extremamente alto, R² negativo, indicando um ajuste extremamente ruim. Isso provavelmente indica problemas com o dataset ou necessidade de mais recursos, ou tratamento de variáveis outliers.
* **Random Forest:** MAE = 177.11, MSE = 74694.77, R² = 0.855.  O R² indica que o modelo explica aproximadamente 85.5% da variância no preço dos laptops.

![image](https://github.com/user-attachments/assets/dd5debd3-5403-421a-bdf4-bf561e0b3523)

![image](https://github.com/user-attachments/assets/95399cde-4360-41b0-9553-f0434798d309)

## Modelos Salvos

Os modelos treinados (`Random Forest` e `Linear Regression`) e o scaler foram salvos para uso futuro usando a biblioteca `joblib`. Os arquivos salvos são:

* `laptop_price_model.pkl`: Modelo Random Forest (o melhor modelo).
* `scaler.pkl`: Objeto `StandardScaler` utilizado para escalonar os dados.
* `model_and_data.joblib`: Arquivo contendo o melhor modelo e o dataframe original.


### Dependências

* pandas
* scikit-learn
* joblib

## Exemplo de Requisição

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "Product": "Inspiron 15",
           "TypeName": "Notebook",
           "Ram": 8,
           "OS": "Windows 10"
         }'
		
