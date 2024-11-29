ARIMA Predictor

Bem-vindo ao ARIMA Predictor, um pacote Python para previsão de séries temporais usando ARIMA e SARIMAX, com suporte para variáveis exógenas. Este pacote é projetado para facilitar a análise de dados temporais e prever tendências futuras de maneira eficiente.
📌 Sobre o Pacote

Este pacote tem como objetivo:

    Automatizar o treinamento de modelos ARIMA e SARIMAX.
    Simplificar o processo de previsão de séries temporais, incluindo suporte a variáveis exógenas.
    Oferecer ferramentas para análise e visualização de resultados preditivos.

📂 Estrutura do Projeto

O projeto está organizado da seguinte maneira:

etssaxt/ ├── tsSarimax/

🚀 Como Usar

Aqui está um exemplo básico de como utilizar o pacote para treinar um modelo e prever tendências futuras:

from tsSarimax.models.predictor import ARIMAPredictor
import pandas as pd

# Carregar os dados
data = pd.read_csv('time_series_data.csv')

# Inicializar o preditor
predictor = ARIMAPredictor(data, target_col='target_column')

# Treinar o modelo e prever
train_preds, test_preds, future_preds, model_fit, train_metrics, test_metrics = predictor.train_and_predict_sarimax(
    feature_cols=['feature1', 'feature2'],
    seasonal=True,
    stepwise=True,
    forecast_horizon=30
)

# Exibir resultados
print("Train Metrics:", train_metrics)
print("Test Metrics:", test_metrics)


