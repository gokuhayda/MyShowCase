🚀 eTechShowCase - etssax

Bem-vindo ao etssax, um projeto inovador que torna o trabalho com modelos SARIMAX mais intuitivo, especialmente ao incorporar variáveis exógenas. Este pacote Python foi projetado para simplificar o processo de previsão de séries temporais e otimizar análises de tendências futuras.


---

📌 Destaques

💡 Novidade: Variáveis Exógenas Mais Intuitivas

Sempre quis usar variáveis exógenas no SARIMAX, mas achou a implementação complicada? Agora, com o etssax, você pode integrá-las de forma prática e eficiente, sem alterar o comportamento do pacote original.

✨ Principais Funcionalidades

Suporte direto a variáveis exógenas durante o treinamento de modelos SARIMAX.

Processo otimizado para integrar variáveis adicionais na previsão de tendências futuras.

Busca stepwise automatizada para identificar o melhor modelo com base no AIC.



---

📂 Estrutura do Projeto

O projeto está organizado da seguinte maneira:

etssaxt/
├── tsSarimax/


---

🚀 Como Usar

Aqui está um exemplo simples para começar a trabalhar com o etssax:

from tsSarimax.models.predictor import ARIMAPredictor
import pandas as pd

# Carregar os dados
data = pd.read_csv('time_series_data.csv')

# Inicializar o preditor
predictor = ARIMAPredictor(data, target_col='target_column')

# Treinar o modelo e prever
train_preds, test_preds, future_preds, model_fit, train_metrics, test_metrics = predictor.train_and_predict_sarimax(
    feature_cols=['feature1', 'feature2'],  # Suporte a variáveis exógenas
    seasonal=True,
    stepwise=True,
    forecast_horizon=30
)

# Exibir resultados
print("Train Metrics:", train_metrics)
print("Test Metrics:", test_metrics)


---

🛠️ Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para melhorar este projeto.


---
