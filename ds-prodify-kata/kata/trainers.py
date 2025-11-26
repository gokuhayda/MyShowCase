import pandas as pd
from typing import Any

# Simulando sklearn para não precisar da biblioteca instalada
class LinearRegression:
    def fit(self, X, y): print(f"🤖 Treinando modelo com {len(X)} linhas...")

class ModelTrainer:
    def train(self, df: pd.DataFrame) -> Any:
        # Prepara features (X) e target (y)
        X = df[['qtd', 'preco']]
        y = df['total']
        
        # Treina o modelo
        model = LinearRegression()
        model.fit(X, y)
        return model
