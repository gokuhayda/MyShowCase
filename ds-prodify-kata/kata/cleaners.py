import pandas as pd
from typing import Protocol

# Interface
class CleanerStrategy(Protocol):
    def clean(self, df: pd.DataFrame) -> pd.DataFrame: ...

# Implementação Concreta
class SalesDataCleaner:
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🧹 Limpando dados...")
        # Regras de negócio:
        # 1. Remove linhas com valores nulos
        df = df.dropna().copy()
        # 2. Cria coluna 'total'
        df['total'] = df['qtd'] * df['preco']
        return df
