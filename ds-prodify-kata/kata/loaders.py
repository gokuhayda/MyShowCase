import pandas as pd
from typing import Protocol

# Interface (Contrato)
class LoaderStrategy(Protocol):
    def load(self, path: str) -> pd.DataFrame: ...

# Implementação Concreta
class CsvLoader:
    def load(self, path: str) -> pd.DataFrame:
        print(f"📂 Lendo CSV: {path}")
        # Simulação de retorno para o exercício
        # Em produção, aqui estaria: return pd.read_csv(path)
        return pd.DataFrame({
            'qtd': [1, 2, None, 4], 
            'preco': [10.0, 20.0, 30.0, 40.0]
        })
