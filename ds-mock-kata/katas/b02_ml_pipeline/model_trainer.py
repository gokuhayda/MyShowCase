from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

class ModelTrainer:
    """
    Responsável por treinar e avaliar modelos.
    Em produção, isso é lento e pesado.
    """
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test) -> float:
        print("🌲 Iniciando treino da Random Forest (Lento)...")
        
        # Dependência Externa: RandomForestClassifier
        # Se não mockarmos, ele vai rodar o algoritmo matemático real aqui.
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Passo 1: Treino
        model.fit(X_train, y_train)
        
        # Passo 2: Predição
        predictions = model.predict(X_test)
        
        # Passo 3: Métrica
        return accuracy_score(y_test, predictions)
