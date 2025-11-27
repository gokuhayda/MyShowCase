import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import json
import os

# 1. Configuração (Idealmente viria de um params.yaml)
SEED = 42
TEST_SIZE = 0.2

def load_data():
    # Simulando um dataset. Em prod, seria lido do DVC/S3.
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def train():
    print("🔄 Carregando dados...")
    df = load_data()
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    
    print("🤖 Treinando modelo...")
    model = RandomForestClassifier(n_estimators=10, random_state=SEED)
    model.fit(X_train, y_train)
    
    # Avaliação
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    
    print(f"📊 Métricas: Accuracy={acc:.2f}, F1={f1:.2f}")
    
    # 2. SALVAR MÉTRICAS (Crucial para CD4ML)
    # O GitHub Actions vai ler este arquivo para gerar relatório
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump({"accuracy": acc, "f1": f1}, f)
        
    # Salvar modelo
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("✅ Modelo salvo em models/model.pkl")
