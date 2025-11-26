import requests
# Imagine que isto carrega um modelo pesado de 5GB
# from sklearn.linear_model import LogisticRegression
# MODEL = LogisticRegression() 

# Simulação para o exercício não quebrar sem scikit-learn instalado
class FakeModel:
    def predict_proba(self, X): return [[0.2, 0.8]] # 80% chance
MODEL = FakeModel()

def generate_customer_score(customer_id: int) -> float:
    """
    CÓDIGO LEGADO: Tente não chorar.
    - Acoplado a API externa
    - Lógica misturada com I/O
    - Dependência global oculta
    """
    # I/O Misturado (Boundary)
    print(f"Chamando API para ID {customer_id}...")
    response = requests.get(f"https://api.fake.com/customers/{customer_id}")
    data = response.json()
    
    # Lógica de Negócio (Core) - Difícil de testar isoladamente!
    age = data["age"]
    income = data["income"]
    history = data["history"]
    
    base_score = (age * 0.1) + (income / 1000) + (history * 5)
    
    # Dependência Global (Boundary)
    # Se quisermos testar com outro modelo, não conseguimos!
    ml_prob = MODEL.predict_proba([[age, income, history]])[0][1]
    
    final_score = base_score * ml_prob
    
    return final_score
