from .scoring_logic import ScoringLogic

class CustomerScoreOrchestrator:
    """
    Coordena o fluxo.
    Recebe as dependências (API e Modelo) no construtor (Injeção).
    """
    
    def __init__(self, api_client, ml_model):
        # INJEÇÃO DE DEPENDÊNCIA: O segredo da testabilidade.
        # Não criamos o cliente aqui, recebemos ele pronto.
        self.api_client = api_client
        self.ml_model = ml_model
        self.logic = ScoringLogic()
        
    def generate_score(self, customer_id: int) -> float:
        # 1. Buscar Dados (Usa dependência injetada)
        data = self.api_client.get_customer_data(customer_id)
        
        # 2. Calcular Base (Usa lógica pura)
        base = self.logic.calculate_base_score(
            data['age'], data['income'], data['history']
        )
        
        # 3. Predição ML (Usa dependência injetada)
        # Note: O modelo espera [[age, income, history]]
        features = [[data['age'], data['income'], data['history']]]
        prob = self.ml_model.predict_proba(features)[0][1]
        
        # 4. Finalizar
        return self.logic.calculate_final_score(base, prob)
