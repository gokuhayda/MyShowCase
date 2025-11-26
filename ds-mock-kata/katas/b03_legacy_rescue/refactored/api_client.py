import requests

class CustomerApiClient:
    """Responsabilidade Única: Falar com a API externa."""
    
    def get_customer_data(self, customer_id: int) -> dict:
        # Em produção real, aqui haveria tratamento de erros, retries, etc.
        response = requests.get(f"https://api.fake.com/customers/{customer_id}")
        return response.json()
