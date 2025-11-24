# ==========================================
# SIMULAÇÃO DE USO (Factory manual no main)
# ==========================================
if __name__ == "__main__":
    weight = 10.0 # 10kg
    
    # Cenário 1: Cliente escolheu Correios
    # Injetamos a estratégia CORREIOS no serviço
    service_correios = ShippingService(strategy=CorreiosStrategy())
    cost = service_correios.get_shipping_cost(weight)
    print(f"Custo Correios: R$ {cost:.2f}")
    
    # Cenário 2: Cliente mudou para FedEx
    # O serviço é recriado ou reconfigurado (fácil de testar!)
    service_fedex = ShippingService(strategy=FedExStrategy())
    cost = service_fedex.get_shipping_cost(weight)
    print(f"Custo FedEx:    R$ {cost:.2f}")
