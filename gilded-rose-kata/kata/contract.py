from abc import ABC, abstractmethod
from domain import Item

# ============================================================
# 2. CONTRACT (Interface / Abstração)
# Define "o que" deve ser feito, não "como".
# Todas as estratégias concretas devem implementar este contrato.
# ============================================================

class UpdateStrategy(ABC):
    """
    Contrato para estratégias de atualização de itens.
    
    Cada estratégia concreta deve implementar o método `update`,
    recebendo um objeto Item e aplicando suas regras específicas.
    
    Este padrão garante o DIP (Dependency Inversion Principle):
    o sistema depende de abstrações, não de implementações.
    """
    
    @abstractmethod
    def update(self, item: Item):
        """Executa a lógica de atualização do item."""
        pass
