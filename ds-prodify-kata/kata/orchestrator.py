from dataclasses import dataclass
from typing import Any
# Imports relativos funcionam bem dentro do pacote
from .loaders import LoaderStrategy
from .cleaners import CleanerStrategy
from .trainers import ModelTrainer

@dataclass
class TrainingPipeline:
    """
    Orquestra o fluxo. Perceba que ele pede as peças no construtor.
    Isso é Injeção de Dependência.
    """
    loader: LoaderStrategy
    cleaner: CleanerStrategy
    trainer: ModelTrainer
    
    def run(self, input_path: str) -> Any:
        print("\n🚀 Iniciando Pipeline de Produção...")
        
        # Passo 1: Ingestão
        raw_data = self.loader.load(input_path)
        
        # Passo 2: Limpeza
        clean_data = self.cleaner.clean(raw_data)
        
        # Passo 3: Treino
        model = self.trainer.train(clean_data)
        
        print("✅ Pipeline finalizada com sucesso!")
        return model
