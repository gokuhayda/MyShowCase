import pandas as pd

class S3DataLoader:
    """
    Responsável apenas por carregar dados do S3.
    É um 'Boundary' (Fronteira) com o mundo externo.
    """
    
    def load_csv(self, s3_path: str) -> pd.DataFrame:
        """
        Lê um ficheiro CSV diretamente do S3.
        
        Args:
            s3_path: Caminho completo (ex: 's3://my-bucket/data.csv')
        """
        # Este print prova que o método foi chamado, mas o teste deve evitar o I/O real
        print(f"🔌 Conectando ao S3 para ler: {s3_path}")
        
        # Dependência externa: pandas.read_csv com protocolo s3://
        return pd.read_csv(s3_path)
