from .directions import North, South, East, West

class DirectionFactory:
    """Cria instâncias de direção a partir de caracteres."""
    @staticmethod
    def from_char(c: str):
        mapping = {"N": North(), "S": South(), "E": East(), "W": West()}
        if c not in mapping:
            raise ValueError(f"Direção inválida: {c}")
        return mapping[c]
