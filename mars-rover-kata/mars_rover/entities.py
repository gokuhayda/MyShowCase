from dataclasses import dataclass

@dataclass(frozen=True)
class Position:
    """Representa coordenadas (x, y) no platô."""
    x: int
    y: int


@dataclass(frozen=True)
class Plateau:
    """Representa o ambiente onde o rover navega.
    Define até onde o rover pode ir, evitando sair do mapa.
    """
    width: int
    height: int

    def is_valid(self, pos: Position) -> bool:
        return 0 <= pos.x <= self.width and 0 <= pos.y <= self.height
