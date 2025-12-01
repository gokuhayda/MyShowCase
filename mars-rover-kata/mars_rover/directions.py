from abc import ABC, abstractmethod
from .entities import Position

class Direction(ABC):
    """é um objeto que sabe andar e girar."""
    
    @abstractmethod
    def move(self, pos: Position) -> Position:
        """calcula a nova posição baseado na direção."""
        pass

    @abstractmethod
    def turn_left(self):
        pass

    @abstractmethod
    def turn_right(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class North(Direction):
    def move(self, pos): return Position(pos.x, pos.y + 1)
    def turn_left(self): return West()
    def turn_right(self): return East()
    def __repr__(self): return "N"


class South(Direction):
    def move(self, pos): return Position(pos.x, pos.y - 1)
    def turn_left(self): return East()
    def turn_right(self): return West()
    def __repr__(self): return "S"


class East(Direction):
    def move(self, pos): return Position(pos.x + 1, pos.y)
    def turn_left(self): return North()
    def turn_right(self): return South()
    def __repr__(self): return "E"


class West(Direction):
    def move(self, pos): return Position(pos.x - 1, pos.y)
    def turn_left(self): return South()
    def turn_right(self): return North()
    def __repr__(self): return "W"
