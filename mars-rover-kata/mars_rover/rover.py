from .entities import Position, Plateau
from .directions import Direction

class Rover:
    def __init__(self, plateau: Plateau, position: Position, direction: Direction):
        self.plateau = plateau
        self.position = position
        self.direction = direction

    def execute(self, commands: str):
        for command in commands:
            if command == "L":
                self.direction = self.direction.turn_left()
            elif command == "R":
                self.direction = self.direction.turn_right()
            elif command == "M":
                self._move()
            else:
                raise ValueError(f"Comando desconhecido: {command}")

    def _move(self):
        new_position = self.direction.move(self.position)
        if self.plateau.is_valid(new_position):
            self.position = new_position
        else:
            print(f"⚠️ Movimento bloqueado: {new_position} fora dos limites.")

    def __repr__(self):
        return f"{self.position.x} {self.position.y} {self.direction}"
