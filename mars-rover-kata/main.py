from mars_rover.entities import Plateau, Position
from mars_rover.factory import DirectionFactory
from mars_rover.rover import Rover

if __name__ == "__main__":
    plateau = Plateau(5, 5)

    # Rover 1
    rover1 = Rover(plateau, Position(1, 2), DirectionFactory.from_char("N"))
    commands1 = "LMLMLMLMM"
    print("=== Rover 1 ===")
    print("Início:", rover1)
    rover1.execute(commands1)
    print("Final :", rover1)
    assert str(rover1) == "1 3 N"

    # Rover 2
    rover2 = Rover(plateau, Position(3, 3), DirectionFactory.from_char("E"))
    commands2 = "MMRMMRMRRM"
    print("\n=== Rover 2 ===")
    print("Início:", rover2)
    rover2.execute(commands2)
    print("Final :", rover2)
    assert str(rover2) == "5 1 E"

    print("\n🎉 Todos os testes passaram!")
