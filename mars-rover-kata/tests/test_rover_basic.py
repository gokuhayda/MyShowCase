from mars_rover.entities import Plateau, Position
from mars_rover.factory import DirectionFactory
from mars_rover.rover import Rover

def test_scenario_1():
    plateau = Plateau(5, 5)
    rover = Rover(plateau, Position(1, 2), DirectionFactory.from_char("N"))
    rover.execute("LMLMLMLMM")
    assert str(rover) == "1 3 N"
