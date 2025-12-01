from mars_rover.entities import Plateau, Position
from mars_rover.factory import DirectionFactory
from mars_rover.rover import Rover

def test_scenario_2():
    plateau = Plateau(5, 5)
    rover = Rover(plateau, Position(3, 3), DirectionFactory.from_char("E"))
    rover.execute("MMRMMRMRRM")
    assert str(rover) == "5 1 E"
