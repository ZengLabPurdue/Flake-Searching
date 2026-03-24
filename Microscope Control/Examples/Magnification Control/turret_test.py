import os
import sys
from pathlib import Path

home_dir = os.path.dirname(os.path.abspath(__file__))
turret_api_path = Path(home_dir) / "Turret API"

sys.path.insert(0, str(turret_api_path))

from turret_api import TurretController as tc

#turret = tr(f"COM{sys.argv[4]}")
turret = tc(f"COM4")
timeout = 0
while (timeout != 5):
    if (turret.check_if_log_in() == False):
        timeout += 1
    else:
        break

while True:
    print(turret.check_position())
    position = input("Go to position -> ")
    turret.turn_to_position(int(position))