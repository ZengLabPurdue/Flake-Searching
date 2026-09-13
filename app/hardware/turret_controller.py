from config import RELATIVE_XY, RELATIVE_Z
from hardware.turret_api import turret

class TurretController:

    def __init__(
        self, 
        app,
        stage,
        turret_port,
        start_auto_focus_thread,
    ):
        self.app = app
        self.stage = stage
        self.turret = turret(turret_port)
        self.start_auto_focus_thread = start_auto_focus_thread

    def get_position(self):
        return self.turret.check_position()
    
    def turn_to_position(self, position, cancel_check=None, timeout=30):
        self.turret.turn_to_position(
            position,
            cancel_check=cancel_check,
            timeout=timeout,
        )

    def return_to_first_objective_and_focus(self):
        """Leave the microscope on objective 1 with a completed autofocus."""
        if self.get_position() == 1:
            self.app.set_magnification("2X")
            self.app.frame_processor.set_default_exposure()
            self.start_auto_focus_thread()
            return
        self.change_objective(1)

    def close(self):
        self.turret.close()

    def change_objective(
        self,
        position,
        cancel_check=None,
        autofocus=True,
    ):
        objective_map = {
            1: ("2X", RELATIVE_XY["2X"]["X"], RELATIVE_XY["2X"]["Y"], RELATIVE_Z["2X"]),
            2: ("10X", RELATIVE_XY["10X"]["X"], RELATIVE_XY["10X"]["Y"], RELATIVE_Z["10X"]),
            3: ("20X", RELATIVE_XY["20X"]["X"], RELATIVE_XY["20X"]["Y"], RELATIVE_Z["20X"]),
            4: (None, None, None, None),
            5: ("100X", RELATIVE_XY["100X"]["X"], RELATIVE_XY["100X"]["Y"], RELATIVE_Z["100X"]),
        }

        self.app.disable_buttons()

        try:
            current_position = self.get_position()

            if position == current_position:
                return

            current_x, current_y, current_z = self.stage.get_position()

            _, current_rel_x, current_rel_y, current_rel_z = objective_map[current_position]
            magnification, target_rel_x, target_rel_y, target_rel_z = objective_map[position]

            change_x = target_rel_x - current_rel_x
            change_y = target_rel_y - current_rel_y
            change_z = target_rel_z - current_rel_z

            moved_xy = self.stage.move_to_xy(
                current_x + change_x,
                current_y + change_y,
                cancel_check=cancel_check,
            )
            if not moved_xy:
                raise RuntimeError(
                    "The stage was busy before the objective XY offset move."
                )
            moved_z = self.stage.move_to_z(
                current_z + change_z,
                cancel_check=cancel_check,
            )
            if not moved_z:
                raise RuntimeError(
                    "The stage was busy before the objective Z offset move."
                )
            self.turn_to_position(
                position,
                cancel_check=cancel_check,
            )

            # Autofocus selects its objective-specific range and motion
            # profile from the app's active magnification.
            self.app.set_magnification(magnification)
            self.app.frame_processor.set_default_exposure()

            if position == 5:
                self.app.frame_processor.set_auto_exposure(True)
            if magnification is not None and autofocus:
                self.start_auto_focus_thread()

        finally:
            self.app.enable_buttons()
