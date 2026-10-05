import carb
import numpy as np
import omni
import omni.appwindow  # Contains handle to keyboard


# THis can only be used after the scene is loaded 
class keyboard_cmd:
    def __init__(self,
                 base_command: np.array = np.array([0.0, 0.0, 0.0]),
                 input_keyboard_mapping: dict = {
                                        # forward command
                                        "W": [1.0, 0.0, 0.0],
                                        # backward command
                                        "S": [-1.0, 0.0, 0.0],
                                        # leftward command
                                        "A": [0.0, 1.0, 0.0],
                                        # rightward command
                                        "D": [0.0, -1.0, 0.0],
                                        # rise command
                                        "UP": [0.0, 0.0, 1.0],
                                        # sink command
                                        "DOWN": [0.0, 0.0, -1.0],
                                        }
                ) -> None:
        # Copy so the (mutable) default argument is never mutated in place by the
        # += / -= updates below, which would leak command state across instances.
        self._base_command = np.array(base_command, dtype=float)

        self._input_keyboard_mapping = input_keyboard_mapping

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)


    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Subscriber callback to when kit is updated."""
        # when a key is pressedor released  the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # on pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # on release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True


    def cleanup(self):
        # Actually release the carb subscription: just nulling the Python refs
        # left the callback registered in carb's input system, so it kept firing
        # after cleanup (and kept `self` alive). A second keyboard_cmd created
        # after cleanup() then received every key press TWICE (both callbacks
        # increment their _base_command), doubling the teleop command.
        if self._input is not None and self._sub_keyboard is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            except Exception:  # noqa: BLE001 - teardown must not raise mid-shutdown
                pass
        self._appwindow = None
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None
