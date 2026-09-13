import math
import tkinter as tk
from tkinter import Frame, Label, StringVar
from tkinter import ttk

class FocusPanel:
    def __init__(
        self,
        parent,
        app,
        focus_controller,
        default_range=None,
        default_velocity=None,
        default_acceleration=None,
    ):
        self.parent = parent
        self.app = app
        self.focus_controller = focus_controller

        self.default_range = default_range
        self.default_velocity = default_velocity
        self.default_acceleration = default_acceleration

        self.sharpness_var = tk.StringVar(value="Sharpness: Unknown")

        self.frame = self._build_panel()

    def _build_panel(self):
        panel = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=203
        )
        panel.place(relx=1.0, rely=0.0, anchor="ne")

        background = Frame(
            panel,
            bg="white",
            width=200,
            height=201
        )
        background.place(x=2, y=0)

        title = Label(
            panel,
            text="Focus Control",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title.place(relx=0.5, y=5, anchor="n")

        sharpness_label = Label(
            panel,
            textvariable=self.sharpness_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )
        sharpness_label.place(relx=0.5, y=35, anchor="n")

        label_x = 10
        entry_x = 130

        range_label = Label(
            panel,
            text="Range (±):",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        range_label.place(relx=0.0, rely=0.0, x=label_x, y=65)

        self.range_var = StringVar(
            value=self._default_entry_value(self.default_range)
        )
        self.range_entry = ttk.Entry(
            panel,
            textvariable=self.range_var,
            width=8
        )
        self.range_entry.place(relx=0.0, rely=0.0, x=entry_x, y=65)

        velocity_label = Label(
            panel,
            text="Velocity:",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        velocity_label.place(relx=0.0, rely=0.0, x=label_x, y=95)

        self.velocity_var = StringVar(
            value=self._default_entry_value(self.default_velocity)
        )
        self.velocity_entry = ttk.Entry(
            panel,
            textvariable=self.velocity_var,
            width=8
        )
        self.velocity_entry.place(relx=0.0, rely=0.0, x=entry_x, y=95)

        acceleration_label = Label(
            panel,
            text="Acceleration:",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        acceleration_label.place(relx=0.0, rely=0.0, x=label_x, y=125)

        self.acceleration_var = StringVar(
            value=self._default_entry_value(self.default_acceleration)
        )
        self.acceleration_entry = ttk.Entry(
            panel,
            textvariable=self.acceleration_var,
            width=8
        )
        self.acceleration_entry.place(relx=0.0, rely=0.0, x=entry_x, y=125)

        self.auto_focus_btn = ttk.Button(
            panel,
            text="Auto Focus",
            style="Normal.TButton",
            command=self.run_auto_focus
        )
        self.auto_focus_btn.place(relx=0.5, y=160, anchor="n")

        self.app.register_button(self.auto_focus_btn)

        return panel

    def update_sharpness(self, sharpness):
        self.sharpness_var.set(f"Sharpness: {sharpness:.3f}")

    @staticmethod
    def _default_entry_value(value):
        return "Auto" if value is None else str(value)

    @staticmethod
    def _optional_positive_value(value, integer=False):
        value = str(value).strip()
        if not value or value.casefold() in {"auto", "default"}:
            return None
        parsed = float(value)
        if not math.isfinite(parsed) or not parsed > 0:
            raise ValueError
        if integer:
            if not parsed.is_integer():
                raise ValueError
            return int(parsed)
        return parsed

    def run_auto_focus(self):
        if not self.app.hardware_controls_available():
            return
        try:
            focus_range = self._optional_positive_value(
                self.range_var.get()
            )
            z_velo = self._optional_positive_value(
                self.velocity_var.get(),
                integer=True,
            )
            z_accel = self._optional_positive_value(
                self.acceleration_var.get(),
                integer=True,
            )
            self.focus_controller.start_auto_focus_thread(
                focus_range=focus_range,
                z_velo=z_velo,
                z_accel=z_accel,
            )

        except (TypeError, ValueError):
            self.sharpness_var.set("Invalid focus values")
            return
