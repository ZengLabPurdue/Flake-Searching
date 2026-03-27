import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from tkinter import *
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from PIL import Image, ImageTk

from pathlib import Path
home_dir = os.path.dirname(os.path.abspath(__file__))
api_path = Path(home_dir) / "Turret API"

import amcam
from prior_api import Prior_Controller
from turret_api import Turret_Controller
import chip_edge_classifier

DLL_PATH = os.getcwd() + r"\PriorSDK1.9.2\x64\PriorScientificSDK.dll"
PRIOR_COM_PORT = sys.argv[1]
TURRET_COM_PORT = sys.argv[2]
DEFAULT_EXPOSURE = 60

CENTER_CROP_WIDTH = 3000
CENTER_CROP_HEIGHT = 3000

X_SIZE_2 = 10568
Y_SIZE_2 = 7506

Y_SIZE_10 = 3660 
X_SIZE_10 = 2435

MAGNIFICATION = 2

try:
    pc = Prior_Controller(PRIOR_COM_PORT, DLL_PATH)
    tc = Turret_Controller(TURRET_COM_PORT)
    pc.get_curr_pos()
    x_pos = pc.x
    y_pos = pc.y
    z_pos = pc.z
except Exception as e:
    print("Failed to connect to Prior Controller:", e)
    sys.exit(1)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Scanning App")
        self.main_frame = Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill=BOTH, expand=True)

        self.map_canvas = Canvas(self.main_frame, bg="white")
        self.map_canvas.pack(fill=BOTH, expand=True)

        self.true_map = np.zeros((3000, 3000, 3), dtype=np.uint8)
        self.filter_map = np.zeros((3000, 3000), dtype=np.int8)

        self.img_label = Label(self.main_frame, bg="#f0f0f0")
        self.img_label.pack(fill=BOTH, expand=True)

        self.filter_on = False
        self.view_mode = None
        self.set_view("Camera View", False)

        self.scan_running = False

        self.step_size = 1000
        self.hold_speed = 2600
        self.hold_job = None
        self.is_hold = False

        self.hcam = None
        self.buf = None
        self.prevImg = None
        self.width = 0
        self.height = 0
        
        self.panels = []

        self.panels.append({
            "name": "Info Panel",
            "frame": self.init_info_panel(),
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Control Panel",
            "frame": self.init_manual_control_panel(),
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Capture Image Panel",
            "frame": self.init_capture_image_panel(),
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Adjust Exposure Panel",
            "frame": self.init_adjust_exposure_panel(),
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Objective Control Panel",
            "frame": self.init_objective_control_panel(),
            "var": BooleanVar(value=False)
        })

        self.panels.append({
            "name": "Focus Panel",
            "frame": self.init_focus_panel(),
            "var": BooleanVar(value=False)
        })

        self.update_panels()

        self.init_menu_bar()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------- Initialization -------------

    def init_menu_bar(self):
        menu_bar = Menu(root)
        file_menu = Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Quit", command=self.on_close)
        menu_bar.add_cascade(label="File", menu=file_menu)

        view_menu = Menu(menu_bar, tearoff=0)

        map_menu = Menu(view_menu, tearoff=0)
        map_menu.add_radiobutton(label="No Filter", command=lambda: self.set_view("Map", False))
        map_menu.add_radiobutton(label="Filter", command=lambda: self.set_view("Map", True))

        camera_menu = Menu(view_menu, tearoff=0)
        camera_menu.add_radiobutton(label="No Filter", command=lambda: self.set_view("Camera View", False))
        camera_menu.add_radiobutton(label="Filter", command=lambda: self.set_view("Camera View", True))

        view_menu.add_cascade(label="Map", menu=map_menu)
        view_menu.add_cascade(label="Camera View", menu=camera_menu)

        menu_bar.add_cascade(label="View", menu=view_menu)

        panel_menu = Menu(menu_bar, tearoff=0)

        for panel in self.panels:
            panel_menu.add_checkbutton(
                label=panel["name"],
                variable=panel["var"],
                command=self.update_panels
            )

        menu_bar.add_cascade(label="Panels", menu=panel_menu)

        scan_menu = Menu(menu_bar, tearoff=0)
        scan_menu.add_command(label="Run Full Scan", command=lambda: None)
        scan_menu.add_command(label="Run Chip Mapping Scan", command=self.run_chip_mapping_scan)
        scan_menu.add_command(label="Run Flake Finding Scan", command=lambda: None)
        menu_bar.add_cascade(label="Scan", menu=scan_menu)

        root.config(menu=menu_bar)

    # Panel Initialization 

    def init_info_panel(self):

        self.info_panel = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=114
        )
        self.info_panel.place(relx=1.0, rely=0.0, anchor="ne")

        self.info_background = Frame(
            self.info_panel,
            bg="white",
            width=200,
            height=112
        )
        self.info_background.place(x=2, y=0)

        title_label = Label(
            self.info_panel,
            text="Info",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title_label.place(relx=0.5, y=10, anchor="n")

        self.info_magnification_label = Label(
            self.info_panel,
            text=f"Magnification: {MAGNIFICATION}x",
            bg="white",
            fg="black"
        )
        self.info_magnification_label.place(relx=0.5, y=35, anchor="n")


        self.info_progress_label = Label(
            self.info_panel,
            text="Progress: Not Started",
            bg="white",
            fg="black"
        )
        self.info_progress_label.place(relx=0.5, y=55, anchor="n")


        self.info_time_label = Label(
            self.info_panel,
            text="Time Elapsed: Not Started",
            bg="white",
            fg="black"
        )
        self.info_time_label.place(relx=0.5, y=75, anchor="n")

        return self.info_panel

    def init_manual_control_panel(self):

        self.manual_control_panel = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=444
        )
        self.manual_control_panel.place(relx=1.0, rely=0.0, anchor="ne")

        self.manual_control_background = Frame(
            self.manual_control_panel,
            bg="white",
            width=200,
            height=442
        )
        self.manual_control_background.place(x=2, y=0)

        title_label = Label(
            self.manual_control_panel,
            text="Manual Control",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title_label.place(relx=0.5, y=10, anchor="n")

        label_x = 10
        entry_x = 130

        step_label = Label(
            self.manual_control_panel,
            text="Step Size (µm):",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        step_label.place(relx=0.0, rely=0.0, x=label_x, y=45)

        self.step_size_var = StringVar(value=str(self.step_size))
        self.step_entry = ttk.Entry(
            self.manual_control_panel,
            textvariable=self.step_size_var,
            width=8
        )
        self.step_entry.place(relx=0.0, rely=0.0, x=entry_x, y=45)

        hold_speed_label = Label(
            self.manual_control_panel,
            text="Hold Speed (µm/s):",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        hold_speed_label.place(relx=0.0, rely=0.0, x=label_x, y=75)

        self.hold_speed_var = StringVar(value=str(self.hold_speed))
        self.hold_speed_var.trace_add("write", self.on_hold_speed_change)
        self.hold_speed_entry = ttk.Entry(
            self.manual_control_panel,
            textvariable=self.hold_speed_var,
            width=8
        )
        self.hold_speed_entry.place(relx=0.0, rely=0.0, x=entry_x, y=75)

        x_label = Label(
            self.manual_control_panel,
            text="X (µm):",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        x_label.place(relx=0.0, rely=0.0, x=label_x, y=105)

        self.x_coord_var = StringVar(value=str(x_pos))
        self.x_coord_entry = ttk.Entry(
            self.manual_control_panel,
            textvariable=self.x_coord_var,
            width=8
        )
        self.x_coord_entry.place(relx=0.0, rely=0.0, x=entry_x, y=105)

        y_label = Label(
            self.manual_control_panel,
            text="Y (µm):",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        y_label.place(relx=0.0, rely=0.0, x=label_x, y=135)

        self.y_coord_var = StringVar(value=str(y_pos))
        self.y_coord_entry = ttk.Entry(
            self.manual_control_panel,
            textvariable=self.y_coord_var,
            width=8
        )
        self.y_coord_entry.place(relx=0.0, rely=0.0, x=entry_x, y=135)

        style = ttk.Style()
        style.configure("Normal.TButton", font="TkDefaultFont")
        style.configure("Normal.TButton", background="white")
        style.configure("Normal.TButton", relief="flat")

        self.reset_button = ttk.Button(
            self.manual_control_panel,
            text="Set Origin",
            style="Normal.TButton",
            command=self.set_origin
        )
        self.reset_button.place(relx=0.5, y=170, anchor="n")

        self.move_to_button = ttk.Button(
            self.manual_control_panel,
            text="Move to (x,y)",
            style="Normal.TButton",
            command=self.go_to_position
        )
        self.move_to_button.place(relx=0.5, y=205, anchor="n")

        self.XY_manual_control_button_panel = Frame(
            self.manual_control_panel,
            bg="white",
            width=120, 
            height=90 
        )
        self.XY_manual_control_button_panel.place(relx=0.5, x=0, y=240, anchor="n")
        self.XY_manual_control_button_panel.pack_propagate(False)

        controls = Frame(self.XY_manual_control_button_panel, bg="white")
        controls.pack(expand=True, fill="both")

        style = ttk.Style()
        style.configure("Arrow.TButton", font=("TkDefaultFont", 15), padding=5)
        style.configure("Arrow.TButton", background="white")
        style.configure("Arrow.TButton", relief="flat")

        self.btn_forward = ttk.Button(controls, text="▴", style="Arrow.TButton")
        self.btn_backward = ttk.Button(controls, text="▾", style="Arrow.TButton")
        self.btn_left = ttk.Button(controls, text="◂", style="Arrow.TButton")
        self.btn_right = ttk.Button(controls, text="▸", style="Arrow.TButton")

        self.btn_forward.bind("<ButtonPress-1>", self.on_press_forward)
        self.btn_forward.bind("<ButtonRelease-1>", self.on_release_forward)
        self.btn_backward.bind("<ButtonPress-1>", self.on_press_backward)
        self.btn_backward.bind("<ButtonRelease-1>", self.on_release_backward)
        self.btn_left.bind("<ButtonPress-1>", self.on_press_left)
        self.btn_left.bind("<ButtonRelease-1>", self.on_release_left)
        self.btn_right.bind("<ButtonPress-1>", self.on_press_right)
        self.btn_right.bind("<ButtonRelease-1>", self.on_release_right)

        for r in [0, 1]:
            controls.rowconfigure(r, weight=1)
        for c in [0, 1, 2]:
            controls.columnconfigure(c, weight=1)

        self.btn_forward.grid(row=0, column=1, sticky="nsew")
        self.btn_left.grid(row=1, column=0, sticky="nsew")
        self.btn_right.grid(row=1, column=2, sticky="nsew")
        self.btn_backward.grid(row=1, column=1, sticky="nsew")

        z_label = Label(
            self.manual_control_panel,
            text="Z (µm):",
            bg="white",
            fg="black",
            width=15,
            anchor="e"
        )
        z_label.place(relx=0.0, rely=0.0, x=label_x, y=345)

        self.z_coord_var = StringVar(value=str(z_pos))
        self.z_coord_entry = ttk.Entry(
            self.manual_control_panel,
            textvariable=self.z_coord_var,
            width=8
        )
        self.z_coord_entry.place(relx=0.0, rely=0.0, x=entry_x, y=345)

        self.Z_manual_control_button_panel = Frame(
            self.manual_control_panel,
            bg="white",
            width=80, 
            height=45 
        )
        self.Z_manual_control_button_panel.place(relx=0.5, x=0, y=380, anchor="n")
        self.Z_manual_control_button_panel.pack_propagate(False)

        z_controls = Frame(self.Z_manual_control_button_panel, bg="white")
        z_controls.pack(expand=True, fill="both")

        self.btn_up = ttk.Button(z_controls, text="▴", style="Arrow.TButton")
        self.btn_down = ttk.Button(z_controls, text="▾", style="Arrow.TButton")

        self.btn_up.bind("<ButtonPress-1>", self.on_press_up)
        self.btn_up.bind("<ButtonRelease-1>", self.on_release_up)
        self.btn_down.bind("<ButtonPress-1>", self.on_press_down)
        self.btn_down.bind("<ButtonRelease-1>", self.on_release_down)

        z_controls.rowconfigure(0, weight=1)
        z_controls.columnconfigure(0, weight=1)
        z_controls.columnconfigure(1, weight=1)

        self.btn_up.grid(row=0, column=0, sticky="nsew")
        self.btn_down.grid(row=0, column=1, sticky="nsew")

        return self.manual_control_panel

    def init_capture_image_panel(self):

        self.capture_panel = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=80
        )
        self.capture_panel.place(relx=1.0, rely=0.0, anchor="ne")

        self.capture_background = Frame(
            self.capture_panel,
            bg="white",
            width=200,
            height=78
        )
        self.capture_background.place(x=2, y=0)

        capture_title = Label(
            self.capture_panel,
            text="Image Capture",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        capture_title.place(relx=0.5, y=5, anchor="n")

        style = ttk.Style()
        style.configure("Save.TButton", background="white")
        style.configure("Save.TButton", relief="flat")

        self.capture_button = ttk.Button(
            self.capture_background,
            text="Capture & Save",
            style="Save.TButton",
            command=self.save_image
        )
        self.capture_button.place(relx=0.5, y=50, anchor="center")

        return self.capture_panel

    def init_adjust_exposure_panel(self):
        self.adjust_exposure_panel = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=100
        )
        self.adjust_exposure_panel.place(relx=1.0, rely=0.0, anchor="ne")

        self.adjust_exposure_background = Frame(
            self.adjust_exposure_panel,
            bg="white",
            width=200,
            height=98
        )
        self.adjust_exposure_background.place(x=2, y=0)

        adjust_exposure_title = Label(
            self.adjust_exposure_panel,
            text="Adjust Exposure",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        adjust_exposure_title.place(relx=0.5, y=5, anchor="n")

        style = ttk.Style()
        style.configure("Custom.Horizontal.TScale", background="white")

        self.exposure_var = DoubleVar(value=DEFAULT_EXPOSURE)
        self.adjust_exposure_slider = ttk.Scale(
            self.adjust_exposure_background,
            from_=30,
            to=120,
            orient="horizontal",
            variable=self.exposure_var,
            command=self.adjust_exposure,
            style="Custom.Horizontal.TScale"
        )
        self.adjust_exposure_slider.place(relx=0.5, y=50, anchor="center")

        self.exposure_value_label = Label(
            self.adjust_exposure_background,
            text=f"Exposure: {DEFAULT_EXPOSURE}",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 8)
        )
        self.exposure_value_label.place(relx=0.5, y=70, anchor="n")

        return self.adjust_exposure_panel

    def init_objective_control_panel(self):
        self.objective_control_panel = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=240
        )
        self.objective_control_panel.place(relx=1.0, rely=0.0, anchor="ne", y=442)

        self.objective_control_background = Frame(
            self.objective_control_panel,
            bg="white",
            width=200,
            height=238
        )
        self.objective_control_background.place(x=2, y=0)

        objective_control_title = Label(
            self.objective_control_panel,
            text="Objective Control",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        objective_control_title.place(relx=0.5, y=5, anchor="n")

        self.sharpness_var = tk.StringVar()
        self.sharpness_var.set("Objective: Unknown")

        self.sharpness_label = Label(
            self.objective_control_panel,
            textvariable=self.sharpness_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )

        self.sharpness_label.place(relx=0.5, y=40, anchor="n")

        style = ttk.Style()
        style.configure("Custom.Horizontal.TScale", background="white")

        self.objective_control_button_panel = Frame(
            self.objective_control_panel,
            bg="white",
            width=150,
            height=150
        )
        self.objective_control_button_panel.place(x=26, y=70)
        self.objective_control_button_panel.pack_propagate(False)

        controls = Frame(self.objective_control_button_panel, bg="white")
        controls.pack(expand=True, fill="both")

        style = ttk.Style()
        style.configure("Custom.TButton", font=("TkDefaultFont", 10), padding=5)
        style.configure("Custom.TButton", background="white", relief="flat")
        
        self.btn1 = ttk.Button(controls, text="1", style="Custom.TButton")
        self.btn2 = ttk.Button(controls, text="2", style="Custom.TButton")
        self.btn3 = ttk.Button(controls, text="3", style="Custom.TButton")
        self.btn4 = ttk.Button(controls, text="4", style="Custom.TButton")
        self.btn5 = ttk.Button(controls, text="5", style="Custom.TButton")

        self.objective_buttons = [self.btn1, self.btn2, self.btn3, self.btn4, self.btn5]

        for r in range(3):
            controls.rowconfigure(r, weight=1)
        for c in range(2):
            controls.columnconfigure(c, weight=1)

        self.btn1.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.btn2.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.btn3.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.btn4.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)
        self.btn5.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)

        self.btn1.bind("<ButtonPress-1>", lambda e: self.change_objective(1))
        self.btn2.bind("<ButtonPress-1>", lambda e: self.change_objective(2))
        self.btn3.bind("<ButtonPress-1>", lambda e: self.change_objective(3))
        self.btn4.bind("<ButtonPress-1>", lambda e: self.change_objective(4))
        self.btn5.bind("<ButtonPress-1>", lambda e: self.change_objective(5))

        self.change_objective(1)

        return self.objective_control_panel

    def init_focus_panel(self):
        self.focus_panel = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=120
        )
        self.focus_panel.place(relx=1.0, rely=0.0, anchor="ne", y=442)

        self.focus_background = Frame(
            self.focus_panel,
            bg="white",
            width=200,
            height=118
        )
        self.focus_background.place(x=2, y=0)

        focus_title = Label(
            self.focus_panel,
            text="Focus Control",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        focus_title.place(relx=0.5, y=5, anchor="n")

        self.sharpness_var = tk.StringVar()
        self.sharpness_var.set("Sharpness: Unknown")

        self.sharpness_label = Label(
            self.focus_panel,
            textvariable=self.sharpness_var,
            bg="white",
            fg="black",
            font="TkDefaultFont"
        )

        self.sharpness_label.place(relx=0.5, y=40, anchor="n")

        self.auto_focus_btn = ttk.Button(self.focus_panel, text="Auto Focus", style="Normal.TButton", command=self.auto_focus)
        self.auto_focus_btn.place(relx=0.5, y=75, anchor="n")

        '''
        self.focus_button_panel = Frame(
            self.manual_control_panel,
            bg="white",
            width=80, 
            height=45 
        )
        self.focus_button_panel.place(relx=0.5, x=0, y=380, anchor="n")
        self.focus_button_panel.pack_propagate(False)

        focus_controls = Frame(self.Z_manual_control_button_panel, bg="white")
        focus_controls.pack(expand=True, fill="both")

        self.btn_find_focus = ttk.Button(focus_controls, text="▴", style="Arrow.TButton")
        self.btn_down = ttk.Button(focus_controls, text="▾", style="Arrow.TButton")

        self.btn_up.bind("<ButtonPress-1>", self.on_press_up)
        self.btn_up.bind("<ButtonRelease-1>", self.on_release_up)
        self.btn_down.bind("<ButtonPress-1>", self.on_press_down)
        self.btn_down.bind("<ButtonRelease-1>", self.on_release_down)

        focus_controls.rowconfigure(0, weight=1)
        focus_controls.columnconfigure(0, weight=1)
        focus_controls.columnconfigure(1, weight=1)

        self.btn_up.grid(row=0, column=0, sticky="nsew")
        self.btn_down.grid(row=0, column=1, sticky="nsew")
        '''

        return self.focus_panel
      
    # ------------- Stage/Objective Control Functions -------------

    def set_origin(self):
        pc.set_origin()
        self.get_position()

    def go_to_position(self):
        pc.go_to_pos(int(self.x_coord_var.get()), int(self.y_coord_var.get()))
        self.get_position()

    def get_position(self):
        global x_pos, y_pos
        pc.get_curr_pos()
        x_pos = pc.x
        y_pos = pc.y
        z_pos = pc.z
        self.x_coord_var.set(str(x_pos))
        self.y_coord_var.set(str(y_pos))
        self.z_coord_var.set(str(z_pos))

    def start_hold_forward(self):
        self.is_hold = True
        pc.start_forward_y_motor()

    def on_press_forward(self, event):
        self.is_hold = False
        self.hold_job = self.root.after(200, self.start_hold_forward)

    def on_release_forward(self, event):

        if self.hold_job is not None:
            self.root.after_cancel(self.hold_job)

        if self.is_hold:
            pc.stop_y_motor()   # stop continuous motion
        else:
            global y_pos
            y_pos -= int(self.step_entry.get())
            pc.go_to_pos(x_pos, y_pos)

        self.get_position()

    def start_hold_backward(self):
        self.is_hold = True
        pc.start_backward_y_motor()

    def on_press_backward(self, event):
        self.is_hold = False
        self.hold_job = self.root.after(200, self.start_hold_backward)

    def on_release_backward(self, event):
    
        if self.hold_job is not None:
            self.root.after_cancel(self.hold_job)
    
        if self.is_hold:
            pc.stop_y_motor()
        else:
            global y_pos
            y_pos += int(self.step_entry.get())
            pc.go_to_pos(x_pos, y_pos)

        self.get_position()

    def start_hold_left(self):
        self.is_hold = True
        pc.start_forward_x_motor()

    def on_press_left(self, event):
        self.is_hold = False
        self.hold_job = self.root.after(200, self.start_hold_left)

    def on_release_left(self, event):
    
        if self.hold_job is not None:
            self.root.after_cancel(self.hold_job)
    
        if self.is_hold:
            pc.stop_x_motor()
        else:
            global x_pos
            x_pos += int(self.step_entry.get())
            pc.go_to_pos(x_pos, y_pos)

        self.get_position()

    def start_hold_right(self):
        self.is_hold = True
        pc.start_backward_x_motor()

    def on_press_right(self, event):
        self.is_hold = False
        self.hold_job = self.root.after(200, self.start_hold_right)

    def on_release_right(self, event):
    
        if self.hold_job is not None:
            self.root.after_cancel(self.hold_job)
    
        if self.is_hold:
            pc.stop_x_motor()
        else:
            global x_pos
            x_pos -= int(self.step_entry.get())
            pc.go_to_pos(x_pos, y_pos)

        self.get_position()

    def start_hold_up(self):
        self.is_hold = True
        pc.start_backward_z_motor()

    def on_press_up(self, event):
        self.is_hold = False
        self.hold_job = self.root.after(200, self.start_hold_up)

    def on_release_up(self, event):
    
        if self.hold_job is not None:
            self.root.after_cancel(self.hold_job)
    
        if self.is_hold:
            pc.stop_z_motor()
        else:
            global z_pos
            z_pos -= int(self.step_entry.get())
            pc.go_to_z_pos(z_pos)

        self.get_position()

    def start_hold_down(self):
        self.is_hold = True
        pc.start_forward_z_motor()

    def on_press_down(self, event):
        self.is_hold = False
        self.hold_job = self.root.after(200, self.start_hold_down)

    def on_release_down(self, event):
    
        if self.hold_job is not None:
            self.root.after_cancel(self.hold_job)
    
        if self.is_hold:
            pc.stop_z_motor()
        else:
            global z_pos
            z_pos += int(self.step_entry.get())
            pc.go_to_z_pos(z_pos)

        self.get_position()

    def on_hold_speed_change(self, *args):
        try:
            speed = int(self.hold_speed_var.get())
            pc.set_velocity(speed)
        except ValueError:
            pass

    def change_objective(self, position):
        tc.turn_to_position(position)

        self.sharpness_var.set(f"Objective: {position}")

    def find_sharpness(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (3,3), 0)

        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        self.sharpness_var.set(f"Sharpness: {sharpness:.3f}")

        return sharpness

    def find_best_focus(self, z_start, z_end, steps):

        best_focus = -1
        best_z = z_start

        z_positions = [
            z_start + i*(z_end-z_start)/steps
            for i in range(steps+1)
        ]

        if (abs(pc.z - z_positions[0]) < abs(pc.z - z_positions[-1])):
            z_positions.reverse()

        for z in z_positions:

            pc.go_to_z_pos(z)
            self.get_position()

            image = self.capture_frame(num_images=1)
            score = self.find_sharpness(image)

            #print(f"Z: {z}, Sharpness: {score}")

            if score > best_focus:
                best_focus = score
                best_z = z

        pc.go_to_z_pos(best_z)

        return best_z

    def auto_focus(self, start_range = 1000):
        _range = start_range
        best_z = pc.z
        while _range > 100:
            best_z = self.find_best_focus(best_z-_range, best_z+_range, 10)
            _range = int(_range / 2)
            image = self.capture_frame()
            sharpness = self.find_sharpness(image)
            #print(f"Best Z: {best_z}, Sharpness: {sharpness}, Range: {_range}")
            #print("-----------------------------------")

    def stop_all_motors(self):
        pc.stop_x_motor()
        pc.stop_y_motor()
        pc.stop_z_motor()

    # ------------- Scanning Functions -------------

    def run_chip_mapping_scan(self, zoom=25):

        print("Chip mapping scan running...")

        start_time = time.time()
        self.true_map = np.zeros((3000, 3000, 3), dtype=np.uint8)
        self.filter_map = np.zeros((3000, 3000), dtype=np.int8)
        self.scan_running = True

        global x_pos, y_pos

        num_steps_x = 6
        num_steps_y = 6

        center_x = x_pos
        center_y = y_pos

        #coords, total_frames = self.generate_spiral_coords(max(num_steps_x, num_steps_y))
        coords, total_frames = self.generate_rect_coords(7, 3)

        i = 0
        for offset_x, offset_y in coords:
            target_x = center_x + offset_x * X_SIZE_2
            target_y = center_y - offset_y * Y_SIZE_2 

            pc.go_to_pos(target_x, target_y)
            x_pos, y_pos = target_x, target_y

            print(f"Move Time: {pc.wait_until_not_busy()}")

            img = self.capture_frame()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            binary = chip_edge_classifier.chip_filter(img, display=False)
            img_binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

            if self.view_mode == "Camera View":
                self.display_live_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif self.view_mode == "Filter":
                self.display_live_image(cv2.cvtColor(chip_edge_classifier.chip_filter(img), cv2.COLOR_GRAY2RGB))

            map_x = int(self.filter_map.shape[1] / 2 + (offset_x - 0.5) * img_binary_rgb.shape[1] / zoom)
            map_y = int(self.filter_map.shape[0] / 2 + (offset_y - 0.5) * img_binary_rgb.shape[0] / zoom)

            h, w = img_binary_rgb.shape[:2]
            img_small = img_rgb[::zoom, ::zoom]
            img_binary_small = img_binary_rgb[::zoom, ::zoom, 0]

            x_start = max(0, map_x)
            y_start = max(0, map_y)
            x_end = min(self.filter_map.shape[1], x_start + img_binary_small.shape[1])
            y_end = min(self.filter_map.shape[0], y_start + img_binary_small.shape[0])

            self.true_map[y_start:y_end, x_start:x_end] = img_small[:y_end - y_start, :x_end - x_start]
            self.filter_map[y_start:y_end, x_start:x_end] = img_binary_small[:y_end - y_start, :x_end - x_start]

            elapsed = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            progress_percent = f"{(i+1)}/{total_frames} ({(i+1)*100//total_frames}%)"
            i = i + 1
    
            self.update_scan_status(progress=progress_percent, elapsed_time=elapsed_str)

        self.scan_running = False
        pc.go_to_pos(center_x, center_y)
        print("Scan finished!")

    def generate_rect_coords(self, x, y):

        rect_coords = []
        total_frames = x * y

        for i in range(x):

            if i % 2 == 0:
                y_range = range(y)
            else:
                y_range = range(y - 1, -1, -1)

            for j in y_range:
                rect_coords.append((i - x // 2, j - y // 2))

        return rect_coords, total_frames

    def generate_spiral_coords(self, length):

        spiral_coords = []
        total_frames = length ** 2

        dx, dy = 0, 0
        step = 1
        direction = 0

        while len(spiral_coords) < total_frames:
            for _ in range(2):
                for _ in range(step):
                    if len(spiral_coords) >= total_frames:
                        break
                    spiral_coords.append((dx, dy))
                    if direction == 0:
                        dx += 1
                    elif direction == 1:
                        dy += 1
                    elif direction == 2:
                        dx -= 1
                    else:
                        dy -= 1
                direction = (direction + 1) % 4
            step += 1

        return spiral_coords, total_frames

    def update_scan_status(self, progress=None, elapsed_time=None, magnification=None):
        if magnification is not None:
            self.info_magnification_label.config(text=f"Magnification: {magnification}x")

        if progress is not None:
            self.info_progress_label.config(text=f"Progress: {progress}")

        if elapsed_time is not None:
            self.info_time_label.config(text=f"Time Elapsed: {elapsed_time}")

        self.display_map()
        self.info_panel.update()

    # ------------- Display Functions -------------

    def set_view(self, mode, filter_status):
        self.view_mode = mode

        self.filter_on = filter_status

        if mode == "Map":
            self.display_map()
            self.map_canvas.pack(fill=BOTH, expand=True) # Show map canvas
            self.img_label.pack_forget() # Hide image label
            pass
        else:
            self.img_label.pack(fill=BOTH, expand=True)
            self.map_canvas.pack_forget() # Hide map canvas

    def draw_map(self):
        map_img = Image.fromarray((self.filter_map * 255).astype(np.uint8), mode="L")

        canvas_w = self.map_canvas.winfo_width()
        canvas_h = self.map_canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            self.root.after(100, self.draw_map)
            return

        scale = min(canvas_w / map_img.width, canvas_h / map_img.height, 1.0)
        new_w = int(map_img.width * scale)
        new_h = int(map_img.height * scale)
        map_img_resized = map_img.resize((new_w, new_h), Image.Resampling.NEAREST)

        self.map_tk = ImageTk.PhotoImage(map_img_resized)

        x_center = canvas_w // 2
        y_center = canvas_h // 2

        self.map_canvas.delete("all")
        self.map_canvas.create_image(
            x_center, y_center,
            anchor="center",
            image=self.map_tk
        )

    def display_map(self):
        if self.filter_on:
            self.map_image = Image.fromarray(self.filter_map.astype(np.uint8))
        else:
            self.map_image = Image.fromarray(self.true_map.astype(np.uint8))

        canvas_width = self.map_canvas.winfo_width()
        canvas_height = self.map_canvas.winfo_height()

        if canvas_width == 1 or canvas_height == 1:
            self.map_canvas.bind("<Configure>", lambda e: self.display_map())
            return

        img_ratio = self.map_image.width / self.map_image.height
        canvas_ratio = canvas_width / canvas_height

        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)

        img_resized = self.map_image.resize((new_width, new_height), Image.NEAREST)
        self.tk_map_image = ImageTk.PhotoImage(img_resized)

        self.map_canvas.delete("all")

        x_center = canvas_width // 2
        y_center = canvas_height // 2
        self.map_canvas.create_image(x_center, y_center, image=self.tk_map_image, anchor="center")

    def display_live_image(self, img_rgb):

        h, w = img_rgb.shape[:2]

        cx = w // 2
        cy = h // 2

        x1 = cx - CENTER_CROP_WIDTH // 2
        y1 = cy - CENTER_CROP_HEIGHT // 2
        x2 = cx + CENTER_CROP_WIDTH // 2
        y2 = cy + CENTER_CROP_HEIGHT // 2

        img_rgb = img_rgb.copy()
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 5)

        img_pil = Image.fromarray(img_rgb)

        lbl_w = self.img_label.winfo_width() or self.width
        lbl_h = self.img_label.winfo_height() or self.height


        if lbl_w < 10 or lbl_h < 10:
            return

        img_pil_copy = img_pil.copy()
        img_pil_copy.thumbnail((lbl_w, lbl_h), Image.Resampling.LANCZOS)

        display_img = Image.new("RGB", (lbl_w, lbl_h), "#f0f0f0")
        x_offset = (lbl_w - img_pil_copy.width) // 2
        y_offset = (lbl_h - img_pil_copy.height) // 2
        display_img.paste(img_pil_copy, (x_offset, y_offset))

        img_tk = ImageTk.PhotoImage(display_img)
        self.img_label.configure(image=img_tk)
        self.img_label.image = img_tk

    # ------------- Setting and Saving Functions -------------

    def adjust_exposure(self, exposure):
        self.hcam.put_AutoExpoTarget(int(float(exposure)))
        self.exposure_value_label.config(text=f"Exposure: {int(float(self.hcam.get_AutoExpoTarget()))}")

    # ------------- Util Functions -------------

    def update_panels(self):
        y_position = -2

        for panel in self.panels:
            frame = panel["frame"]
            frame.place_forget()

            if panel["var"].get():
                frame.place(
                    relx=1.0,
                    rely=0.0,
                    anchor="ne",
                    y=y_position
                )

                frame.update_idletasks()

                y_position += frame.winfo_height()

        # ------------- Camera Handling -------------

    @staticmethod
    def cameraCallback(nEvent, ctx):
        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            ctx.on_image()

    def on_image(self):
        try:
            self.hcam.PullImageV2(self.buf, 24, None)
            self.frame_id += 1

            row_bytes = ((self.width * 24 + 31) // 32 * 4)
            img = np.frombuffer(self.buf, dtype=np.uint8).reshape(self.height, row_bytes)
            img = img[:, :self.width * 3].reshape(self.height, self.width, 3)
            img = cv2.flip(img, -1)
            self.current_frame = img.copy()
    
            if self.view_mode == "Map": return
            
            if self.view_mode == "Camera View" and not self.scan_running:
                if self.filter_on:
                    self.display_live_image(cv2.cvtColor(chip_edge_classifier.chip_filter(img), cv2.COLOR_GRAY2RGB))
                else:
                    self.display_live_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
            self.find_sharpness(img)

        except amcam.HRESULTException as ex:
            print("Camera error:", ex)

    def run_camera(self):
        cams = amcam.Amcam.EnumV2()
        if not cams:
            print("No camera found")
            return

        self.hcam = amcam.Amcam.Open(cams[0].id)
        self.hcam.put_AutoExpoEnable(True)
        self.hcam.put_AutoExpoTarget(DEFAULT_EXPOSURE)

        self.width, self.height = self.hcam.get_Size()
        bufsize = ((self.width * 24 + 31) // 32 * 4) * self.height
        self.buf = bytes(bufsize)

        self.frame_id = 0

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        scale = min(screen_width / self.width, screen_height / self.height, 1.0)
        win_width = int(self.width * scale)
        win_height = int(self.height * scale)
        self.root.geometry(f"{win_width}x{win_height}")

        self.root.update_idletasks()
        self.root.update()

        self.hcam.StartPullModeWithCallback(self.cameraCallback, self)

    def save_image(self):
        if not hasattr(self, "current_frame") or self.current_frame is None:
            print("No frame available to save.")
            return

        save_dir = "Saved Images"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)

        cv2.imwrite(filepath, self.current_frame)

        print(f"Image saved to {filepath}")

    def wait_until_new_frame(self):
        old_frame = self.frame_id
        start = time.time()
        while self.frame_id == old_frame:
            self.root.update()
            time.sleep(0.005)
            if time.time() - start > 1:
                print("Frame timeout")
                break

    def capture_frame(self, num_images=2):
        self.wait_until_new_frame()

        sum_frame = np.zeros_like(self.current_frame, dtype=np.float32)

        for _ in range(num_images):
            self.wait_until_new_frame()
            sum_frame += self.current_frame.astype(np.float32)

        avg_frame = (sum_frame / num_images).astype(self.current_frame.dtype)
        return avg_frame

    def on_close(self):
        self.hcam = None
        self.buf = None
        pc.disconnect()
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    app.run_camera()
    root.mainloop()