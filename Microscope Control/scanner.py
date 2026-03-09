import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from tkinter import *
from tkinter import ttk
import tkinter.font as tkFont
from PIL import Image, ImageTk

import amcam
from prior import prior
import chip_edge_classifier

DLL_PATH = os.getcwd() + r"\PriorSDK1.9.2\x64\PriorScientificSDK.dll"
COM_PORT = sys.argv[1]
DEFAULT_EXPOSURE = 60

X_SIZE_2 = 7410 # 6 Frames
Y_SIZE_2 = 4635 # 6 Frames

Y_SIZE_4 = 3660 
X_SIZE_4 = 2435

MAGNIFICATION = 2

try:
    pr = prior(COM_PORT, DLL_PATH)
    pr.get_curr_pos()
    x_pos = pr.x
    y_pos = pr.y
except Exception as e:
    print("Failed to connect to Prior:", e)
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
        menu_bar.add_command(label="Run Scan", command=self.run_scan)

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
            height=354
        )
        self.manual_control_panel.place(relx=1.0, rely=0.0, anchor="ne")

        self.manual_control_background = Frame(
            self.manual_control_panel,
            bg="white",
            width=200,
            height=352
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

        self.manual_control_button_panel = Frame(
            self.manual_control_panel,
            bg="white",
            width=120, 
            height=90 
        )
        self.manual_control_button_panel.place(relx=0.5, x=0, y=240, anchor="n")
        self.manual_control_button_panel.pack_propagate(False)

        controls = Frame(self.manual_control_button_panel, bg="white")
        controls.pack(expand=True, fill="both")

        style = ttk.Style()
        style.configure("Arrow.TButton", font=("TkDefaultFont", 15), padding=5)
        style.configure("Arrow.TButton", background="white")
        style.configure("Arrow.TButton", relief="flat")

        self.btn_up = ttk.Button(controls, text="▴", style="Arrow.TButton")
        self.btn_down = ttk.Button(controls, text="▾", style="Arrow.TButton")
        self.btn_left = ttk.Button(controls, text="◂", style="Arrow.TButton")
        self.btn_right = ttk.Button(controls, text="▸", style="Arrow.TButton")

        self.btn_up.bind("<ButtonPress-1>", self.on_press_up)
        self.btn_up.bind("<ButtonRelease-1>", self.on_release_up)
        self.btn_down.bind("<ButtonPress-1>", self.on_press_down)
        self.btn_down.bind("<ButtonRelease-1>", self.on_release_down)
        self.btn_left.bind("<ButtonPress-1>", self.on_press_left)
        self.btn_left.bind("<ButtonRelease-1>", self.on_release_left)
        self.btn_right.bind("<ButtonPress-1>", self.on_press_right)
        self.btn_right.bind("<ButtonRelease-1>", self.on_release_right)

        for r in [0, 1]:
            controls.rowconfigure(r, weight=1)
        for c in [0, 1, 2]:
            controls.columnconfigure(c, weight=1)

        self.btn_up.grid(row=0, column=1, sticky="nsew")
        self.btn_left.grid(row=1, column=0, sticky="nsew")
        self.btn_right.grid(row=1, column=2, sticky="nsew")
        self.btn_down.grid(row=1, column=1, sticky="nsew")

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

    # ------------- Stage Control Functions -------------

    def set_origin(self):
        pr.set_origin()
        self.get_position()

    def go_to_position(self):
        pr.go_to_pos(int(self.x_coord_var.get()), int(self.y_coord_var.get()))
        self.get_position()

    def get_position(self):
        global x_pos, y_pos
        pr.get_curr_pos()
        x_pos = pr.x
        y_pos = pr.y
        self.x_coord_var.set(str(x_pos))
        self.y_coord_var.set(str(y_pos))

    def start_hold_up(self):
        self.is_hold = True
        pr.start_forward_y_motor()

    def on_press_up(self, event):
        self.is_hold = False
        self.hold_job = self.root.after(200, self.start_hold_up)

    def on_release_up(self, event):

        if self.hold_job is not None:
            self.root.after_cancel(self.hold_job)

        if self.is_hold:
            pr.stop_y_motor()   # stop continuous motion
        else:
            global y_pos
            y_pos -= int(self.step_entry.get())
            pr.go_to_pos(x_pos, y_pos)

        self.get_position()

    def start_hold_down(self):
        self.is_hold = True
        pr.start_backward_y_motor()

    def on_press_down(self, event):
        self.is_hold = False
        self.hold_job = self.root.after(200, self.start_hold_down)

    def on_release_down(self, event):
    
        if self.hold_job is not None:
            self.root.after_cancel(self.hold_job)
    
        if self.is_hold:
            pr.stop_y_motor()
        else:
            global y_pos
            y_pos += int(self.step_entry.get())
            pr.go_to_pos(x_pos, y_pos)

        self.get_position()

    def start_hold_left(self):
        self.is_hold = True
        pr.start_backward_x_motor()

    def on_press_left(self, event):
        self.is_hold = False
        self.hold_job = self.root.after(200, self.start_hold_left)

    def on_release_left(self, event):
    
        if self.hold_job is not None:
            self.root.after_cancel(self.hold_job)
    
        if self.is_hold:
            pr.stop_x_motor()
        else:
            global x_pos
            x_pos -= int(self.step_entry.get())
            pr.go_to_pos(x_pos, y_pos)

        self.get_position()

    def start_hold_right(self):
        self.is_hold = True
        pr.start_forward_x_motor()

    def on_press_right(self, event):
        self.is_hold = False
        self.hold_job = self.root.after(200, self.start_hold_right)

    def on_release_right(self, event):
    
        if self.hold_job is not None:
            self.root.after_cancel(self.hold_job)
    
        if self.is_hold:
            pr.stop_x_motor()
        else:
            global x_pos
            x_pos += int(self.step_entry.get())
            pr.go_to_pos(x_pos, y_pos)

        self.get_position()

    def on_hold_speed_change(self, *args):
        try:
            speed = int(self.hold_speed_var.get())
            pr.set_velocity(speed)
        except ValueError:
            pass

    # ------------- Scanning Functions -------------

    def run_scan(self, zoom=25):

        print("Scan running...")

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
        coords, total_frames = self.generate_rect_coords(19, 9)

        i = 0
        for offset_x, offset_y in coords:
            target_x = center_x + offset_x * X_SIZE_2
            target_y = center_y - offset_y * Y_SIZE_2 

            pr.go_to_pos(target_x, target_y)
            x_pos, y_pos = target_x, target_y

            print(f"Move Time: {pr.wait_until_not_busy()}")

            img = self.capture_frame()
            img = np.flipud(img)
            img = np.fliplr(img)
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
        pr.go_to_pos(center_x, center_y)
        print("Scan finished!")

    def generate_rect_coords(self, x, y):
        
        rect_coords = []
        total_frames = x * y

        for i in range(x):
            for j in range(y):
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

            # hide everything first
            frame.place_forget()

            if panel["var"].get():   # if checked

                frame.place(
                    relx=1.0,
                    rely=0.0,
                    anchor="ne",
                    y=y_position
                )

                y_position += frame.winfo_height()

        # ------------- Camera Handling -------------

    @staticmethod
    def cameraCallback(nEvent, ctx):
        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            ctx.on_image()

    def on_image(self):
        try:
            self.hcam.PullImageV2(self.buf, 24, None)
    
            row_bytes = ((self.width * 24 + 31) // 32 * 4)
            img = np.frombuffer(self.buf, dtype=np.uint8).reshape(self.height, row_bytes)
            img = img[:, :self.width * 3].reshape(self.height, self.width, 3)
            self.current_frame = img.copy()
    
            if self.view_mode == "Map": return
            
            if self.view_mode == "Camera View" and not self.scan_running:
                if self.filter_on:
                    self.display_live_image(cv2.cvtColor(chip_edge_classifier.chip_filter(img), cv2.COLOR_GRAY2RGB))
                else:
                    self.display_live_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
        except amcam.HRESULTException as ex:
            print(f"Camera error: 0x{ex.hr:x}")

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

    def capture_frame(self):
        return self.current_frame.copy()

    def on_close(self):
        self.hcam = None
        self.buf = None
        pr.disconnect()
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    app.run_camera()
    root.mainloop()