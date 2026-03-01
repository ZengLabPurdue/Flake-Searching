import os
import sys
import cv2
import numpy as np
from tkinter import *
from tkinter import ttk
import tkinter.font as tkFont
from PIL import Image, ImageTk

import amcam
from prior import prior
from datetime import datetime
import chip_edge_classifier

DLL_PATH = os.getcwd() + r"\PriorSDK1.9.2\x64\PriorScientificSDK.dll"
COM_PORT = sys.argv[1]
DEFAULT_EXPOSURE = 60

Y_SIZE = 3660
X_SIZE = 2435

MAGNIFICATION = 4

try:
    pr = prior(COM_PORT, DLL_PATH)
    pr.get_curr_pos()
    x_pos = pr.x
    y_pos = pr.y
    print("Connected to Prior stage")
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

        self.map = np.zeros((3000, 3000), dtype=np.int8)

        self.img_label = Label(self.main_frame, bg="#f0f0f0")
        self.img_label.pack(fill=BOTH, expand=True)

        self.view_mode = "Camera View"
        self.set_view(self.view_mode)

        self.init_menu_bar()
        self.init_info_panel()
        self.init_settings_panel()

        self.hcam = None
        self.buf = None
        self.prevImg = None
        self.width = 0
        self.height = 0

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_menu_bar(self):
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Quit", command=self.on_close)
        menubar.add_cascade(label="File", menu=filemenu)

        viewmenu = Menu(menubar, tearoff=0)
        viewmenu.add_command(label="Map", command=lambda: self.set_view("Map"))
        livemenu = Menu(viewmenu, tearoff=0)
        livemenu.add_command(label="Camera View", command=lambda: self.set_view("Camera View"))
        livemenu.add_command(label="Filter", command=lambda: self.set_view("Filter"))
        viewmenu.add_cascade(label="Live", menu=livemenu)
        menubar.add_cascade(label="View", menu=viewmenu)

        menubar.add_command(label="Settings", command=lambda: self.open_settings())
        menubar.add_command(label="Run Scan", command=lambda: self.run_scan())

        root.config(menu=menubar)

    def init_info_panel(self):

        self.info_panel_border = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=114
        )
        self.info_panel_border.place(relx=1.0, rely=0.0, anchor="ne", y=-2)

        self.info_panel = Frame(
            self.info_panel_border,
            bg="white",
            width=200,
            height=112
        )
        self.info_panel.place(x=2, y=0)

        title_label = Label(
            self.info_panel_border,
            text="Info",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title_label.place(relx=0.5, y=10, anchor="n")

        for text, y in [
            (f"Magnification: {MAGNIFICATION}x", 35),
            ("Progress: Not Started", 55),
            ("Time Elapsed: Not Started", 75),
        ]:
            Label(
                self.info_panel_border,
                text=text,
                bg="white",
                fg="black"
            ).place(relx=0.5, y=y, anchor="n")

    def init_settings_panel(self):
        self.settings_border = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=100
        )
        self.settings_border.place(relx=1.0, rely=0.0, anchor="ne", y=112)

        self.settings_panel = Frame(
            self.settings_border,
            bg="white",
            width=200,
            height=98
        )
        self.settings_panel.place(x=2, y=0)

        adjust_exposure_title = Label(
            self.settings_border,
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
            self.settings_panel,
            from_=30,
            to=120,
            orient="horizontal",
            variable=self.exposure_var,
            command=self.adjust_exposure,
            style="Custom.Horizontal.TScale"
        )
        self.adjust_exposure_slider.place(relx=0.5, y=50, anchor="center")

        self.exposure_value_label = Label(
            self.settings_panel,
            text=f"Exposure: {DEFAULT_EXPOSURE}",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 8)
        )
        self.exposure_value_label.place(relx=0.5, y=70, anchor="n")

    def adjust_exposure(self, exposure):
        self.hcam.put_AutoExpoTarget(int(float(exposure)))
        self.exposure_value_label.config(text=f"Exposure: {int(float(self.hcam.get_AutoExpoTarget()))}")

    def open_settings(self):
        pass

    # Complete chip scan method
    def run_scan(self):
        pr.get_curr_pos()
        self.start_location = (pr.x, pr.y)
        pass

    def set_view(self, mode):
        self.view_mode = mode
    
        if mode == "Map":
            self.map_canvas.pack(fill=BOTH, expand=True) # Show map canvas
            self.img_label.pack_forget() # Hide image label
            pass
        else:
            self.img_label.pack(fill=BOTH, expand=True)
            self.map_canvas.pack_forget() # Hide map canvas
            if mode == "Camera View":
                pass 
            elif mode == "Filter":
                pass

    def draw_map(self):
        map_img = Image.fromarray((self.map * 255).astype(np.uint8), mode="L")

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

    @staticmethod
    def cameraCallback(nEvent, ctx):
        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            ctx.on_image()

    def on_image(self):
        try:
            if self.view_mode == "Map": return

            self.hcam.PullImageV2(self.buf, 24, None)
    
            row_bytes = ((self.width * 24 + 31) // 32 * 4)
            img = np.frombuffer(self.buf, dtype=np.uint8).reshape(self.height, row_bytes)
            img = img[:, :self.width * 3].reshape(self.height, self.width, 3)
            self.current_frame = img.copy()
    
            if self.view_mode == "Camera View":
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif self.view_mode == "Filter":
                img_rgb = chip_edge_classifier.chip_filter(img)
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

    def capture_frame(self):
        self.hcam.Snap(1)
        self.hcam.PullImageV2(self.buf, 24, None)
    
        row_bytes = ((self.width * 24 + 31) // 32 * 4)
        img = np.frombuffer(self.buf, np.uint8).reshape(self.height, row_bytes)
        img = img[:, :self.width * 3].reshape(self.height, self.width, 3)
    
        return img.copy()

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