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
from PIL import Image
import chip_edge_classifier

DLL_PATH = os.getcwd() + r"\PriorSDK1.9.2\x64\PriorScientificSDK.dll"
COM_PORT = sys.argv[1]
DEFAULT_EXPOSURE = 60

Y_SIZE = 3660
X_SIZE = 2435

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
        self.root.title("Stage Control")

        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="Quit", command=self.on_close)
        menubar.add_cascade(label="File", menu=filemenu)
        root.config(menu=menubar)

        self.main_frame = Frame(root, bg="black")
        self.main_frame.pack(fill=BOTH, expand=True)

        self.img_label = Label(self.main_frame, bg="black")
        self.img_label.pack(fill=BOTH, expand=True)

        self.step_size = 5

        self.init_manual_control_button_panel()
        self.init_capture_image_panel()
        self.init_adjust_exposure_panel()

        self.hcam = None
        self.buf = None
        self.prevImg = None
        self.width = 0
        self.height = 0

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_manual_control_button_panel(self):

        self.manual_control_border = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=184
        )
        self.manual_control_border.place(relx=1.0, rely=0.0, anchor="ne", y=-2)

        self.manual_control_panel = Frame(
            self.manual_control_border,
            bg="white",
            width=200,
            height=182
        )
        self.manual_control_panel.place(x=2, y=0)

        title_label = Label(
            self.manual_control_border,
            text="Manual Control",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title_label.place(relx=0.5, y=10, anchor="n")

        step_label = Label(
            self.manual_control_border,
            text="Step Size (µm):",
            bg="white",
            fg="black"
        )
        step_label.place(relx=0.0, rely=0.0, x=80, y=45, anchor="n")

        self.step_size_var = StringVar(value=str(self.step_size))  # bind to current step size
        self.step_entry = ttk.Entry(
            self.manual_control_border,
            textvariable=self.step_size_var,
            width=5
        )
        self.step_entry.place(relx=0.0, rely=0.0, x=150, y=45, anchor="n")

        self.manual_control_button_panel = Frame(
            self.manual_control_border,
            bg="white",
            width=120, 
            height=90 
        )
        self.manual_control_button_panel.place(relx=0.5, x=0, y=80, anchor="n")
        self.manual_control_button_panel.pack_propagate(False)

        style = ttk.Style()
        style.configure("Arrow.TButton", font=("TkDefaultFont", 15), padding=5)
        style.configure("Arrow.TButton", background="white")
        style.configure("Arrow.TButton", relief="flat")

        controls = Frame(self.manual_control_button_panel, bg="white")
        controls.pack(expand=True, fill="both")

        self.btn_up = ttk.Button(controls, text="▴", style="Arrow.TButton", command=self.move_up)
        self.btn_down = ttk.Button(controls, text="▾", style="Arrow.TButton", command=self.move_down)
        self.btn_left = ttk.Button(controls, text="◂", style="Arrow.TButton", command=self.move_left)
        self.btn_right = ttk.Button(controls, text="▸", style="Arrow.TButton", command=self.move_right)

        for r in [0, 1]:
            controls.rowconfigure(r, weight=1)
        for c in [0, 1, 2]:
            controls.columnconfigure(c, weight=1)

        self.btn_up.grid(row=0, column=1, sticky="nsew")
        self.btn_left.grid(row=1, column=0, sticky="nsew")
        self.btn_right.grid(row=1, column=2, sticky="nsew")
        self.btn_down.grid(row=1, column=1, sticky="nsew")

    def init_capture_image_panel(self):

        self.capture_panel_border = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=80
        )
        self.capture_panel_border.place(relx=1.0, rely=0.0, anchor="ne", y=182)

        self.capture_panel = Frame(
            self.capture_panel_border,
            bg="white",
            width=200,
            height=78
        )
        self.capture_panel.place(x=2, y=0)

        capture_title = Label(
            self.capture_panel_border,
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
            self.capture_panel,
            text="Capture & Save",
            style="Save.TButton",
            command=self.capture_image
        )
        self.capture_button.place(relx=0.5, y=50, anchor="center")

    def init_adjust_exposure_panel(self):
        self.adjust_exposure_border = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=100
        )
        self.adjust_exposure_border.place(relx=1.0, rely=0.0, anchor="ne", y=262)

        self.adjust_exposure_panel = Frame(
            self.adjust_exposure_border,
            bg="white",
            width=200,
            height=98
        )
        self.adjust_exposure_panel.place(x=2, y=0)

        adjust_exposure_title = Label(
            self.adjust_exposure_border,
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
            self.adjust_exposure_panel,
            from_=30,
            to=120,
            orient="horizontal",
            variable=self.exposure_var,
            command=self.adjust_exposure,
            style="Custom.Horizontal.TScale"
        )
        self.adjust_exposure_slider.place(relx=0.5, y=50, anchor="center")

        self.exposure_value_label = Label(
            self.adjust_exposure_panel,
            text=f"Exposure: {DEFAULT_EXPOSURE}",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 8)
        )
        self.exposure_value_label.place(relx=0.5, y=70, anchor="n")

    def move_up(self):
        global y_pos
        y_pos -= int(self.step_entry.get())
        pr.go_to_pos(x_pos, y_pos)

    def move_down(self):
        global y_pos
        y_pos += int(self.step_entry.get())
        pr.go_to_pos(x_pos, y_pos)

    def move_left(self):
        global x_pos
        x_pos -= int(self.step_entry.get())
        pr.go_to_pos(x_pos, y_pos)

    def move_right(self):
        global x_pos
        x_pos += int(self.step_entry.get())
        pr.go_to_pos(x_pos, y_pos)

    def adjust_exposure(self, exposure):
        self.hcam.put_AutoExpoTarget(int(float(exposure)))
        self.exposure_value_label.config(text=f"Exposure: {int(float(self.hcam.get_AutoExpoTarget()))}")

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
    
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img_rgb = chip_edge_classifier.chip_filter(img)
            img_pil = Image.fromarray(img_rgb)
    
            lbl_w = self.img_label.winfo_width() or self.width
            lbl_h = self.img_label.winfo_height() or self.height
    
            img_pil_copy = img_pil.copy()
            img_pil_copy.thumbnail((lbl_w, lbl_h), Image.Resampling.LANCZOS)
    
            display_img = Image.new("RGB", (lbl_w, lbl_h), (0, 0, 0))
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

    def capture_image(self):
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