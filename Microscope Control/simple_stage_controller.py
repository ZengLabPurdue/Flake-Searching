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


DLL_PATH = os.getcwd() + r"\Motor Control\PriorSDK1.9.2\x64\PriorScientificSDK.dll"
COM_PORT = sys.argv[1]

try:
    pr = prior(COM_PORT, DLL_PATH)
    pr.get_curr_pos()
    x_pos = pr.x
    y_pos = pr.y
    print("Connected to Prior stage")
except Exception as e:
    print("Failed to connect to Prior:", e)
    sys.exit(1)

def runRegistration(currImgArray, prevImgArray):
    if prevImgArray is None:
        return 0, 0
    gray1 = np.float32(cv2.cvtColor(currImgArray, cv2.COLOR_BGR2GRAY))
    gray2 = np.float32(cv2.cvtColor(prevImgArray, cv2.COLOR_BGR2GRAY))
    (dx, dy), _ = cv2.phaseCorrelate(gray1, gray2)
    height, width = currImgArray.shape[:2]
    num_pixels_x = dx * width
    num_pixels_y = dy * height
    print(f"Pixels mapped: X={num_pixels_x}, Y={num_pixels_y}")
    return num_pixels_x, num_pixels_y

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera + Stage Control")

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
            font=("TkDefaultFont", 15, "bold")
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
        self.step_entry.bind("<Return>", self.on_enter_step_size)
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
        style.configure("Arrow.TButton", font=("Ariel", 12, "bold"), padding=5)
        style.configure("Arrow.TButton", background="white")
        style.configure("Arrow.TButton", relief="flat")

        controls = Frame(self.manual_control_button_panel, bg="white")
        controls.pack(expand=True, fill="both")

        self.btn_up = ttk.Button(controls, text="▲", style="Arrow.TButton", command=self.move_up)
        self.btn_down = ttk.Button(controls, text="▼", style="Arrow.TButton", command=self.move_down)
        self.btn_left = ttk.Button(controls, text="◀", style="Arrow.TButton", command=self.move_left)
        self.btn_right = ttk.Button(controls, text="▶", style="Arrow.TButton", command=self.move_right)

        for r in [0, 1]:
            controls.rowconfigure(r, weight=1)
        for c in [0, 1, 2]:
            controls.columnconfigure(c, weight=1)

        self.btn_up.grid(row=0, column=1, sticky="nsew")
        self.btn_left.grid(row=1, column=0, sticky="nsew")
        self.btn_right.grid(row=1, column=2, sticky="nsew")
        self.btn_down.grid(row=1, column=1, sticky="nsew")

    def move_up(self):
        global y_pos
        y_pos -= self.step_size
        pr.go_to_pos(x_pos, y_pos)

    def move_down(self):
        global y_pos
        y_pos += self.step_size
        pr.go_to_pos(x_pos, y_pos)

    def move_left(self):
        global x_pos
        x_pos -= self.step_size
        pr.go_to_pos(x_pos, y_pos)

    def move_right(self):
        global x_pos
        x_pos += self.step_size
        pr.go_to_pos(x_pos, y_pos)
    
    def on_enter_step_size(self, event=None):
        print(self.step_size)
        self.step_size = int(self.step_entry.get())

    @staticmethod
    def cameraCallback(nEvent, ctx):
        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            ctx.on_image()

    def on_image(self):
        try:
            self.hcam.PullImageV2(self.buf, 24, None)

            img = np.frombuffer(self.buf, dtype=np.uint8).reshape(
                self.height, self.width, 3
            )

            runRegistration(img, self.prevImg)
            self.prevImg = img

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)

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
        self.width, self.height = self.hcam.get_Size()
        bufsize = ((self.width * 24 + 31) // 32 * 4) * self.height
        self.buf = bytes(bufsize)

        self.hcam.StartPullModeWithCallback(self.cameraCallback, self)

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