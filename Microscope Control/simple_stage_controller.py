import os
import sys
import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import amcam
from prior import prior


STEP_SIZE = 5
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

        self.init_manual_control_panel()

        self.hcam = None
        self.buf = None
        self.prevImg = None
        self.width = 0
        self.height = 0

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_manual_control_panel(self):

        self.manual_control_border = Frame(
            self.main_frame,
            bg="#f0f0f0",
            width=204,
            height=104
        )
        self.manual_control_border.place(relx=1.0, rely=0.0, anchor="ne", y=-2)

        self.manual_control_panel = Frame(
            self.manual_control_border,
            bg="white",
            width=200,
            height=102
        )
        self.manual_control_panel.place(x=2, y=0)

        self.manual_control_panel.pack_propagate(False)

        controls = Frame(self.manual_control_panel, bg="white")
        controls.pack()

        btn_up = Button(controls, text="▲", width=5, height=2, command=self.move_up)
        btn_down = Button(controls, text="▼", width=5, height=2, command=self.move_down)
        btn_left = Button(controls, text="◀", width=5, height=2, command=self.move_left)
        btn_right = Button(controls, text="▶", width=5, height=2, command=self.move_right)

        btn_up.grid(row=0, column=1)
        btn_left.grid(row=1, column=0)
        btn_right.grid(row=1, column=2)
        btn_down.grid(row=1, column=1)

    def move_up(self):
        global y_pos
        y_pos -= STEP_SIZE
        pr.go_to_pos(x_pos, y_pos)

    def move_down(self):
        global y_pos
        y_pos += STEP_SIZE
        pr.go_to_pos(x_pos, y_pos)

    def move_left(self):
        global x_pos
        x_pos -= STEP_SIZE
        pr.go_to_pos(x_pos, y_pos)

    def move_right(self):
        global x_pos
        x_pos += STEP_SIZE
        pr.go_to_pos(x_pos, y_pos)

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

            # Run registration on full image
            runRegistration(img, self.prevImg)
            self.prevImg = img

            # Convert for Tk
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