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
        
        # Camera
        self.hcam = None
        self.buf = None
        self.prevImg = None
        self.total = 0
        
        self.img_label = Label(root)
        self.img_label.pack(padx=10, pady=10)
        
        # Stage buttons
        frame = Frame(root)
        frame.pack(pady=10)

        btn_up = Button(frame, text="▲", width=5, height=2, command=self.move_up)
        btn_down = Button(frame, text="▼", width=5, height=2, command=self.move_down)
        btn_left = Button(frame, text="◀", width=5, height=2, command=self.move_left)
        btn_right = Button(frame, text="▶", width=5, height=2, command=self.move_right)

        btn_up.grid(row=0, column=1)
        btn_left.grid(row=1, column=0)
        btn_right.grid(row=1, column=2)
        btn_down.grid(row=1, column=1)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ===== Stage Movement =====
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

    # ===== Camera Callbacks =====
    @staticmethod
    def cameraCallback(nEvent, ctx):
        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            ctx.CameraCallback(nEvent)

    def CameraCallback(self, nEvent):
        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            try:
                self.hcam.PullImageV2(self.buf, 24, None)
                self.total += 1
                img = np.frombuffer(self.buf, dtype=np.uint8).reshape(self.height, self.width, 3)

                # Optional cropping
                cropwidth = min(500, self.height//2, self.width//2)
                img = img[self.height//2-cropwidth:self.height//2+cropwidth,
                          self.width//2-cropwidth:self.width//2+cropwidth]

                # Registration
                runRegistration(img, self.prevImg)
                self.prevImg = img

                # Convert to Tkinter image
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.img_label.imgtk = img_tk
                self.img_label.config(image=img_tk)
            except amcam.HRESULTException as ex:
                print(f"Pull image failed, hr=0x{ex.hr:x}")

    # ===== Start Camera =====
    def run_camera(self):
        cams = amcam.Amcam.EnumV2()
        if len(cams) == 0:
            print("No camera found")
            return
        self.hcam = amcam.Amcam.Open(cams[0].id)
        if not self.hcam:
            print("Failed to open camera")
            return

        width, height = self.hcam.get_Size()
        self.width = width
        self.height = height
        bufsize = ((width * 24 + 31) // 32 * 4) * height
        self.buf = bytes(bufsize)

        self.hcam.StartPullModeWithCallback(self.cameraCallback, self)

    # ===== Cleanup =====
    def on_close(self):
        if self.hcam:
            self.hcam.Close()
        pr.disconnect()
        self.root.destroy()

# ===== Main =====
if __name__ == "__main__":
    root = Tk()
    app = App(root)
    app.run_camera()
    root.mainloop()