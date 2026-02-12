import sys
import os
from pathlib import Path

home_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(home_dir) 
camera_api_path = Path(parent_dir) / "Camera API"

print(os.listdir(camera_api_path))

sys.path.insert(0, str(camera_api_path))
import amcam

print(os.listdir(camera_api_path))

sys.path.insert(0, str(camera_api_path))
import amcam

import amcam
import cv2
import numpy as np

def runRegistration(currImgArray, prevImgArray):
    if prevImgArray is None:
        return 0,0
    # do registration stuff
    # grayscale+float conversions
    gray1 = np.float32(cv2.cvtColor(currImgArray, cv2.COLOR_BGR2GRAY))
    gray2 = np.float32(cv2.cvtColor(prevImgArray, cv2.COLOR_BGR2GRAY))

    # Calculate phase correlation
    (dx, dy), _ = cv2.phaseCorrelate(gray1, gray2)

    # Get image size
    height, width = currImgArray.shape[:2]

    # Calculate number of pixels mapped in X and Y directions
    num_pixels_x = dx * width
    num_pixels_y = dy * height

    print("Number of pixels mapped in X direction:", num_pixels_x)
    print("Number of pixels mapped in Y direction:", num_pixels_y)
    # save this frame's array data for the next frame of registration
    return num_pixels_x, num_pixels_y

class App:
    def __init__(self):
        self.hcam = None
        self.buf = None
        self.total = 0

# the vast majority of callbacks come from amcam.dll/so/dylib internal threads
    @staticmethod
    def cameraCallback(nEvent, ctx):
        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            ctx.CameraCallback(nEvent)

    def CameraCallback(self, nEvent):
        if nEvent == amcam.AMCAM_EVENT_IMAGE:
            self.hcam.PullImageV2(self.buf, 24, None)
            img = np.frombuffer(self.buf, dtype=np.uint8).reshape(self.height, self.width, 3)
            cv2.imshow('image', img)
            key = cv2.waitKey(1) & 0xFF

            # If window closed, stop everything
            if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
                print("\nProgram exited!")
                try:
                    self.hcam.Stop()
                    print("Run")
                except amcam.HRESULTException as ex:
                    print(f"Failed to stop camera: 0x{ex.hr:x}")
                self.hcam.Close()
                self.hcam = None
                print("Run")
                cv2.destroyAllWindows()
                sys.exit()
                return

    def run(self):
        self.prevImg = None
        a = amcam.Amcam.EnumV2()
        if len(a) > 0:
            print('{}: flag = {:#x}, preview = {}, still = {}'.format(
                a[0].displayname, a[0].model.flag, a[0].model.preview, a[0].model.still))
            for r in a[0].model.res:
                print('\t = [{} x {}]'.format(r.width, r.height))

            try:
                self.hcam = amcam.Amcam.Open(a[0].id)
                if not self.hcam:
                    print('failed to open camera')
                    return

                self.width, self.height = self.hcam.get_Size()
                bufsize = ((self.width * 24 + 31) // 32 * 4) * self.height
                self.buf = bytes(bufsize)

                try:
                    self.hcam.StartPullModeWithCallback(self.cameraCallback, self)
                except amcam.HRESULTException as ex:
                    print('failed to start camera, hr=0x{:x}'.format(ex.hr))
                    return

                input('Press ENTER to exit')

            finally:
                if self.hcam:
                    try:
                        self.hcam.Stop()
                    except amcam.HRESULTException as ex:
                        print('failed to stop pull mode, hr=0x{:x}'.format(ex.hr))
                    self.hcam.Close()
                    self.hcam = None
                    self.buf = None

        else:
            print('no camera found')

if __name__ == '__main__':
    app = App()
    app.run()