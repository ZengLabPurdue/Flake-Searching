import os
import sys
from tkinter import *
from prior import prior

# CONFIG

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

def move_up():
    global y_pos
    y_pos -= STEP_SIZE
    pr.go_to_pos(x_pos, y_pos)

def move_down():
    global y_pos
    y_pos += STEP_SIZE
    pr.go_to_pos(x_pos, y_pos)

def move_left():
    global x_pos
    x_pos -= STEP_SIZE
    pr.go_to_pos(x_pos, y_pos)

def move_right():
    global x_pos
    x_pos += STEP_SIZE
    pr.go_to_pos(x_pos, y_pos)

def on_close():
    pr.disconnect()
    root.destroy()

root = Tk()
root.title("Prior XY Control")

frame = Frame(root)
frame.pack(padx=20, pady=20)

btn_up = Button(frame, text="▲", width=5, height=2, command=move_up)
btn_down = Button(frame, text="▼", width=5, height=2, command=move_down)
btn_left = Button(frame, text="◀", width=5, height=2, command=move_left)
btn_right = Button(frame, text="▶", width=5, height=2, command=move_right)

# Arrow-key layout
btn_up.grid(row=0, column=1)
btn_left.grid(row=1, column=0)
btn_right.grid(row=1, column=2)
btn_down.grid(row=1, column=1)

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()