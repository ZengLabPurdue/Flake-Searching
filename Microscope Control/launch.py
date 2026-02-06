import os
import subprocess
import sys
from tkinter import *
from tkinter import ttk
from pathlib import Path

homedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(homedir)

def on_close():
    file = open("port.config", "w")
    
    pr_args_final = 0 if pr_args.get() == "" else pr_args.get()

    file.write(f"{pr_args_final}\n")
    file.close()

    root.destroy()

    homedir = os.path.dirname(os.path.abspath(__file__))
    #motor_control_dir = Path(homedir) / "Motor Control"
    #os.chdir(motor_control_dir)

    python_exe = Path(homedir) / ".venv" / "Scripts" / "python.exe"
    script = Path(homedir) / "simple_stage_controller.py"

    cmd = [
        str(python_exe),
        str(script),
        str(pr_args_final),
    ]

    subprocess.run(cmd, check=True)

file = open("port.config", "r")

root = Tk()
root.title("Setup")
Prior_Label = Label(root, text="Prior COM Port:")

pr_args = StringVar(value=file.readline()[:-1])

Pr_Textbox = Entry(root, textvariable=pr_args)

file.close()

Confirm_Button = Button(root, text="Confirm Setting", command=on_close)

Prior_Label.grid(column=0, row=1, sticky="nsew")

Pr_Textbox.grid(column=1, row=1, sticky="nsew")

Confirm_Button.grid(column=0, row=3, columnspan=2)

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()