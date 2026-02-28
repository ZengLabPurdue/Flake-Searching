import os
import subprocess
from pathlib import Path
from tkinter import Tk, StringVar
from tkinter import ttk
from tkinter import filedialog

homedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(homedir)

# Hello roommate

file_path = filedialog.askopenfilename(
    title="Select Python file",
    filetypes=[("Python files", "*.py")]
)

def on_close():
    pr_args_final = pr_args.get() or "0"

    with open("port.config", "w") as f:
        f.write(f"{pr_args_final}\n")

    root.destroy()

    python_exe = Path(homedir) / ".venv" / "Scripts" / "python.exe"
    script = file_path

    cmd = [str(python_exe), str(script), str(pr_args_final)]
    subprocess.run(cmd, check=True)

try:
    with open("port.config", "r") as f:
        pr_args_value = f.readline().strip()
except FileNotFoundError:
    pr_args_value = ""

root = Tk()
root.title("Launch")
root.resizable(False, False)

frame = ttk.Frame(root, padding=15)
frame.grid(row=0, column=0)

prior_label = ttk.Label(frame, text="Prior COM Port:")
prior_label.grid(row=0, column=0, padx=(0,10), pady=(0,10), sticky="e")

pr_args = StringVar(value=pr_args_value)
pr_entry = ttk.Entry(frame, textvariable=pr_args, width=10)
pr_entry.grid(row=0, column=1, pady=(0,10), sticky="w")
pr_entry.focus()

confirm_btn = ttk.Button(frame, text="Launch", command=on_close)
confirm_btn.grid(row=1, column=0, columnspan=2, pady=(5,0))

root.bind("<Return>", lambda e: on_close())

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()