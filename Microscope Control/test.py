import subprocess
import os

dll_folder = r"C:\Users\Zengl\OneDrive\Desktop\Flake-Searching\Microscope Control\PriorSDK1.9.2\x64"

env = os.environ.copy()
env["PATH"] = dll_folder + ";" + env["PATH"]  # prepend the DLL folder

cmd = [
    r"C:\Users\Zengl\OneDrive\Desktop\Flake-Searching\.venv\Scripts\python.exe",
    r"C:\Users\Zengl\OneDrive\Desktop\Flake-Searching\Microscope Control\scanner.py",
    "3"
]

subprocess.run(cmd, check=True, cwd=r"C:\Users\Zengl\OneDrive\Desktop\Flake-Searching\Microscope Control", env=env)