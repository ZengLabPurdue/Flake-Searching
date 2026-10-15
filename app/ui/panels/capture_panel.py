import cv2
import tkinter as tk
from tkinter import Frame, Label, messagebox
from tkinter import ttk


class CapturePanel:
    def __init__(
        self,
        parent,
        app,
        save_image,
    ):
        self.parent = parent
        self.app = app
        self.save_image = save_image
        self.crop_image_var = tk.BooleanVar(value=True)
        self.vignette_image_var = tk.BooleanVar(value=False)
        self.filter_image_var = tk.BooleanVar(value=False)
        self.auto_video_fps_var = tk.BooleanVar(value=False)
        self.frame = self._build_panel()

    def _build_panel(self):
        frame = Frame(
            self.parent,
            bg="#f0f0f0",
            width=204,
            height=261
        )
        frame.place(relx=1.0, rely=0.0, anchor="ne")

        background = Frame(
            frame,
            bg="white",
            width=200,
            height=259
        )
        background.place(x=2, y=0)

        title = Label(
            frame,
            text="Capture",
            bg="white",
            fg="black",
            font=("TkDefaultFont", 13)
        )
        title.place(relx=0.5, y=10, anchor="n")

        style = ttk.Style()
        style.configure("Save.TButton", background="white")
        style.configure("Save.TButton", relief="flat")

        self.capture_image_button = ttk.Button(
            background,
            text="Save Image",
            style="Save.TButton",
            command=self.save_capture_image
        )
        self.capture_image_button.place(relx=0.5, y=55, anchor="center")
        self.app.register_button(self.capture_image_button)

        style.configure("Capture.TCheckbutton", background="white")

        self.crop_image_checkbox = ttk.Checkbutton(
            background,
            text="Save cropped image",
            variable=self.crop_image_var,
            style="Capture.TCheckbutton",
        )
        self.crop_image_checkbox.place(relx=0.5, y=85, anchor="center")

        self.vignette_image_checkbox = ttk.Checkbutton(
            background,
            text="Apply vignette filter",
            variable=self.vignette_image_var,
            style="Capture.TCheckbutton",
        )
        self.vignette_image_checkbox.place(relx=0.5, y=115, anchor="center")

        self.capture_map_button = ttk.Button(
            background,
            text="Save Map",
            style="Save.TButton",
            command=self.save_map
        )
        self.capture_map_button.place(relx=0.5, y=235, anchor="center")
        self.app.register_button(self.capture_map_button)

        self.filter_image_checkbox = ttk.Checkbutton(
            background,
            text="Apply chip filter",
            variable=self.filter_image_var,
            style="Capture.TCheckbutton",
        )
        self.filter_image_checkbox.place(relx=0.5, y=145, anchor="center")

        self.capture_video_button = ttk.Button(
            background,
            text="Capture Video",
            command=self.toggle_video,
            style="Save.TButton",
        )
        self.capture_video_button.place(relx=0.5, y=175, anchor="center")
        # A transparent icon keeps the native ttk button style and a red dot.
        self.recording_icon = tk.PhotoImage(master=background, width=12, height=12)
        for y in range(12):
            for x in range(12):
                if (x - 5.5) ** 2 + (y - 5.5) ** 2 <= 25:
                    self.recording_icon.put("#c62828", (x, y))
        self.auto_video_fps_checkbox = ttk.Checkbutton(
            background, text="Auto / max FPS",
            variable=self.auto_video_fps_var, style="Capture.TCheckbutton",
        )
        self.auto_video_fps_checkbox.place(relx=0.5, y=205, anchor="center")

        return frame

    def toggle_video(self):
        recorder = self.app.frame_processor.video_recorder
        if recorder.recording:
            recorder.stop()
            self.capture_video_button.configure(text="Saving Video…", state="disabled", image="")
            return
        try:
            if self.app.frame_processor.get_camera() is None:
                raise RuntimeError("Connect a camera before recording video.")
            processor = self.app.frame_processor
            recorder.start(
                resolution=self.app.get_resolution(),
                auto_fps=self.auto_video_fps_var.get(),
                camera_fps=processor.camera_fps,
            )
        except Exception as exc:
            messagebox.showerror("Capture Video Error", str(exc), parent=self.parent)
            return
        self.capture_video_button.configure(
            text="Stop Video", image=self.recording_icon, compound="left",
        )
        self.auto_video_fps_checkbox.state(["disabled"])
        self.parent.after(100, self._poll_video)

    def _poll_video(self):
        recorder = self.app.frame_processor.video_recorder
        if not recorder.done.is_set():
            if not recorder.recording:
                self.capture_video_button.configure(text="Saving Video…", state="disabled", image="")
            self.parent.after(100, self._poll_video)
            return
        self.capture_video_button.configure(
            text="Capture Video", state="normal", image="",
        )
        self.auto_video_fps_checkbox.state(["!disabled"])
        if recorder.error:
            messagebox.showerror(
                "Video Recording Error",
                f"{recorder.error}\n\nRecording folder (video may be incomplete):\n{recorder.folder}",
                parent=self.parent,
            )
        else:
            messagebox.showinfo(
                "Video Saved", f"Video saved to:\n{recorder.video_path}",
                parent=self.parent,
            )

    def _save_with_notification(self, image_name, image=None, **save_options):
        try:
            filepath = self.save_image(image=image, **save_options)
            if filepath is None:
                raise RuntimeError(
                    f"No {image_name.lower()} was available to save."
                )
        except FileNotFoundError as exc:
            messagebox.showwarning(
                "Vignette Filter Unavailable",
                str(exc),
                parent=self.parent,
            )
            return None
        except Exception as exc:
            messagebox.showerror(
                f"Save {image_name} Error",
                f"Could not save the {image_name.lower()}:\n\n{exc}",
                parent=self.parent,
            )
            return None

        messagebox.showinfo(
            f"{image_name} Saved",
            f"{image_name} saved to:\n{filepath}",
            parent=self.parent,
        )
        return filepath

    def save_capture_image(self):
        return self._save_with_notification(
            "Image",
            crop=self.crop_image_var.get(),
            apply_vignette=self.vignette_image_var.get(),
            apply_chip_filter=self.filter_image_var.get(),
        )

    def save_map(self):
        try:
            true_map = self.app.get_true_map()
            true_map_bgr = cv2.cvtColor(true_map, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            messagebox.showerror(
                "Save Map Error",
                f"Could not prepare the map for saving:\n\n{exc}",
                parent=self.parent,
            )
            return None

        return self._save_with_notification("Map", image=true_map_bgr)
