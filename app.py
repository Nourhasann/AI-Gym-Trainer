"""
Tkinter GUI for the AI Gym Trainer.

Features:
- Dark theme, gradient header
- Compact, letterboxed camera display
- Circular rep-progress ring
- Correct / incorrect statistics
- Exercise detection
- Left / Right / Both arm selection
- Workout report

Threading model:
    The camera + MediaPipe processing happens on a background thread
    (video_loop). It NEVER touches Tkinter widgets directly -- Tkinter is
    not thread-safe, and doing so causes exactly the kind of visual
    glitching/tearing you get from updating widgets off the main thread.
    Instead, each finished frame + state is placed on a queue. A separate
    loop (_process_queue) runs on the MAIN thread via root.after() and is
    the only place that ever touches a Tkinter widget.

Usage:
    python app.py
"""

import math
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import scrolledtext

import cv2
from PIL import Image, ImageTk, ImageDraw
import mediapipe as mp

from core.pose_estimator import PoseEstimator
from core.classifier import ExercisePredictor
from exercises import EXERCISES
from core.report import generate_report, save_report

mp_pose = mp.solutions.pose

# ---- trendy dark theme palette (deep indigo/navy base, violet->pink accents) ----
BG = "#0a0e1a"
PANEL = "#12172a"
CARD = "#171d33"
BORDER = "#2a3152"
TEXT = "#f2f4fb"
MUTED = "#8891ad"
ACCENT = "#8b6bff"          # violet
ACCENT2 = "#ff5ca8"         # pink (used for gradients/highlights)
GOOD = "#22e0a4"            # neon mint
BAD = "#ff5470"             # neon coral
BTN_BG = "#7c5cff"
BTN_BG_HOVER = "#9c81ff"
BTN_DISABLED = "#1c2238"
BTN_STOP_BG = "#ff5470"
BTN_STOP_HOVER = "#ff7791"

REP_TARGET = 12          # ring fills once every REP_TARGET reps, purely visual pacing
VIDEO_W, VIDEO_H = 560, 300

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def rounded_rect(canvas, x1, y1, x2, y2, radius=16, **kwargs):
    points = [
        x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
        x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
        x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


def make_gradient(width, height, c1, c2, vertical=False):
    base = Image.new("RGB", (width, height), c1)
    top = Image.new("RGB", (width, height), c2)
    mask = Image.new("L", (width, height))
    if vertical:
        mask_data = []
        for y in range(height):
            mask_data.extend([int(255 * (y / height))] * width)
    else:
        mask_data = [int(255 * (x / width)) for x in range(width)] * height
    mask.putdata(mask_data)
    base.paste(top, (0, 0), mask)
    return ImageTk.PhotoImage(base)


def build_shimmer_frames(width, height, color_pairs, steps_per_pair=12):
    """
    Precomputes a looping sequence of gradient frames that smoothly cross-fade
    through the given (c1, c2) color pairs, for a subtle animated "shimmer"
    header. Purely decorative -- no effect on app behavior.
    """
    raw_images = []
    for c1, c2 in color_pairs:
        base = Image.new("RGB", (width, height), c1)
        top = Image.new("RGB", (width, height), c2)
        mask = Image.new("L", (width, height))
        mask_data = [int(255 * (x / width)) for x in range(width)] * height
        mask.putdata(mask_data)
        img = base.copy()
        img.paste(top, (0, 0), mask)
        raw_images.append(img)

    frames = []
    n = len(raw_images)
    for i in range(n):
        img_a = raw_images[i]
        img_b = raw_images[(i + 1) % n]
        for s in range(steps_per_pair):
            t = s / steps_per_pair
            frames.append(ImageTk.PhotoImage(Image.blend(img_a, img_b, t)))
    return frames


def load_cover(path, width, height):
    """Loads an image and scales+center-crops it to exactly fill (width, height)
    with no distortion -- like CSS 'background-size: cover'. Good for photos
    whose aspect ratio already roughly matches the target frame."""
    img = Image.open(path).convert("RGB")
    src_ratio = img.width / img.height
    tgt_ratio = width / height
    if src_ratio > tgt_ratio:
        new_h = height
        new_w = max(width, int(round(height * src_ratio)))
    else:
        new_w = width
        new_h = max(height, int(round(width / src_ratio)))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    x = (new_w - width) // 2
    y = (new_h - height) // 2
    return img.crop((x, y, x + width, y + height))


def make_circular_badge(path, size, ring_color, ring_width=4):
    """Loads a photo and returns a circular thumbnail with a colored ring border,
    on a transparent background -- used for the small header photo badge."""
    photo = load_cover(path, size, size)
    photo = photo.convert("RGBA")

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, size, size], fill=255)

    circular = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    circular.paste(photo, (0, 0), mask)

    ring_size = size + ring_width * 2
    out = Image.new("RGBA", (ring_size, ring_size), (0, 0, 0, 0))
    ImageDraw.Draw(out).ellipse([0, 0, ring_size, ring_size], fill=ring_color)
    out.paste(circular, (ring_width, ring_width), circular)
    return out


def load_letterboxed(path, width, height, bg="#0d1117"):
    """Loads an image file and letterboxes it into a fixed-size RGB canvas,
    matching the same fit-without-distortion approach used for the camera feed."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), bg)
    x = (width - img.width) // 2
    y = (height - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


class GymTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GymSpot")
        self.root.configure(bg=BG)
        self.root.geometry("660x870")
        self.root.minsize(500, 400)
        self.root.resizable(True, True)

        # ---------------- scrollable container ----------------
        # The full UI (header, video, stats, gallery, etc.) is taller than many
        # screens, so everything is packed into a scrollable canvas instead of
        # directly onto root. This is purely a layout container -- nothing here
        # touches app state or the video/threading pipeline.
        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True)
        self.scroll_canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(outer, orient="vertical", command=self.scroll_canvas.yview,
                                       bg=PANEL, troughcolor=BG)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        content = tk.Frame(self.scroll_canvas, bg=BG)
        self._content_window = self.scroll_canvas.create_window((0, 0), window=content, anchor="nw")

        def _on_content_configure(event):
            self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

        def _on_canvas_configure(event):
            self.scroll_canvas.itemconfig(self._content_window, width=event.width)

        content.bind("<Configure>", _on_content_configure)
        self.scroll_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            if event.num == 4:
                self.scroll_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.scroll_canvas.yview_scroll(1, "units")
            else:
                self.scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.scroll_canvas.bind_all("<Button-4>", _on_mousewheel)
        self.scroll_canvas.bind_all("<Button-5>", _on_mousewheel)

        # Exercise switching is deliberately conservative so one noisy frame
        # cannot turn an idle person into a Bicep Curl (or switch exercises).
        self.pending_label = None
        self.pending_count = 0
        self.EXERCISE_CONFIRM_FRAMES = 12
        self.STARTUP_IDLE_SECONDS = 3.0
        self.camera_start_time = None

        root = content  # everything below is built inside the scrollable frame

        # ---------------- header ----------------
        header_h = 68
        self.header_canvas = tk.Canvas(root, width=640, height=header_h, highlightthickness=0, bg=BG)
        self.header_canvas.pack(fill="x")
        # Precomputed frames for a slow, looping shimmer across the header gradient.
        self._header_frames = build_shimmer_frames(
            640, header_h,
            [("#5b21ff", "#ff2d8a"), ("#7c3aed", "#f472b6"), ("#4338ca", "#ec4899")],
            steps_per_pair=12,
        )
        self._header_frame_idx = 0
        self._header_image_id = self.header_canvas.create_image(
            0, 0, image=self._header_frames[0], anchor="nw")

        # GymSpot logo (icon + wordmark), placed at the left of the header.
        logo_path = os.path.join(ASSETS_DIR, "gymspot_logo.png")
        if os.path.exists(logo_path):
            logo_img = Image.open(logo_path).convert("RGBA")
            scale = 52 / logo_img.height
            logo_img = logo_img.resize((int(logo_img.width * scale), 52), Image.Resampling.LANCZOS)
            self._logo_img = ImageTk.PhotoImage(logo_img)
            self.header_canvas.create_image(16, header_h // 2, image=self._logo_img, anchor="w")
        self.header_canvas.create_text(430, header_h // 2 - 8, text="YOUR AI SPOTTER",
                                        fill="#f3e8ff", font=("Segoe UI", 8, "bold"))
        self.header_canvas.create_text(430, header_h // 2 + 10, text="\u2728 form check \u2022 rep count \u2022 real-time",
                                        fill="#f3e8ff", font=("Segoe UI", 7))

        # Small circular "motivation" photo badge, top-right of the header --
        # always visible, no scrolling required.
        self._motivation_pulse_on = False
        motivation_path = os.path.join(ASSETS_DIR, "motivation_1.jpg")
        if os.path.exists(motivation_path):
            badge_size = 44
            ring_pad = 5
            bx, by = 596, header_h // 2
            self._motivation_glow_id = self.header_canvas.create_oval(
                bx - badge_size // 2 - ring_pad - 3, by - badge_size // 2 - ring_pad - 3,
                bx + badge_size // 2 + ring_pad + 3, by + badge_size // 2 + ring_pad + 3,
                outline=ACCENT2, width=2, fill="")
            badge_img = make_circular_badge(motivation_path, badge_size, "#ffffff", ring_width=ring_pad)
            self._motivation_badge_img = ImageTk.PhotoImage(badge_img)
            self.header_canvas.create_image(bx, by, image=self._motivation_badge_img, anchor="center")
        else:
            self._motivation_glow_id = None

        # Use the icon (square, no wordmark) as the window/taskbar icon if available.
        icon_path = os.path.join(ASSETS_DIR, "gymspot_icon.png")
        if os.path.exists(icon_path):
            self._icon_img = ImageTk.PhotoImage(Image.open(icon_path).convert("RGBA"))
            self.root.iconphoto(False, self._icon_img)

        # ---------------- video (fixed size, letterboxed) ----------------
        self.video_border = tk.Frame(root, bg=BORDER, padx=3, pady=3, width=VIDEO_W + 6, height=VIDEO_H + 6)
        self.video_border.pack(pady=(14, 6))
        self.video_border.pack_propagate(False)
        self.video_label = tk.Label(self.video_border, bg=PANEL)
        self.video_label.pack(fill="both", expand=True)

        # ---------------- status row ----------------
        status_row = tk.Frame(root, bg=BG)
        status_row.pack(pady=(4, 0))
        self.status_dot = tk.Canvas(status_row, width=14, height=14, bg=BG, highlightthickness=0)
        self.status_dot_id = self.status_dot.create_oval(2, 2, 12, 12, fill=MUTED, outline="")
        self.status_dot.pack(side=tk.LEFT, padx=(0, 8))
        self.exercise_label = tk.Label(status_row, text="IDLE", font=("Segoe UI", 14, "bold"), bg=BG, fg=MUTED)
        self.exercise_label.pack(side=tk.LEFT)

        # ---------------- rep ring ----------------
        ring_size = 120
        self.ring_canvas = tk.Canvas(root, width=ring_size, height=ring_size, bg=BG, highlightthickness=0)
        self.ring_canvas.pack(pady=(6, 0))
        pad = 8
        self._ring_bg = self.ring_canvas.create_oval(pad, pad, ring_size - pad, ring_size - pad,
                                                       outline=BORDER, width=9)
        self._ring_arc = self.ring_canvas.create_arc(pad, pad, ring_size - pad, ring_size - pad,
                                                       start=90, extent=0, style="arc", outline=ACCENT, width=9)
        self.reps_text = self.ring_canvas.create_text(ring_size / 2, ring_size / 2 - 5, text="0",
                                                        fill=TEXT, font=("Segoe UI", 30, "bold"))
        self.ring_canvas.create_text(ring_size / 2, ring_size / 2 + 25, text="REPS", fill=MUTED,
                                      font=("Segoe UI", 8, "bold"))

        # ---------------- stat cards ----------------
        cards_row = tk.Frame(root, bg=BG)
        cards_row.pack(pady=(8, 0))

        self.correct_canvas = tk.Canvas(cards_row, width=215, height=54, bg=BG, highlightthickness=0)
        self.correct_canvas.pack(side=tk.LEFT, padx=5)
        self._correct_card_bg = rounded_rect(self.correct_canvas, 2, 2, 213, 52, radius=16,
                                              fill=CARD, outline=GOOD, width=1)
        self.correct_canvas.create_text(30, 27, text="\u2713", fill=GOOD, font=("Segoe UI", 19, "bold"))
        self.correct_text = self.correct_canvas.create_text(126, 27, text="Correct: 0",
                                                              fill=TEXT, font=("Segoe UI", 11, "bold"))

        self.incorrect_canvas = tk.Canvas(cards_row, width=215, height=54, bg=BG, highlightthickness=0)
        self.incorrect_canvas.pack(side=tk.LEFT, padx=5)
        self._incorrect_card_bg = rounded_rect(self.incorrect_canvas, 2, 2, 213, 52, radius=16,
                                                fill=CARD, outline=BAD, width=1)
        self.incorrect_canvas.create_text(30, 27, text="\u2717", fill=BAD, font=("Segoe UI", 19, "bold"))
        self.incorrect_text = self.incorrect_canvas.create_text(126, 27, text="Incorrect: 0",
                                                                  fill=TEXT, font=("Segoe UI", 11, "bold"))

        # ---------------- feedback banner ----------------
        self.feedback_canvas = tk.Canvas(root, width=570, height=40, bg=BG, highlightthickness=0)
        self.feedback_canvas.pack(pady=(8, 4))
        self._feedback_bg = rounded_rect(self.feedback_canvas, 2, 2, 568, 38, radius=12,
                                          fill=CARD, outline=BORDER, width=1)
        self.feedback_text_id = self.feedback_canvas.create_text(
            285, 20, text="Choose an arm and press Start Workout", fill=MUTED, font=("Segoe UI", 10))

        # ---------------- arm selection ----------------
        arm_frame = tk.Frame(root, bg=BG)
        arm_frame.pack(pady=(4, 4))
        tk.Label(arm_frame, text="Arm to Track:", bg=BG, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))

        self.arm_selection = tk.StringVar(value="both")
        radio_kwargs = dict(bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                             activeforeground=ACCENT, font=("Segoe UI", 9))
        self.left_radio = tk.Radiobutton(arm_frame, text="Left", variable=self.arm_selection,
                                          value="left", **radio_kwargs)
        self.left_radio.pack(side=tk.LEFT, padx=5)
        self.right_radio = tk.Radiobutton(arm_frame, text="Right", variable=self.arm_selection,
                                           value="right", **radio_kwargs)
        self.right_radio.pack(side=tk.LEFT, padx=5)
        self.both_radio = tk.Radiobutton(arm_frame, text="Both", variable=self.arm_selection,
                                          value="both", **radio_kwargs)
        self.both_radio.pack(side=tk.LEFT, padx=5)

        # ---------------- buttons ----------------
        button_frame = tk.Frame(root, bg=BG)
        button_frame.pack(pady=(4, 5))
        self.start_button = tk.Button(
            button_frame, text="\u25B6  Start Workout", command=self.start_workout,
            bg=BTN_BG, fg="white", activebackground=BTN_BG_HOVER, activeforeground="white",
            font=("Segoe UI", 10, "bold"), relief="flat", padx=18, pady=7, bd=0, cursor="hand2",
            highlightthickness=2, highlightbackground=BTN_BG, highlightcolor=BTN_BG)
        self.start_button.pack(side=tk.LEFT, padx=6)
        self.start_button.bind("<Enter>", lambda e: self._hover(self.start_button, True))
        self.start_button.bind("<Leave>", lambda e: self._hover(self.start_button, False))

        self.stop_button = tk.Button(
            button_frame, text="\u25A0  Finish Workout", command=self.stop_workout,
            bg=BTN_DISABLED, fg=MUTED, font=("Segoe UI", 10, "bold"),
            relief="flat", padx=18, pady=7, bd=0, state=tk.DISABLED, cursor="hand2")
        self.stop_button.pack(side=tk.LEFT, padx=6)

        # ---------------- report box ----------------
        self.report_box = scrolledtext.ScrolledText(
            root, width=62, height=4, bg=PANEL, fg=TEXT, insertbackground=TEXT,
            font=("Consolas", 9), relief="flat", bd=0, padx=10, pady=8,
            highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT)
        self.report_box.pack(pady=(2, 5))

        # ---------------- state ----------------
        self.running = False
        self.thread = None
        self.cap = None
        self.estimator = None
        self.predictor = None
        self.active_label = None
        self.active_exercise = None
        self.completed_exercises = []
        self._last_reps = 0
        self._dot_on = False
        self._dots_phase = 0
        self._ring_extent = 0.0
        self.selected_arm = "both"
        self._last_correct = 0
        self._last_incorrect = 0
        self._start_pulse_on = False

        # Thread-safe handoff between the background camera thread and the
        # main-thread UI. Only ever written to from video_loop() and read
        # from _process_queue(). maxsize=2 means if the UI falls behind,
        # we drop old frames instead of building up a backlog/lag.
        self.frame_queue = queue.Queue(maxsize=2)

        # ---------------- decorative, always-on animations ----------------
        self._animate_header()
        self._pulse_start_button()
        self._breathe_ring()
        if self._motivation_glow_id is not None:
            self._pulse_motivation_badge()

    def _pulse_motivation_badge(self):
        """Soft glow pulse around the header's circular motivation photo."""
        self._motivation_pulse_on = not self._motivation_pulse_on
        color = ACCENT if self._motivation_pulse_on else ACCENT2
        self.header_canvas.itemconfig(self._motivation_glow_id, outline=color)
        self.root.after(700, self._pulse_motivation_badge)

    # ---------------- hover / small helpers ----------------

    def _animate_header(self):
        """Loops through precomputed gradient frames for a slow shimmer effect."""
        self._header_frame_idx = (self._header_frame_idx + 1) % len(self._header_frames)
        self.header_canvas.itemconfig(self._header_image_id, image=self._header_frames[self._header_frame_idx])
        self.root.after(90, self._animate_header)

    def _pulse_start_button(self):
        """Gentle breathing glow around Start Workout while it's clickable."""
        if str(self.start_button["state"]) == "normal":
            self._start_pulse_on = not self._start_pulse_on
            color = ACCENT2 if self._start_pulse_on else BTN_BG
            self.start_button.configure(highlightbackground=color, highlightcolor=color)
        else:
            self.start_button.configure(highlightbackground=BTN_DISABLED, highlightcolor=BTN_DISABLED)
        self.root.after(650, self._pulse_start_button)

    def _breathe_ring(self):
        """Subtle ambient thickness pulse on the rep ring's background track when idle."""
        if not self.running:
            w = 7 + int(round(2 * math.sin(time.time() * 2)))
            self.ring_canvas.itemconfig(self._ring_bg, width=w)
        self.root.after(80, self._breathe_ring)

    def _animate_card_pulse(self, canvas, item_id, step=0):
        """Smooth multi-step border pulse on a stat card when its value changes."""
        widths = [1, 2, 4, 3, 2, 1]
        if step >= len(widths):
            return
        canvas.itemconfig(item_id, width=widths[step])
        self.root.after(80, lambda: self._animate_card_pulse(canvas, item_id, step + 1))

    def _hover(self, button, entering):
        if button["state"] == tk.DISABLED:
            return
        if button is self.stop_button:
            button.configure(bg=BTN_STOP_HOVER if entering else BTN_STOP_BG)
        else:
            button.configure(bg=BTN_BG_HOVER if entering else BTN_BG)

    def _pulse_dot(self):
        if self.running:
            self._dot_on = not self._dot_on
            color = ACCENT2 if self._dot_on else "#5c1e4a"
            self.status_dot.itemconfig(self.status_dot_id, fill=color)
            self.video_border.configure(bg=ACCENT if self._dot_on else BORDER)
        else:
            self.status_dot.itemconfig(self.status_dot_id, fill=MUTED)
            self.video_border.configure(bg=BORDER)
        self.root.after(600, self._pulse_dot)

    def _animate_detecting(self):
        if self.running and self.active_exercise is None:
            self._dots_phase = (self._dots_phase + 1) % 4
            self.exercise_label.configure(text="DETECTING" + "." * self._dots_phase, fg=ACCENT)
        self.root.after(400, self._animate_detecting)

    def _animate_ring(self, target_extent):
        diff = target_extent - self._ring_extent
        self._ring_extent = target_extent if abs(diff) < 1 else self._ring_extent + diff * 0.25
        self.ring_canvas.itemconfig(self._ring_arc, extent=self._ring_extent)
        color = ACCENT if self._ring_extent < 359 else GOOD
        self.ring_canvas.itemconfig(self._ring_arc, outline=color)
        if abs(target_extent - self._ring_extent) >= 1:
            self.root.after(16, lambda: self._animate_ring(target_extent))

    def _flash_reps(self):
        self.ring_canvas.itemconfig(self.reps_text, fill=ACCENT2)
        self.root.after(200, lambda: self.ring_canvas.itemconfig(self.reps_text, fill=TEXT))

    def _flash_feedback(self, is_warning):
        color = BAD if is_warning else GOOD
        self.feedback_canvas.itemconfig(self._feedback_bg, outline=color)
        self.root.after(400, lambda: self.feedback_canvas.itemconfig(self._feedback_bg, outline=BORDER))

    # ---------------- workout control ----------------

    def start_workout(self):
        self.selected_arm = self.arm_selection.get()

        self.running = True
        self.start_button.config(state=tk.DISABLED, bg=BTN_DISABLED, fg=MUTED)
        self.stop_button.config(state=tk.NORMAL, bg=BTN_STOP_BG, fg="white")
        self.stop_button.bind("<Enter>", lambda e: self._hover(self.stop_button, True))
        self.stop_button.bind("<Leave>", lambda e: self._hover(self.stop_button, False))
        self.report_box.delete("1.0", tk.END)
        self._last_reps = 0
        self._ring_extent = 0.0
        self._last_correct = 0
        self._last_incorrect = 0

        self.left_radio.config(state=tk.DISABLED)
        self.right_radio.config(state=tk.DISABLED)
        self.both_radio.config(state=tk.DISABLED)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.estimator = PoseEstimator()

        # Every workout starts in IDLE. The classifier is ignored briefly while
        # MediaPipe/camera landmarks stabilize, preventing a false Bicep Curl
        # as soon as the camera opens.
        self.active_label = "idle"
        self.active_exercise = None
        self.pending_label = None
        self.pending_count = 0
        self.camera_start_time = time.time()

        # Keep one exercise instance per label for the entire workout.
        # This is what lets Bicep Curl -> Shoulder Press -> Bicep Curl
        # continue the original Bicep counter instead of resetting it.
        self.completed_exercises = []
        self.exercise_instances = {}

        model_path = "models/exercise_classifier.pkl"
        if os.path.exists(model_path):
            self.predictor = ExercisePredictor(model_path=model_path)
        else:
            self.predictor = None
            self.active_label = "idle"
            self.active_exercise = None
            self.feedback_canvas.itemconfig(
            self.feedback_text_id,
           text="Model not found. Train the classifier first.",
    )

        # Drain any stale frames from a previous session before starting fresh.
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

        self._pulse_dot()
        self._animate_detecting()
        self._process_queue()

    def video_loop(self):
        """
        Runs on a BACKGROUND thread. Does camera capture, pose estimation,
        exercise updates, and image prep -- but never touches a Tkinter
        widget. Finished (image, state) pairs go on frame_queue for the
        main thread to pick up.
        """
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            image, landmarks, pose_landmarks_raw = self.estimator.process(frame)

            if landmarks is not None:
                if self.predictor is not None:
                    predicted_label = self.predictor.predict(landmarks, mp_pose)

                    # ---------------------------------------------------------
                    # Stable automatic exercise switching
                    # ---------------------------------------------------------
                    # During the first few seconds, always stay IDLE. This
                    # prevents a startup misclassification from creating an
                    # exercise before the camera/pose landmarks settle.
                    startup_elapsed = (
                        time.time() - self.camera_start_time
                        if self.camera_start_time is not None
                        else self.STARTUP_IDLE_SECONDS
                    )

                    if startup_elapsed < self.STARTUP_IDLE_SECONDS:
                        self.active_label = "idle"
                        self.active_exercise = None
                        self.pending_label = None
                        self.pending_count = 0

                    elif predicted_label is not None:
                        # Count consecutive predictions for the same label.
                        # We only switch after enough agreement.
                        if predicted_label == self.pending_label:
                            self.pending_count += 1
                        else:
                            self.pending_label = predicted_label
                            self.pending_count = 1

                        if self.pending_count >= self.EXERCISE_CONFIRM_FRAMES:
                            confirmed_label = self.pending_label

                            if confirmed_label != self.active_label:
                                self.active_label = confirmed_label

                                if confirmed_label == "idle":
                                    # Resting: stop updating any exercise, but
                                    # keep all exercise instances in memory.
                                    self.active_exercise = None

                                elif confirmed_label in self.exercise_instances:
                                    # Returning to an exercise: REUSE the same
                                    # object, so its reps/correct/incorrect counts
                                    # continue where they left off.
                                    self.active_exercise = self.exercise_instances[confirmed_label]

                                else:
                                    exercise_cls = EXERCISES.get(confirmed_label)
                                    if exercise_cls:
                                        if confirmed_label == "bicep_curl":
                                            self.active_exercise = exercise_cls(
                                                arm_selection=self.selected_arm
                                            )
                                        else:
                                            self.active_exercise = exercise_cls()

                                        self.exercise_instances[confirmed_label] = self.active_exercise
                                    else:
                                        # Unknown/non-exercise labels are treated
                                        # as idle rather than updating the wrong
                                        # exercise.
                                        self.active_label = "idle"
                                        self.active_exercise = None

                            # Once a label is confirmed, require a fresh run of
                            # consecutive predictions before switching again.
                            self.pending_label = None
                            self.pending_count = 0

                # IMPORTANT: only the currently confirmed exercise receives
                # landmark updates. Therefore Bicep Curl cannot count while
                # Shoulder Press (or IDLE) is active.
                if self.active_exercise is not None:
                    try:
                        self.active_exercise.update(landmarks, mp_pose)
                    except Exception as e:
                        print(f"Update error: {e}")

            image = self.estimator.draw(image, pose_landmarks_raw)

            # Letterbox into a fixed-size canvas so the video panel never resizes/jitters.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image_rgb)
            img.thumbnail((VIDEO_W, VIDEO_H), Image.Resampling.LANCZOS)
            display_image = Image.new("RGB", (VIDEO_W, VIDEO_H), "black")
            x = (VIDEO_W - img.width) // 2
            y = (VIDEO_H - img.height) // 2
            display_image.paste(img, (x, y))

            state = self.active_exercise.get_state() if self.active_exercise is not None else None

            # Non-blocking: if the UI hasn't caught up yet, drop the oldest
            # queued frame rather than let the queue (and lag) grow.
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_queue.put_nowait((display_image, state))
            except queue.Full:
                pass

            time.sleep(0.005)

    def _process_queue(self):
        """
        Runs on the MAIN thread via root.after(). This is the only place
        that touches Tkinter widgets while a workout is running -- fixes
        the glitching caused by the old version updating widgets directly
        from the background thread.
        """
        try:
            display_image, state = self.frame_queue.get_nowait()

            imgtk = ImageTk.PhotoImage(image=display_image)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            if state is not None:
                self.exercise_label.configure(text=state["exercise"].upper(), fg=TEXT)
                self.ring_canvas.itemconfig(self.reps_text, text=str(state["counter"]))
                self.correct_canvas.itemconfig(self.correct_text, text=f"Correct: {state['correct']}")
                self.incorrect_canvas.itemconfig(self.incorrect_text, text=f"Incorrect: {state['incorrect']}")

                if state["correct"] != self._last_correct:
                    self._last_correct = state["correct"]
                    self._animate_card_pulse(self.correct_canvas, self._correct_card_bg)
                if state["incorrect"] != self._last_incorrect:
                    self._last_incorrect = state["incorrect"]
                    self._animate_card_pulse(self.incorrect_canvas, self._incorrect_card_bg)

                feedback = state["feedback"] or "Good form"
                is_warning = bool(state["feedback"]) and "keep" in state["feedback"].lower()
                self.feedback_canvas.itemconfig(
                    self.feedback_text_id, text=feedback, fill=(BAD if is_warning else GOOD))

                progress = (state["counter"] % REP_TARGET) / REP_TARGET
                if state["counter"] > 0 and state["counter"] % REP_TARGET == 0:
                    progress = 1.0
                self._animate_ring(360 * progress)

                if state["counter"] != self._last_reps:
                    self._last_reps = state["counter"]
                    self._flash_reps()
                    self._flash_feedback(is_warning)
            else:
                # No active exercise means the confirmed state is IDLE.
                self.exercise_label.configure(text="IDLE", fg=MUTED)
                self.feedback_canvas.itemconfig(
                    self.feedback_text_id,
                    text="Ready for your next exercise",
                    fill=MUTED,
                )
                self.ring_canvas.itemconfig(self.reps_text, text="0")
        except queue.Empty:
            pass

        if self.running:
            self.root.after(15, self._process_queue)

    def stop_workout(self):
        self.running = False
        time.sleep(0.2)

        self.completed_exercises = list(self.exercise_instances.values())

        if self.cap is not None:
            self.cap.release()
        if self.estimator is not None:
            self.estimator.close()

        self.left_radio.config(state=tk.NORMAL)
        self.right_radio.config(state=tk.NORMAL)
        self.both_radio.config(state=tk.NORMAL)

        self.start_button.config(state=tk.NORMAL, bg=BTN_BG, fg="white")
        self.stop_button.config(state=tk.DISABLED, bg=BTN_DISABLED, fg=MUTED)
        self.exercise_label.configure(text="IDLE", fg=MUTED)

        report = generate_report(self.completed_exercises)
        save_report(report)
        self.display_report(report)

    def display_report(self, report):
        self.report_box.delete("1.0", tk.END)
        self.report_box.insert(tk.END, "\u2728 WORKOUT REPORT\n")
        self.report_box.insert(tk.END, f"Total reps: {report['total_reps']}\n")
        self.report_box.insert(tk.END, f"Correct: {report['total_correct']}  Incorrect: {report['total_incorrect']}\n")
        self.report_box.insert(tk.END, f"Overall form score: {report['overall_form_score']}%\n\n")
        for ex in report["exercises"]:
            self.report_box.insert(
                tk.END,
                f"- {ex['name']}: {ex['total_reps']} reps "
                f"({ex['correct_reps']} correct / {ex['incorrect_reps']} incorrect), "
                f"form score {ex['form_score']}%\n"
            )
        if report["common_mistakes"]:
            self.report_box.insert(tk.END, "\nCommon mistakes:\n")
            for mistake, count in report["common_mistakes"].items():
                self.report_box.insert(tk.END, f"- {mistake}: {count}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = GymTrainerApp(root)
    root.mainloop()