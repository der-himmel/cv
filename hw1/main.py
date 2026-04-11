import cv2
import tkinter as tk
from PIL import Image, ImageTk
import platform
import sys
import os

VIDEO_CAP_BACKENDS = [
    cv2.CAP_ANY,
    # getattr(cv2, "CAP_V4L2", 200),          # linux
    # getattr(cv2, "CAP_DSHOW", 700),         # windows
    # getattr(cv2, "CAP_MSMF", 1400),         # windows_alt
    # getattr(cv2, "CAP_AVFOUNDATION", 1200), # mac
]
MAX_VIDEO_CAP_SOURCES_QUANTITY = 3
WIDTH = 704
HEIGHT = 576
RECT_SIZE = 20  # размер прямоугольника

def get_os_info():
    os_info = platform.platform()
    if "linux" in os_info.lower():
        return "linux"
    if "windows" in os_info.lower():
        return "windows"

def detect_video_sources():
    vidstream_sources = []
    try:
        for idx in range(0, MAX_VIDEO_CAP_SOURCES_QUANTITY):
            video_capture = None
            for backend in VIDEO_CAP_BACKENDS:
                video_capture = cv2.VideoCapture(idx, backend)
                if video_capture.isOpened():
                    ok, _ = video_capture.read()
                    if not ok:
                        video_capture.release()
                        video_capture = None
                        continue
                    vidstream_sources.append(str(idx))
                    break
                else:
                    video_capture.release()
                    video_capture = None
            if video_capture:
                video_capture.release()
        return vidstream_sources
    except Exception as e:
        print("Ошибка при поиске источников видео:", e)
        return []

def main():
    rectangles = []   # [{x, y}, ...]
    running = True    # обновление

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    cap = None

    if arg is None or (arg.isdigit() and arg not in detect_video_sources()):
        cap_ids = detect_video_sources()
        if not cap_ids:
            print("\nИсточники видео не найдены(")
            return
        print(f"\nИндексы доступных источников видео: {', '.join(cap_ids)}")
        selected_cap_id = None
        while selected_cap_id not in cap_ids:
            selected_cap_id = input("Введите индекс источника видео: ")
        cap = cv2.VideoCapture(int(selected_cap_id), VIDEO_CAP_BACKENDS[0])
    else:
        if arg.isdigit():
            cap = cv2.VideoCapture(int(arg), VIDEO_CAP_BACKENDS[0])

    if not cap or not cap.isOpened():
        print(f"Не удалось открыть источник видео")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    app = tk.Tk()
    imshow_widget = tk.Label(app)
    imshow_widget.pack()

    def add_rectangle(event):
        rectangles.append({"x": event.x, "y": event.y})

    def clear_rectangles(event=None):
        rectangles.clear()

    def on_key(event):
        k = event.keysym.lower()
        if k == 'q':
            on_close()
        elif k == 'c':
            clear_rectangles()

    def on_close():
        nonlocal running
        if not running:
            return
        running = False
        try:
            if cap:
                cap.release()
        finally:
            app.destroy()

    app.bind_all('<KeyPress>', on_key)
    imshow_widget.bind('<Button-1>', add_rectangle)
    app.protocol('WM_DELETE_WINDOW', on_close)

    def draw_rectangles(cv2image):
        for rect in rectangles:
            x, y = rect.get("x"), rect.get("y")
            x1 = min(x, x - RECT_SIZE // 2)
            y1 = min(y, y - RECT_SIZE // 2)
            x2 = x1 + RECT_SIZE
            y2 = y1 + RECT_SIZE
            cv2.rectangle(cv2image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        return cv2image

    def update_frame():
        if not running:
            return
        ret, frame = cap.read()
        if not ret:
            on_close()
            return

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2image = draw_rectangles(cv2image)

        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        imshow_widget.config(image=imgtk)
        imshow_widget.image = imgtk

        app.after(10, update_frame)

    app.after(0, update_frame)
    try:
        app.mainloop()
    finally:
        if cap:
            cap.release()

if __name__ == "__main__":
    main()
