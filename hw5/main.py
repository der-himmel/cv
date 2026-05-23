import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

VIDEO_PATH = "2.mov"          # путь к видео
MODEL_PATH = "yolo11n.pt"     # YOLO модель
CONF_THRESHOLD = 0.5          # confidence threshold
MAX_AGE = 120                 # срок потери трекинга (для DeepSORT)
OUTPUT_PATH = "output_counted.mp4"
SHOW_WINDOW = True
COOLDOWN_FRAMES = 35       # кулдаун после пересечения линии

line_points = []
line_ready = False

def mouse_callback(event, x, y, flags, param):
    global line_points, line_ready
    if event == cv2.EVENT_LBUTTONDOWN and len(line_points) < 2:
        line_points.append((x, y))
        if len(line_points) == 2:
            line_ready = True

def get_line_side(point, line_start, line_end):
    """Возвращает знак: с какой стороны линии находится точка"""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    return (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)

def bbox_center(bbox):
    """Возвращает центр bbox"""
    x1, y1, x2, y2 = bbox
    return (int((x1+x2)/2), int((y1+y2)/2))

def draw_counting_line(frame, line_start, line_end):
    cv2.line(frame, line_start, line_end, (0, 255, 255), 3)
    cv2.circle(frame, line_start, 5, (0, 255, 255), -1)
    cv2.circle(frame, line_end, 5, (0, 255, 255), -1)

def select_line(first_frame):
    global line_points, line_ready
    line_points = []
    line_ready = False

    cv2.namedWindow("Select line")
    cv2.setMouseCallback("Select line", mouse_callback)

    while True:
        temp = first_frame.copy()
        for pt in line_points:
            cv2.circle(temp, pt, 5, (0,0,255), -1)
        if len(line_points) == 2:
            cv2.line(temp, line_points[0], line_points[1], (0,255,255), 2)

        cv2.putText(temp, "Set 2 points, ENTER - continue, R - reset, ESC - quit",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Select line", temp)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and line_ready:
            break
        elif key == ord('r'):
            line_points = []
            line_ready = False
        elif key == 27:
            cv2.destroyAllWindows()
            return None, None

    cv2.destroyWindow("Select line")
    return line_points[0], line_points[1]

def is_point_on_segment(point, seg_start, seg_end, tol=5):
    """
    Проверяет, находится ли точка рядом с отрезком seg_start-seg_end
    tol — допустимое расстояние до линии в пикселях
    """
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end

    if x1 == x2 and y1 == y2:
        return np.hypot(px - x1, py - y1) <= tol

    dx = x2 - x1
    dy = y2 - y1

    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)

    if t < 0.0 or t > 1.0:
        return False

    nearest_x = x1 + t*dx
    nearest_y = y1 + t*dy

    dist = np.hypot(px - nearest_x, py - nearest_y)
    return dist <= tol

def main():
    model = YOLO(MODEL_PATH)
    tracker = DeepSort(max_age=MAX_AGE, n_init=2, max_cosine_distance=0.3)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка: не удалось прочитать первый кадр")
        return

    line_start, line_end = select_line(first_frame)
    if line_start is None or line_end is None:
        cap.release()
        return

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    track_last_side = {}
    track_last_cross_frame = {}  
    counted_ids = set()
    count_in, count_out = 0, 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        annotated = frame.copy()
        draw_counting_line(annotated, line_start, line_end)

        detections = model(frame, verbose=False)[0]
        results = []
        for box in detections.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            if int(cls) != 0 or conf < CONF_THRESHOLD:
                continue
            results.append([[int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)], float(conf), int(cls)])

        tracks = tracker.update_tracks(results, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            center = bbox_center([x1, y1, x2, y2])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(annotated, center, 4, (0,0,255), -1)
            cv2.putText(annotated, f"ID {track_id}", (x1, max(y1-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            current_side = get_line_side(center, line_start, line_end)
            last_cross = track_last_cross_frame.get(track_id, -COOLDOWN_FRAMES-1)

            if track_id not in track_last_side:
                track_last_side[track_id] = current_side
            else:
                prev_side = track_last_side[track_id]
                crossed = (prev_side < 0 and current_side > 0) or (prev_side > 0 and current_side < 0)
                if crossed and is_point_on_segment(center, line_start, line_end, tol=25) and (frame_idx - last_cross) > COOLDOWN_FRAMES:
                    counted_ids.add(track_id)
                    track_last_cross_frame[track_id] = frame_idx
                    if prev_side < 0 and current_side > 0:
                        count_in += 1
                    else:
                        count_out += 1

                track_last_side[track_id] = current_side

        cv2.putText(annotated, f"IN: {count_in}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
        cv2.putText(annotated, f"OUT: {count_out}", (20,85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        cv2.putText(annotated, f"TOTAL UNIQUE: {len(counted_ids)}", (20,130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        if SHOW_WINDOW:
            cv2.imshow("People Counting", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        writer.write(annotated)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"IN:{count_in}\nOUT:{count_out}\nUNIQUE:{len(counted_ids)}")

if __name__ == "__main__":
    main()