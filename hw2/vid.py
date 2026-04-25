import cv2
import numpy as np

def detect_screen(image, min_contour_size=10000):
    """
    Функция для поиска экрана на изображении.
    Преобразует изображение в оттенки серого, затем применяет гауссовское размытие и выделяет границы
    с помощью фильтра Canny. Далее ищет контуры и выделяет прямоугольный контур,
    который соответствует экрану.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    screen_contour = None
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_size:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            screen_contour = approx.reshape(4, 2)
            break

    if screen_contour is None:
        print("Экран не найден.")
        return None

    return screen_contour

def is_shut_off_screen(image, screen_contour):
    """
    Функция для проверки, выключен ли экран (отсутствие объектов внутри экрана).
    Создается маска для области экрана, на которой далее проводится поиск контуров.
    Если внутри экрана нет значимых контуров, экран считается выключенным.
    """
    pts = order_screen_edges(screen_contour).astype(np.float32)

    width = int(max(
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3])
    ))
    height = int(max(
        np.linalg.norm(pts[0] - pts[3]),
        np.linalg.norm(pts[1] - pts[2])
    ))

    if width <= 0 or height <= 0:
        return False

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(pts, dst)
    screen = cv2.warpPerspective(image, H, (width, height))

    margin_x = int(width * 0.08)
    margin_y = int(height * 0.08)

    inner = screen[margin_y:height - margin_y, margin_x:width - margin_x]

    if inner.size == 0:
        return False

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    edges = cv2.Canny(blurred, 50, 150)
    edge_ratio = cv2.countNonZero(edges) / edges.size

    brightness_std = np.std(gray)
    _, bright_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    bright_ratio = cv2.countNonZero(bright_mask) / bright_mask.size

    return (
        edge_ratio < 0.025 and
        brightness_std < 50 and
        bright_ratio < 0.1
    )

def apply_perspective_transform(image, screen_contour, slide_image):
    """
    Функция для наложения слайда на экран. Использует гомографию для преобразования
    слайда и наложения его на экран с учетом перспективы.
    """
    if screen_contour is None:
        print("Экран не найден.")
        return image

    dst_points = order_screen_edges(screen_contour)

    width = int(np.linalg.norm(dst_points[0] - dst_points[1]) + np.linalg.norm(dst_points[3] - dst_points[2]))
    height = int(np.linalg.norm(dst_points[0] - dst_points[3]) + np.linalg.norm(dst_points[1] - dst_points[2]))

    slide_height, slide_width = slide_image.shape[:2]
    src_points = np.array([[0, 0], [slide_width, 0], [slide_width, slide_height], [0, slide_height]], dtype=np.float32)

    H, _ = cv2.findHomography(src_points, dst_points)

    warped_slide = cv2.warpPerspective(slide_image, H, (image.shape[1], image.shape[0]))

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_points.astype(int), 255)
    mask_inv = cv2.bitwise_not(mask)

    base_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    slide_fg = cv2.bitwise_and(warped_slide, warped_slide, mask=mask)

    result = cv2.add(base_bg, slide_fg)

    return result

def order_screen_edges(pts):
    """
    Упорядочивает вершины экрана (4 точки) так, чтобы они шли по порядку: верхняя левая, верхняя правая, нижняя правая, нижняя левая.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def process_video(input_video_path, slide_img_path, output_video_path):
    """
    Основная функция для обработки видео. Извлекает кадры из видео, находит экран на каждом кадре,
    проверяет его состояние и накладывает слайд на экран.
    """
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Ошибка при открытии видео.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    slide = cv2.imread(slide_img_path)
    if slide is None:
        print("Ошибка загрузки слайда.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        screeen = False
        min_contour_size, counter = 100000, 0
        
        result_frame = frame.copy()
        while not screeen and counter <= 10:
            screen_contour = detect_screen(frame, min_contour_size)
            if screen_contour is not None:
                if is_shut_off_screen(frame, screen_contour):
                    result_frame = apply_perspective_transform(frame, screen_contour, slide)
                    screeen = True
                    break
            min_contour_size += 20000
            counter += 1
        
        out.write(result_frame)

    cap.release()
    out.release()
    print(f"Обработка видео завершена. Результат сохранен в {output_video_path}")

process_video("input.MOV", "slide.jpg", "output.mp4")