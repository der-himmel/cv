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
        if cv2.contourArea(contour) < min_contour_size:  # Пропускаем маленькие контуры
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:  # Ищем прямоугольник
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
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [screen_contour], -1, 255, thickness=cv2.FILLED)

    screen_area = cv2.bitwise_and(image, image, mask=mask)

    gray_screen = cv2.cvtColor(screen_area, cv2.COLOR_BGR2GRAY)
    edges_in_screen = cv2.Canny(gray_screen, 50, 200)

    edge_count = cv2.countNonZero(edges_in_screen)

    if edge_count <= cv2.contourArea(screen_contour) * 0.025:
        return True
    else:
        return False

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

    print(f"Ширина экрана: {width}, высота: {height}")

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

def process_img(base_img_path, slide_img_path, output_path):
    """
    Основная функция для обработки изображений
    """
    base = cv2.imread(base_img_path)
    slide = cv2.imread(slide_img_path)

    if base is None or slide is None:
        print("Ошибка загрузки изображений.")
        return

    screeen = False # Нашли ли экран
    min_contour_size = 10000

    while not screeen:
        screen_contour = detect_screen(base, min_contour_size)

        if screen_contour is not None:
            if is_shut_off_screen(base, screen_contour):
                screeen = True
        else:
            print("Экран не найден.")
        
        min_contour_size += 10000   # Ищем другой экран большего размера

    result_image = apply_perspective_transform(base, screen_contour, slide)
    cv2.imwrite(output_path, result_image)

process_img("tv.jpg", "slide.jpg", "result.jpg")
process_img("tv_perspective.jpg", "slide.jpg", "result_perspective.jpg")