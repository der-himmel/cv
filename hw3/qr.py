import cv2
import numpy as np

def order_box_edges(pts):
    """
    Упорядочивает вершины QR-кода (4 точки) так, чтобы они шли по часовой стрелке от верхней левой точки
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def decode_qr_code_cv2(img):
    detector = cv2.QRCodeDetector()
    data, bbox, straight_qrcode = detector.detectAndDecode(img)
    
    if data:
        if bbox is not None:
            for i in range(len(bbox[0])):
                pt1 = tuple(bbox[0][i].astype(int))
                pt2 = tuple(bbox[0][(i+1) % len(bbox[0])].astype(int))
                cv2.line(img, pt1, pt2, (255, 0, 0), 3) # Draw blue lines
    else:
        data = False           
    
    return img, data, straight_qrcode, bbox

def correct_perspective(img, src_points):
    src = np.float32(src_points)
        
    q_width = int(img.shape[0]/4)
    q_height = int(img.shape[1]/4)
    size = min(q_height, q_width)
    
    destination_points = [[0, 0], [size , 0], [size , size], [0, size]]
    dst = np.float32(destination_points)
    output_dimensions = (size, size)

    M = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(img, M, output_dimensions)
    return warped_image

def calculate_pov_degree(rectangle):
    """
    Рассчитывает угол поворота QR-кода
    """
    top = rectangle[1][0] - rectangle[0][0]
    bottom = rectangle[2][0] - rectangle[3][0] 
    left = rectangle[3][1] - rectangle[0][1]
    right = rectangle[2][1] - rectangle[1][1]

    degree_x = np.arccos(min(top,bottom)/max(top,bottom))
    degree_y = np.arccos(min(left,right)/max(left,right))

    return np.degrees(degree_x), np.degrees(degree_y)

def put_text_on_frame(frame, text):
    """
    Наложение текста на кадр
    """
    cv2.putText(frame, text, (25, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    return frame


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    degree_x_max, degree_y_max = 0.0, 0.0

    while True:
        ret, frame = cap.read()
        frame_mod = frame.copy()
        degree_x, degree_y = 0.0, 0.0
        img, data, straight_qrcode, bbox = decode_qr_code_cv2(frame)
        if data:
            ordered_points = order_box_edges(bbox[0])
            
            degree_x, degree_y = calculate_pov_degree(ordered_points)
            if degree_x > degree_x_max:
                degree_x_max = degree_x
            if degree_y > degree_y_max:
                degree_y_max = degree_y

            frame_mod = put_text_on_frame(frame_mod, f"x:{degree_x:.1f}, y:{degree_y:.1f} -- {data}")
            corrected_img = correct_perspective(img, bbox[0])
            try:
                if corrected_img is not None:
                    target_h = frame_mod.shape[0] // 4
                    target_w = frame_mod.shape[1] // 4

                    corrected_resized = cv2.resize(corrected_img, (target_w, target_h))

                    frame_mod[0:target_h, 0:target_w] = corrected_resized
            except Exception as e:
                print(e)
        cv2.imshow('QR detector', frame_mod)
        if cv2.waitKey(1) == ord('q'):
            print(f"max x:{degree_x_max:.2f}, y:{degree_y_max:.2f}")
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()