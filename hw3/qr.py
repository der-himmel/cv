import cv2
import numpy as np

def order_box_edges(pts):
    """
    Упорядочивает вершины QR-кода (4 точки) так, чтобы они шли по часовой стрелек от верхней левой точки
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
        # Optionally, draw a bounding box (bbox contains corner coordinates)
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
        
    half_width = int(img.shape[0]/2)
    half_height = int(img.shape[1]/2)
    size = min(half_height, half_width)
    
    destination_points = [[0, 0], [size , 0], [size , size], [0, size]]
    dst = np.float32(destination_points)
    output_dimensions = (size, size)
    shape = [size, size]

    M = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(img, M, output_dimensions)
    return warped_image, shape

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    
    while True:
        ret, frame = cap.read()
        bbox = []
        img, data, straight_qrcode, bbox = decode_qr_code_cv2(frame)
        if data:
            width = img.shape[0]
            height = img.shape[0]
            print(width, height)
            ordered_points = order_box_edges(bbox[0])
            corrected_img, corrected_img_shape = correct_perspective(img, bbox[0])
            if corrected_img_shape:
                img_copy = img.copy()
                overlay_coords = [int(width - corrected_img_shape[0]/4), int(height- + corrected_img_shape[1]/4)]
                print(overlay_coords)
                img_copy[overlay_coords[0],overlay_coords[1],:] = corrected_img[overlay_coords[0],overlay_coords[1],:]
        cv2.imshow('QR detector', img)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
    
    # img = cv2.imread('qr3_bad.jpg')
    # img, data, straight_qrcode, bbox = decode_qr_code_cv2(img)
    # if len(bbox[0])>0:
    #     ordered_points = order_box_edges(bbox[0])
    #     corrected_img = correct_perspective(img, bbox[0])
    #     if corrected_img is not None:
    #     # if img is not None:
    #         cv2.imwrite('qr2_fixed.jpg', corrected_img)
    #         cv2.imshow("Corrected Image", corrected_img)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()