import cv2

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
        data = "QR Code not detected or could not be decoded."            
    
    return img, data

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    
    while True:
        ret, frame = cap.read()
        img, data = decode_qr_code_cv2(frame)
        print(data)
        cv2.imshow('QR detector', img)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
        
if __name__ == "__main__":
    main()
    
# import cv2
# import numpy as np

# def correct_perspective(img, src_points, dst_points, output_size):
#     src = np.float32(src_points)
#     dst = np.float32(dst_points)
#     M = cv2.getPerspectiveTransform(src, dst)
#     warped_image = cv2.warpPerspective(img, M, output_size)
#     return warped_image
# source_points = [[20, 1], [540, 130], [570, 450], [20, 520]] # Example points, replace with your own
# destination_points = [[0, 0], [600, 0], [600, 600], [0, 600]] # Example, adjust as needed
# output_dimensions = (600, 600)

# img = cv2.imread('qr2_bad.jpg')
# corrected_img = correct_perspective(img, source_points, destination_points, output_dimensions)
# if corrected_img is not None:
#     cv2.imwrite('qr2_fixed.jpg', corrected_img)
#     cv2.imshow("Corrected Image", corrected_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
