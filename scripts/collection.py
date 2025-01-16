import cv2
import sys

videoCapture = cv2.VideoCapture('./data/videos/video7.mp4')
    
if not videoCapture.isOpened():
    logging.error("Cannot open video file")
    sys.exit(-1)
   
i = 0
j = 0
while True:
    success, frame = videoCapture.read()
    if success:
        cv2.imshow('Display', frame)
        i += 1
        if i % 20 == 0:
            j += 1
            cv2.imwrite(f'./data/images/img_{j:04d}.jpg', frame)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

