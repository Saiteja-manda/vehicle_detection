import cv2
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

video_path = r'C:\Users\saiteja_manda\Documents\SAITEJA\vehicle_detection\2053100-uhd_3840_2160_30fps.mp4'

def rescaleFrame(video_path, scale =0.25):
    width = int(video_path .shape[1]* scale)
    height = int(video_path.shape[0]* scale)

    dimensions =(width, height)

    return cv2.resize(video_path, dimensions, interpolation = cv2.INTER_AREA)

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()

    frame_resized = rescaleFrame(frame)

    
    results = model(frame_resized)

    image = results[0].plot()

    cv2.imshow('vehicle_detection', image)
    
    if cv2.waitKey(20) & 0xFF==ord('s'):
        break

cap.release()
cv2.destroyAllWindows()      