import random
import pyrealsense2 as rs
import cv2 as cv2
import numpy as np
from ultralytics import YOLO


def draw_bbox(image, bbox, label, color=(0, 255, 0), thickness=2):
    '''
    Функция рисует один bounding box на изображении
    '''
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Преобразование координат в целые числа
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_size = cv2.getTextSize(label, font, 0.5, thickness)[0]
    cv2.rectangle(image, (x1, y1), (x1 + label_size[0], y1 - label_size[1] - 5), color, -1)
    cv2.putText(image, label, (x1, y1 - 5), font, 0.5, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
    return image

model = YOLO("yolov8n.pt")
names = ['rgb_output.avi','depth_output.avi',]
window_titles = ['rgb_output','depth_output',]
cap = [cv2.VideoCapture(i) for i in names]
frames = [None] * len(names)
boxes = []
clss = []
ret = [None] * len(names)
while True:
    for i,c in enumerate(cap):
        if c is not None:
            ret[i], frames[i] = c.read()
    for i,f in enumerate(frames):
        if ret[i] is True:
            if i == 0:
                results = model.predict(f)
                for result in results:
                        if result:    
                            boxes = result.boxes.xyxy.to('cpu').tolist()
                            clss = result.boxes.cls.to('cpu').tolist() 
            else:
                for k, obj in enumerate(clss):
                            label = model.names[obj]
                            print()
                            f = draw_bbox(f, boxes[k], label, color=(0, 255, 0), thickness=2)
            cv2.imshow(window_titles[i], f)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
for c in cap:
    if c is not None:
        c.release()
cv2.destroyAllWindows()





    