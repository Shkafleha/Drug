# https://pysource.com/2021/01/28/object-tracking-with-opencv-and-python/

import cv2
from tracker import *
import time

#    Create tracker object
tracker = EuclideanDistTracker()

# cap = cv2.VideoCapture("2nd_box.mp4")
cap = cv2.VideoCapture("sample1.mp4")
# cap = cv2.VideoCapture("problem.mp4")
cap = cv2.VideoCapture("problem_up.mp4")
cap = cv2.VideoCapture("last20min.mp4")

id_ = 0

# Object detection from Stable camera
threshold = 100 # чем меньше тем чусвствительнее
object_detector = cv2.createBackgroundSubtractorMOG2(history=10000, varThreshold=threshold)

id = -1
while True:
    # считыаем кадр
    ret, frame = cap.read()

    # область детектора
    h = 20
    w = 360
    y1, x1 = 980, 1020
    # y1, x1 = 140, 1020
    y2 = y1 + h
    x2 = x1 + w
    # detector = frame[600: 650, 1100: 1350] # метал лента на конце нижней ленты
    detector = frame[y1: y2, x1: x2]
    # рисуем детектор
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=6)

    # 1. Object Detection
    # создаем маску в которой будет показываться только то, что движется
    # в нужной нам области детектора
    mask = object_detector.apply(detector)
    # mask это черный слой с белыми элементами
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # Находим контуры, RETR_EXTERNAL должен проверять только внешний контур
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        # рисуем квадратик у того что определили как объект
        if area > 300:
            cv2.drawContours(detector, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            # и передаем координаты в detections
            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
# ------------------------------

    if len(boxes_ids):
        print(f'boxes_ids {boxes_ids}')
        id_ += 1
        print(f'id {id_}')

        x, y, w, h, id = boxes_ids[0]
        cv2.putText(detector, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 14, (255, 255, 255), 2)
        cv2.rectangle(detector, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # elif len(boxes_ids)>1:
    #     print(boxes_ids)
    #     print('alaarm')
# ------------------------------

    # for box_id in boxes_ids:
    #     print(111, boxes_ids)
    #     print(222, box_id)
    #
    #     x, y, w, h, id = box_id
    #     cv2.putText(detector,  str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 14, (255, 255, 255), 2)
    #     cv2.rectangle(detector, (x, y), (x + w, y + h), (0, 255, 0), 3)
    # ------------------------------

    # сумма позиций  str(id+1)
    cv2.putText(frame, str(id+1), (1050, 700), cv2.FONT_HERSHEY_PLAIN, 14, (0, 255, 0), 7)
    # cv2.putText(frame, str(id_), (1050, 700), cv2.FONT_HERSHEY_PLAIN, 14, (0, 255, 0), 7)
    cv2.imshow("detector", detector)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(10)
    if key == 32:
        time.sleep(3)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()