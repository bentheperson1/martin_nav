from ultralytics import YOLO
import cv2, math, threading
from playsound import playsound
import pygame
pygame.init()

window_width = 1280
window_height = 960

center_x = window_width / 2
center_y = window_height / 2

cap = cv2.VideoCapture(0)
cap.set(3, window_width)
cap.set(4, window_height)

area_threshold = 100000

model = YOLO("models/yolov8n.pt")

sound_play_timer = 0
sound_play_timer_set = 20

left_sound = "assets/left.wav"
right_sound = "assets/right.wav"
stop_sound = "assets/stop.wav"

interval_size = 350

sound_interval_max = 20

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

alive_classes = ["person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]

def play_sound(sound, pan=[1.0,1.0]):
    sound = pygame.mixer.Sound(sound)
    channel = pygame.mixer.find_channel()
    channel.set_volume(pan[0], pan[1])
    channel.play(sound)

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)

    results = model(img, stream = True)

    sound_play_timer -= 1

    for r in results:
        boxes = r.boxes

        for index, box in enumerate(boxes):
            if index == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                area = (x2 - x1) * (y2 - y1)  

                if area >= area_threshold:
                    sound_play_timer_set = round(((area * 10000) / area ** 1.5))

                    x_pos = round((x2 + x1) / 2)
                    y_pos = round((y2 + y1) / 2)

                    if x_pos < center_x + interval_size and x_pos > center_x - interval_size:
                        if x_pos > center_x:
                            txt = "Move Left"

                            if sound_play_timer <= 0:
                                sound_play_timer = sound_play_timer_set

                                play_sound(left_sound, [1.0, 0])
                        else:
                            txt = "Move Right"

                            if sound_play_timer <= 0:
                                sound_play_timer = sound_play_timer_set

                                play_sound(right_sound, [0, 1.0])

                        cv2.circle(img, (x_pos, y_pos), 3, (255, 255, 255), -1)
                        cv2.putText(img, txt, [32, 32], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (160, 255, 255), 3)
                        cv2.putText(img, classNames[int(box.cls[0])], [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            else:
                if sound_play_timer <= 0:
                    sound_play_timer = sound_play_timer_set

                    play_sound(stop_sound)
    
    r_line_x = round(center_x + interval_size)
    l_line_x = round(center_x - interval_size)
    cv2.line(img, (r_line_x, 0), (r_line_x, window_height), (0,255,0),3)
    cv2.line(img, (l_line_x, 0), (l_line_x, window_height), (0,255,0),3)

    cv2.imshow('Navigation', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()