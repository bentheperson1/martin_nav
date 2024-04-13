from ultralytics import YOLO
import cv2, math, threading
from playsound import playsound

window_width = 1280
window_height = 960

center_x = window_width / 2
center_y = window_height / 2

cap = cv2.VideoCapture(0)
cap.set(3, window_width)
cap.set(4, window_height)

area_threshold = 150000

model = YOLO("yolo-Weights/yolov8n.pt")

sound_play_timer = 0
sound_play_timer_set = 15

left_sound = "left.wav"
right_sound = "blip.wav"

interval_size = 320

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

def play_left_beep():
    playsound(left_sound)

def play_right_beep():
    playsound(right_sound)

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)

    results = model(img, stream = True)

    sound_play_timer -= 1

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            area = (x2 - x1) * (y2 - y1)  

            if area >= area_threshold:
                x_pos = round((x2 + x1) / 2)
                y_pos = round((y2 + y1) / 2)

                if x_pos < center_x + interval_size and x_pos > center_x - interval_size:
                    if x_pos > center_x:
                        txt = "Move Left"

                        if sound_play_timer <= 0:
                            sound_play_timer = sound_play_timer_set

                            audio_thread = threading.Thread(target=play_left_beep)
                            audio_thread.start()
                    else:
                        txt = "Move Right"

                        if sound_play_timer <= 0:
                            sound_play_timer = sound_play_timer_set

                            audio_thread = threading.Thread(target=play_right_beep)
                            audio_thread.start()

                    cv2.circle(img, (x_pos, y_pos), 3, (255, 255, 255), -1)
                    cv2.putText(img, txt, [32, 32], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (160, 255, 255), 3)
                    cv2.putText(img, classNames[int(box.cls[0])], [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    r_line_x = round(center_x + interval_size)
    l_line_x = round(center_x - interval_size)
    cv2.line(img, (r_line_x, 0), (r_line_x, window_height), (0,255,0),3)
    cv2.line(img, (l_line_x, 0), (l_line_x, window_height), (0,255,0),3)

    cv2.imshow('Navigation', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()