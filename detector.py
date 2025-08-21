import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1040)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


object_labels = [
    'cell phone', 'microphone', 'remote', 'laptop',
    'keyboard', 'mouse', 'camera', 'glasses', 'tv', 'bottle'
]


animal_labels = ['cat', 'dog', 'bird', 'horse', 'cow', 'sheep']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

 
        if conf < 0.5:
            continue
        if label in object_labels:
            category = "OBJECT"
            color = (0, 255, 255)
        elif label in animal_labels:
            category = "ANIMAL"
            color = (255, 0, 0)
        else:
            continue 

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f'{category} ({label})', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Only Objects + Animals by Bradley Lenaiyarra", frame)

 
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
