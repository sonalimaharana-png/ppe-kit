from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\Sonali Maharana\Desktop\dataset\runs\detect\train\weights\best.pt")  

url = "http://192.168.69.172:8080/video"
cap = cv2.VideoCapture(url)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        annotated = r.plot()  

    resized = cv2.resize(annotated, (640, 480))  # width=640, height=480

    cv2.imshow("PPE Detection", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
