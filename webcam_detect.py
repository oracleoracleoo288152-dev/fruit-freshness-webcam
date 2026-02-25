from ultralytics import YOLO
import cv2
import os


def main():
    model = YOLO("best1.pt")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results = model.predict(frame, conf=0.5, verbose=False)
        pred = results[0]

        if pred.boxes is not None:
            for box in pred.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                label = pred.names[cls_id]
                text = f"{label} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2
                )

        # Try to display the frame; in headless environments (Streamlit cloud)
        # OpenCV GUI calls will raise an error, so catch and skip them.
        try:
            cv2.imshow("Rotten or Not", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            # Headless: skip imshow/waitKey and continue capturing
            pass

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


if __name__ == "__main__":
    main()