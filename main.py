import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

model = load_model('gender_age')
labels = ['Man', 'Woman']

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.process(rgb_frame)

    if faces.detections:
        for face in faces.detections:
            bboxC = face.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            w = int(bboxC.width * w)
            h = int(bboxC.height * h)

            mp_drawing.draw_detection(frame, face)

            face_img = frame[max(0, y):min(y + h, frame.shape[0]), max(0, x):min(x + w, frame.shape[1])]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img_resized = cv2.resize(face_img, (200, 200))
            face_img_normalized = face_img_resized / 255.0
            face_img_reshaped = np.expand_dims(face_img_normalized, axis=(0, -1))  # Add batch and channel dimensions

            prediction = model.predict(face_img_reshaped)
            # print(prediction)
            gender_label = labels[0 if prediction[1][0] > 0.8 else 1]
            age_label = prediction[0][0].astype(int)
            
            confidence = float(prediction[1][0]) if gender_label == 'Man' else float(1 - prediction[1][0])

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{gender_label} ({confidence:.2f} Age : {age_label})", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


    cv2.imshow('Gender Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
