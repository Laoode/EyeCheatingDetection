import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Define constants
IMG_SIZE = (64, 56)
class_labels = ['center', 'left', 'right']
font_letter = cv2.FONT_HERSHEY_PLAIN

# Inisialisasi MediaPipe Face Landmarker dengan Tasks API
base_options = python.BaseOptions(model_asset_path='Models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# Load model arah mata
model = load_model('Models/eye_modelv3.h5')

# Fungsi untuk crop area mata dari gambar
def crop_eye(img, landmarks, eye_indices):
    h, w = img.shape[:2]
    eye_points = []
    for idx in eye_indices:
        landmark = landmarks[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        eye_points.append([x, y])
    eye_points = np.array(eye_points, dtype=np.int64)

    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    w_eye = (x2 - x1) * 1.2
    h_eye = w_eye * IMG_SIZE[1] / IMG_SIZE[0]

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    min_x, min_y = int(cx - w_eye / 2), int(cy - h_eye / 2)
    max_x, max_y = int(cx + w_eye / 2), int(cy + h_eye / 2)

    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = min(w, max_x), min(h, max_y)

    eye_img = img[min_y:max_y, min_x:max_x]
    eye_rect = [min_x, min_y, max_x, max_y]
    return eye_img, eye_rect

# Inisialisasi video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Tidak bisa buka webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frames_to_alert = 6
alert_frames = 0
alert_triggered = False

# Variabel untuk timestamp manual
frame_count = 0
TIMESTAMP_INCREMENT = 33  # 33 ms per frame (~30 FPS)

# Main loop
while cap.isOpened():
    ret, img = cap.read()
    if not ret or img is None:
        print("Error: Gagal ambil gambar dari webcam.")
        break

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pakai timestamp manual yang bertambah
    timestamp_ms = frame_count * TIMESTAMP_INCREMENT
    frame_count += 1

    # Konversi ke format MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if detection_result.face_landmarks:
        for landmarks in detection_result.face_landmarks:
            left_eye_indices = [33, 133, 159, 145]  # Mata kiri
            right_eye_indices = [362, 263, 386, 374]  # Mata kanan

            # Crop dan klasifikasi mata kiri
            eye_img_l, eye_rect_l = crop_eye(gray, landmarks, left_eye_indices)
            if eye_img_l.size > 0:
                eye_input_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
                eye_input_l = eye_input_l.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.0
                pred_l = model.predict(eye_input_l, verbose=0)
                gaze_l = class_labels[np.argmax(pred_l)]

                cv2.rectangle(img, (eye_rect_l[0], eye_rect_l[1]), (eye_rect_l[2], eye_rect_l[3]), (0, 255, 0), 2)
                cv2.putText(img, gaze_l, (eye_rect_l[0], eye_rect_l[1] - 10), font_letter, 1, (0, 255, 0), 2)

            # Crop dan klasifikasi mata kanan
            eye_img_r, eye_rect_r = crop_eye(gray, landmarks, right_eye_indices)
            if eye_img_r.size > 0:
                eye_input_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
                eye_input_r = eye_input_r.reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.0
                pred_r = model.predict(eye_input_r, verbose=0)
                gaze_r = class_labels[np.argmax(pred_r)]

                cv2.rectangle(img, (eye_rect_r[0], eye_rect_r[1]), (eye_rect_r[2], eye_rect_r[3]), (0, 255, 0), 2)
                cv2.putText(img, gaze_r, (eye_rect_r[0], eye_rect_r[1] - 10), font_letter, 1, (0, 255, 0), 2)

            # Cek gerakan mata mencurigakan
            if gaze_l in ['left', 'right'] or gaze_r in ['left', 'right']:
                alert_frames += 1
                if alert_frames >= frames_to_alert:
                    alert_triggered = True
            else:
                alert_frames = 0
                alert_triggered = False

            if alert_triggered:
                cv2.putText(img, "ALERT: Suspicious Eye Movement!", (10, 30), font_letter, 2, (0, 0, 255), 2)

    cv2.imshow('Eye Gaze Detection', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()