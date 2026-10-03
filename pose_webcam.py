import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/pose_landmarker_full.task"


# Configurar MediaPipe Pose Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

landmarker = vision.PoseLandmarker.create_from_options(options)


# Conexiones principales del cuerpo
CONNECTIONS = [
    (11, 12),  # hombros

    (11, 13),
    (13, 15),  # brazo izquierdo

    (12, 14),
    (14, 16),  # brazo derecho

    (11, 23),
    (12, 24),
    (23, 24),  # torso

    (23, 25),
    (25, 27),  # pierna izquierda

    (24, 26),
    (26, 28),  # pierna derecha

    (27, 29),
    (29, 31),  # pie izquierdo

    (28, 30),
    (30, 32),  # pie derecho
]


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: No se ha podido abrir la cámara.")
    exit()


print("Cámara iniciada.")
print("Pulsa ESC para salir.")


start_time = time.monotonic()

while True:

    ret, frame = cap.read()

    if not ret:
        print("No se ha podido obtener un fotograma.")
        break

    # OpenCV trabaja en BGR; MediaPipe necesita RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int((time.monotonic() - start_time) * 1000)

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    height, width, _ = frame.shape

    # Si MediaPipe ha encontrado una persona
    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # Dibujar conexiones
        for start_idx, end_idx in CONNECTIONS:

            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            x1 = int(p1.x * width)
            y1 = int(p1.y * height)

            x2 = int(p2.x * width)
            y2 = int(p2.y * height)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

        # Dibujar los 33 puntos detectados
        for i, landmark in enumerate(landmarks):

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                str(i),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )

        cv2.putText(
            frame,
            "Persona detectada",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "Persona NO detectada",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("TFG - Deteccion de pose", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break


cap.release()
landmarker.close()
cv2.destroyAllWindows()
