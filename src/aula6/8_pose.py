import cv2
import mediapipe as mp


# Inicializa o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Abre o vídeo
video_path = "../../inputs/aula6/yoga.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converte o frame para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa o frame para detectar poses
    results = pose.process(rgb_frame)

    # Desenha as landmarks e conexões no frame original
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=6, circle_radius=6),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=6, circle_radius=6)
        )

    # Exibe o frame processado
    cv2.imshow("Estimativa de Pose", cv2.resize(frame, (0, 0), fx=0.25, fy=0.25))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()