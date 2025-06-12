import cv2
from ultralytics import YOLO
import os

# Define uma variável de ambiente para evitar conflitos com a biblioteca OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Carrega o modelo YOLO11
model = YOLO("../models/aula6/yolo11n.pt")

# Abre o arquivo de vídeo
video_path = "../../inputs/aula6/walking.mp4"
cap = cv2.VideoCapture(video_path)

# Dicionário para armazenar o histórico de rastreamento
track_history = {}

# Loop para processar os frames do vídeo
while cap.isOpened():
    success, frame = cap.read()  # Lê um frame do vídeo

    if success:
        # Executa o rastreamento YOLO11 no frame
        results = model.track(frame, persist=True)

        # Itera sobre os objetos detectados
        for track in results[0].boxes:
            obj_id = int(track.id)  # ID único do objeto rastreado
            x1, y1, x2, y2 = map(int, track.xyxy[0])  # Coordenadas da caixa delimitadora
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Centro da caixa delimitadora

            # Adiciona a posição atual ao histórico de rastreamento
            if obj_id not in track_history:
                track_history[obj_id] = []
            track_history[obj_id].append((cx, cy))

            # Desenha a caixa delimitadora
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

            # Desenha as trilhas dos objetos rastreados
            for obj_id, points in track_history.items():
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    # Calcula a cor com base na idade do ponto
                    alpha = i / len(points)  # Normaliza alpha para o intervalo [0, 1]
                    color = (255, int(255 * (1 - alpha)), int(255 * (1 - alpha)))  # Azul desbota para branco
                    cv2.line(frame, points[i - 1], points[i], color, 2)

        # Exibe o frame processado
        cv2.imshow("YOLO11 Tracking", frame)

        # Sai do loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Libera os recursos de vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()