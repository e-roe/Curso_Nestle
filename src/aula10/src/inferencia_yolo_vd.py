import os
import cv2
from ultralytics import YOLO

# Dicionário de cores para cada classe detectada
bbox_colors = {'capacete': (255, 100, 100), 'colete': (255, 0, 255), 'abafador': (10, 250, 100)}

if __name__ == "__main__":
    # Pasta com as imagens para inferência
    images_dir = r'C:\Roe\Stepps\projs\Curso_Nestle\src\aula10\data\test'

    # Carrega o modelo YOLO treinado
    model = YOLO(r'C:\Roe\Stepps\projs\Curso_Nestle\src\aula10\runs\detect\train\weights\best.pt')

    # Extensões de imagens suportadas
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    video_path = r'D:\Datasets\Videos_EPI\EPI_Geral\unseen\novos\n6\v1.mp4'
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        
        if frame is None:
            break
        
        # Realiza a inferência com o modelo YOLO
        results = model(frame)
        for result in results:
            for box in result.boxes:
                boxc = box.xyxy.cpu().numpy()[0]
                # Seleciona a cor da classe detectada
                color = bbox_colors[result.names[int(box.cls[0])]]
                # Desenha o retângulo de fundo (preto)
                cv2.rectangle(frame, (int(boxc[0]), int(boxc[1])), (int(boxc[2]), int(boxc[3])), (0, 0, 0), 8)
                # Desenha o retângulo colorido da classe
                cv2.rectangle(frame, (int(boxc[0]), int(boxc[1])), (int(boxc[2]), int(boxc[3])), color, 3)
                # Escreve o nome da classe (fundo preto)
                cv2.putText(frame, result.names[int(box.cls[0])], (int(boxc[0]), int(boxc[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 0), 2)
                # Escreve o nome da classe (texto branco)
                cv2.putText(frame, result.names[int(box.cls[0])], (int(boxc[0]) + 3, int(boxc[1]) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (255, 255, 255), 2)
        # Exibe a imagem com as detecções
        cv2.imshow('', frame)
        cv2.waitKey(1)
    cap.release()