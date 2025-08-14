import cv2
import numpy as np
from ultralytics import YOLO


# Função para calcular o tamanho da fonte de acordo com a resolução do frame
def get_font_scale(frame, base_scale=1.5):
    height, width = frame.shape[:2]
    scale_factor = min(width, height) / 1080  # Escala relativa ao 1080p
    return base_scale * scale_factor


# Função principal para inferência em vídeo
def video_inference(video_path, out_path, model, bbox_colors):
    out = None
    conf_th = 0.0  # limiar de confiança
    cap = cv2.VideoCapture(video_path)  # Abre o vídeo
    while True:
        ret, frame = cap.read()  # Lê um frame
        if not ret:
            break  # Sai do loop se não houver mais frames

        # Inicializa o VideoWriter na primeira iteração
        if out is None:
            fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
            out = cv2.VideoWriter(out_path, fourcc, 20, (frame.shape[1], frame.shape[0]))

        scale = get_font_scale(frame)  # Calcula o tamanho da fonte
        results = model(frame, save=False, verbose=False)  # Faz a inferência
        for result in results:
            for box in result.boxes:
                # Verifica se a classe está nas cores definidas
                if result.names[int(box.cls[0])] not in bbox_colors:
                    continue
                boxc = box.xyxy.cpu().numpy()[0]  # Coordenadas da caixa
                confidence = box.conf.cpu().numpy()[0]  # Confiança da detecção
                if confidence > conf_th:
                    color = bbox_colors[result.names[int(box.cls[0])]]
                    # Desenha o retângulo (preto para borda grossa)
                    cv2.rectangle(frame, (int(boxc[0]), int(boxc[1])), (int(boxc[2]), int(boxc[3])), (0, 0, 0), 8)
                    # Desenha o retângulo colorido
                    cv2.rectangle(frame, (int(boxc[0]), int(boxc[1])), (int(boxc[2]), int(boxc[3])), color, 3)
                    # Escreve o texto com sombra preta
                    cv2.putText(frame, f"{result.names[int(box.cls[0])]} ({confidence:.2f})",
                                (int(boxc[0]), int(boxc[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)
                    # Escreve o texto branco por cima
                    cv2.putText(frame, f"{result.names[int(box.cls[0])]} ({confidence:.2f})",
                                (int(boxc[0]) + 4, int(boxc[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        # Mostra o frame redimensionado na tela
        cv2.imshow('', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1)
        out.write(frame)  # Salva o frame no vídeo de saída
    cap.release()
    out.release()


if __name__ == '__main__':
    classes = ['helmet', 'vest', 'person', 'ear']  # Lista de classes
    colors = {}
    # Carrega o modelo treinado
    model = YOLO(r'seu_cam\models\best.pt')
    video_path = r'seu_cam\inputs\video_n213.mp4'
    # Gera cores aleatórias para cada classe
    for cls in classes:
        colors[cls] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

    out_path = r'seu_cam\outputs\inference.mp4'
    video_inference(video_path, out_path, model, colors)