import os
import cv2
from ultralytics import YOLO

# Dicionário de cores para cada classe detectada
bbox_colors = {'capacete': (255, 100, 100), 'colete': (255, 0, 255), 'abafador': (10, 250, 100)}

if __name__ == "__main__":
    # Pasta com as imagens para inferência
    images_dir = './data/test'

    # Carrega o modelo YOLO treinado
    model = YOLO('./runs/detect/train/weights/best.pt')

    # Extensões de imagens suportadas
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')

    # Percorre todas as imagens na pasta
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(img_exts):
            continue
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Realiza a inferência com o modelo YOLO
        results = model(img)
        for result in results:
            for box in result.boxes:
                boxc = box.xyxy.cpu().numpy()[0]
                # Seleciona a cor da classe detectada
                color = bbox_colors[result.names[int(box.cls[0])]]
                # Desenha o retângulo de fundo (preto)
                cv2.rectangle(img, (int(boxc[0]), int(boxc[1])), (int(boxc[2]), int(boxc[3])), (0, 0, 0), 8)
                # Desenha o retângulo colorido da classe
                cv2.rectangle(img, (int(boxc[0]), int(boxc[1])), (int(boxc[2]), int(boxc[3])), color, 3)
                # Escreve o nome da classe (fundo preto)
                cv2.putText(img, result.names[int(box.cls[0])], (int(boxc[0]), int(boxc[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 0), 2)
                # Escreve o nome da classe (texto branco)
                cv2.putText(img, result.names[int(box.cls[0])], (int(boxc[0]) + 3, int(boxc[1]) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (255, 255, 255), 2)
        # Exibe a imagem com as detecções
        cv2.imshow('', img)
        cv2.waitKey(0)