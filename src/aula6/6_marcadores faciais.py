# Importa as bibliotecas necessárias
# face_recognition para reconhecimento facial, cv2 para manipulação de vídeo/imagem,
# PIL para manipulação de imagens e numpy para arrays
import face_recognition
import cv2
from PIL import Image, ImageDraw
import numpy as np

# Define a largura e altura para redimensionar os frames do vídeo
largura = 1920 // 2
altura = 1080 // 2

# Abre o vídeo para reconhecimento facial
video_path = '../../inputs/aula6/faces.mp4'  # Substitua pelo caminho do seu vídeo
cap = cv2.VideoCapture(video_path)

# Loop para processar cada frame do vídeo
while cap.isOpened():
    ret, frame = cap.read()  # Lê um frame do vídeo
    if not ret:  # Se não houver mais frames, encerra o loop
        break
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Redimensiona o frame para metade do tamanho
    pil_image = Image.fromarray(frame)  # Converte o frame para uma imagem PIL
    d = ImageDraw.Draw(pil_image)  # Cria um objeto para desenhar na imagem

    # Converte o frame para RGB, pois face_recognition usa imagens RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta os marcadores faciais no frame
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    for face_landmarks in face_landmarks_list:
        # Imprime os pontos de cada característica facial detectada
        for facial_feature in face_landmarks.keys():
            print("O {} nesta face tem os seguintes pontos: {}".format(facial_feature,
                                                                            face_landmarks[facial_feature]))
        # Desenha linhas conectando os pontos de cada característica facial
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=3)

    # Converte a imagem PIL de volta para um array NumPy
    image = np.array(pil_image)

    # Exibe o frame com os marcadores faciais desenhados
    cv2.imshow("Marcadores Faciais", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Encerra o loop se a tecla 'q' for pressionada
        break

# Libera o vídeo e fecha todas as janelas abertas
cap.release()
cv2.destroyAllWindows()