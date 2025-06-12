import os
import cv2

os.makedirs('../../outputs/aula6', exist_ok=True)  # Cria a pasta de saída se não existir

largura = 640
altura = 480

# Carrega o modelo SSD pré-treinado e o arquivo de configuração
model_path = "../../models/aula6/frozen_inference_graph.pb"  # Caminho para o modelo
config_path = "../../models/aula6/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # Caminho para o arquivo de configuração
net = cv2.dnn_DetectionModel(model_path, config_path)  # Cria o modelo de detecção

# Carrega os rótulos das classes do dataset COCO
class_labels = []
with open("../../models/aula6/coco.names", "r") as f:  # Abre o arquivo com os nomes das classes
    class_labels = f.read().strip().split("\n")  # Lê e separa os nomes das classes

# Configura o modelo
net.setInputSize(320, 320)  # Define o tamanho da entrada do modelo
net.setInputScale(1.0 / 127.5)  # Escala os valores dos pixels
net.setInputMean((127.5, 127.5, 127.5))  # Define a média para normalização
net.setInputSwapRB(True)  # Troca os canais de cor de BGR para RGB

# Abre um vídeo ou feed da webcam (0)
cap = cv2.VideoCapture(r'../../inputs/aula6/deteccao_carros.mp4')  # Caminho para o vídeo (ou use 0 para webcam)

while cap.isOpened():  # Loop enquanto o vídeo estiver aberto
    ret, frame = cap.read()  # Lê um frame do vídeo
    if not ret:  # Se não conseguir ler o frame, encerra o loop
        break
    frame = cv2.resize(frame, (largura, altura))  # Redimensiona o frame para consistência


    # Realiza a detecção de objetos
    class_ids, confidences, boxes = net.detect(frame, confThreshold=0.5, nmsThreshold=0.4)
    # class_ids: IDs das classes detectadas
    # confidences: Confianças das detecções
    # boxes: Coordenadas das caixas delimitadoras

    # Desenha as detecções no frame
    if len(class_ids) > 0:  # Verifica se há objetos detectados
        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
            label = f"{class_labels[class_id - 1]}"  # Obtém o rótulo da classe detectada
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)  # Desenha a caixa delimitadora
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)  # Adiciona o rótulo acima da caixa

    # Exibe o frame com as detecções
    cv2.imshow("Deteccao de Objetos com SSD", frame)

    # Encerra o loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()  # Libera o vídeo
cv2.destroyAllWindows()  # Fecha todas as janelas abertas