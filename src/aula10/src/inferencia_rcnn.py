import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import numpy as np

# Dicionário que define a cor de cada classe para desenhar as caixas delimitadoras
bbox_colors = {'Capacete': (255, 100, 100), 'Colete': (255, 0, 255), 'Abafador': (10, 250, 100)}

# Dicionário de classes COCO usadas no modelo
COCO_CLASSES = {0: "Background", 1: "Capacete", 2: "Colete", 3: "Abafador"}
num_classes = len(COCO_CLASSES)  # Número de classes

# Função para carregar o modelo Faster R-CNN com backbone ResNet-50
def get_model(num_classes):
    # Carrega o modelo pré-treinado
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Obtém o número de entradas do classificador
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Substitui a cabeça do classificador pelo número de classes desejado
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Função para preparar a imagem para inferência
def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Abre a imagem
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Converte para tensor e adiciona dimensão de batch
    return image_tensor.to(device)

# Função para obter o nome da classe a partir do id
def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")

# Função para desenhar as caixas delimitadoras usando OpenCV
def draw_boxes_cv(image_path, prediction, window_name="Prediction", threshold=0.5):
    image = cv2.imread(image_path)  # Lê a imagem
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Converte para BGR

    boxes = prediction[0]['boxes'].cpu().numpy()  # Caixas previstas
    labels = prediction[0]['labels'].cpu().numpy()  # Labels previstos
    scores = prediction[0]['scores'].cpu().numpy()  # Scores previstos

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:  # Só desenha se a confiança for maior que o limiar
            x_min, y_min, x_max, y_max = map(int, box)
            class_name = get_class_name(label)
            cor = bbox_colors[class_name]
            cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), cor, 2)  # Desenha o retângulo
            cv2.putText(image_cv, f"{class_name} ({score:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)  # Escreve o nome da classe e score

    # Exibe a imagem com as detecções
    cv2.imshow(window_name, cv2.cvtColor(cv2.resize(image_cv, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Função para desenhar as caixas usando Matplotlib
def draw_boxes(image, prediction, fig_size=(10, 10)):
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    threshold = 0.5  # Limiar de confiança

    plt.figure(figsize=fig_size)  # Define o tamanho da figura

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = box
            class_name = get_class_name(label)
            plt.imshow(image)
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              linewidth=2, edgecolor='r', facecolor='none'))
            plt.text(x_min, y_min, f"{class_name} ({score:.2f})", color='r')

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Define o dispositivo (GPU ou CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Carrega o modelo treinado
    model = get_model(num_classes)
    model.load_state_dict(torch.load("../models/fasterrcnn_resnet50_epoch_20.pth"))
    model.to(device)
    model.eval()  # Coloca o modelo em modo de avaliação

    # Pasta com imagens para inferência
    path = "../data/test"
    images = os.listdir(path)
    for image in images:
        image_path = os.path.join(path, image)
        image_tensor = prepare_image(image_path)

        with torch.no_grad():  # Desabilita o cálculo de gradientes
            prediction = model(image_tensor)  # Realiza a inferência

            # Exibe os resultados da inferência
            #draw_boxes(Image.open(image_path), prediction, fig_size=(12, 10))
            draw_boxes_cv(image_path, prediction)