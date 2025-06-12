import os
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Define o caminho raiz onde estão as imagens
root_path = '../../inputs/aula6/imagens'
imagens = os.listdir(root_path)  # Lista todos os arquivos no diretório

# Carrega o modelo pré-treinado ResNet-18
model = models.resnet18(pretrained=True)
model.eval()  # Coloca o modelo em modo de avaliação (não treinar)

# Define as transformações para as imagens
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensiona a imagem para 224x224
    transforms.ToTensor(),  # Converte a imagem para tensor PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza a imagem
])

# Itera sobre todas as imagens no diretório
for imagem in imagens:
    # Abre a imagem e converte para RGB
    image = Image.open(os.path.join(root_path, imagem)).convert("RGB")
    # Aplica as transformações e adiciona a dimensão do batch
    input_tensor = transform(image).unsqueeze(0)

    # Realiza a predição
    with torch.no_grad():  # Desativa o cálculo de gradientes
        outputs = model(input_tensor)  # Passa a imagem pelo modelo
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Aplica softmax para obter probabilidades

    # Carrega os rótulos das classes do ImageNet
    with open("../../models/aula6/imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]  # Lê os rótulos linha por linha

    # Obtém as 3 classes com maior probabilidade
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    for i in range(3):
        print(f"{i + 1}: {labels[top3_idx[i]]} ({top3_prob[i].item():.2f})")  # Exibe o rótulo e a probabilidade

    # Converte a imagem para um array NumPy e altera para o formato BGR (usado pelo OpenCV)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Adiciona o rótulo da classe com maior probabilidade na imagem
    cv2.putText(image, f'{labels[top3_idx[0]]}', (11, 21), cv2.FONT_HERSHEY_SIMPLEX, .64,
                (0, 0, 0),1, lineType=cv2.LINE_AA)

    # Exibe a imagem com as anotações
    cv2.imshow('Imagem', image)
    cv2.waitKey(0)  # Aguarda uma tecla ser pressionada antes de fechar a janela
