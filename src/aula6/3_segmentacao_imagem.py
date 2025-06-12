import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Carrega o modelo pré-treinado DeepLabV3 com ResNet-101
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()  # Coloca o modelo em modo de avaliação (não treinamento)

# Define as transformações para a imagem
transform = transforms.Compose([
    transforms.Resize((520, 520)),  # Redimensiona a imagem para 520x520
    transforms.ToTensor(),  # Converte a imagem para um tensor PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza a imagem
])

# Carrega e pré-processa a imagem
image_path = "../../inputs/aula6/galinhas.jpg"  # Substitua pelo caminho da sua imagem
try:
    image = Image.open(image_path).convert("RGB")  # Abre a imagem e converte para RGB
except FileNotFoundError:
    raise FileNotFoundError(f"Imagem não encontrada no caminho {image_path}")

input_tensor = transform(image).unsqueeze(0)  # Adiciona a dimensão do batch

# Realiza a segmentação
with torch.no_grad():  # Desativa o cálculo de gradientes
    output = model(input_tensor)["out"][0]  # Obtém o tensor de saída
    output_predictions = output.argmax(0)  # Obtém a classe com maior pontuação para cada pixel

# Converte o mapa de segmentação para uma imagem colorida
segmentation_map = output_predictions.byte().cpu().numpy()  # Converte para NumPy
segmentation_colored = cv2.applyColorMap((segmentation_map * 15).astype(np.uint8), cv2.COLORMAP_JET)  # Aplica um mapa de cores

# Sobrepõe o mapa de segmentação na imagem original
image_np = np.array(image)  # Converte a imagem original para um array NumPy

# Redimensiona o mapa de segmentação para combinar com as dimensões da imagem original
segmentation_colored = cv2.resize(segmentation_colored, (image_np.shape[1], image_np.shape[0]))

# Garante que ambos os arrays tenham o mesmo tipo de dado
segmentation_colored = segmentation_colored.astype(image_np.dtype)

# Cria a sobreposição do mapa de segmentação na imagem original
segmentation_overlay = cv2.addWeighted(image_np, 0.5, segmentation_colored, 0.5, 0)

# Exibe o resultado
cv2.imshow("Segmentacao", segmentation_overlay)  # Mostra a imagem segmentada
cv2.waitKey(0)  # Aguarda uma tecla ser pressionada para fechar a janela
cv2.destroyAllWindows()  # Fecha todas as janelas abertas