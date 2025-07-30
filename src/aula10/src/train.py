import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import time

# Classe de transformação para converter imagens PIL em tensores
class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)  # Converte imagem PIL para tensor
        return image, target

# Função para criar o dataset COCO
def get_coco_dataset(img_dir, ann_file):
    return CocoDetection(
        root=img_dir,
        annFile=ann_file,
        transforms=CocoTransform()
    )

# Carrega os datasets de treino e validação
train_dataset = get_coco_dataset(
    img_dir="../data/train",
    ann_file="../data/train/annotations.json"
)

val_dataset = get_coco_dataset(
    img_dir="../data/val",
    ann_file="../data/val/annotations.json"
)

# Função para carregar o modelo Faster R-CNN com backbone ResNet-50
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # Modelo pré-treinado
    in_features = model.roi_heads.box_predictor.cls_score.in_features  # Número de entradas do classificador
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # Substitui a cabeça do classificador
    return model

# Função para treinar o modelo por uma época
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets in data_loader:
        images = [img.to(device) for img in images]  # Move imagens para o dispositivo

        # Processa e valida os targets
        processed_targets = []
        valid_images = []
        for i, target in enumerate(targets):
            boxes = []
            labels = []
            for obj in target:
                bbox = obj["bbox"]  # Formato: [x, y, largura, altura]
                x, y, w, h = bbox
                if w > 0 and h > 0:  # Garante que largura e altura sejam positivas
                    boxes.append([x, y, x + w, y + h])  # Converte para [x_min, y_min, x_max, y_max]
                    labels.append(obj["category_id"])
            if boxes:  # Só adiciona se houver caixas válidas
                processed_target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                    "labels": torch.tensor(labels, dtype=torch.int64).to(device),
                }
                processed_targets.append(processed_target)
                valid_images.append(images[i])
        if not processed_targets:  # Pula se não houver targets válidos
            continue
        images = valid_images  # Garante alinhamento entre imagens e targets

        # Passagem direta pelo modelo
        loss_dict = model(images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Época [{epoch}] - Loss: {losses.item():.4f}")

# DataLoader para treino e validação
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Inicializa o modelo com o número de classes
num_classes = 4 # Fundo + capacete, colete, abafador
model = get_model(num_classes)

# Move o modelo para GPU se disponível
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define o otimizador e o scheduler de taxa de aprendizado
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Loop de treinamento principal
num_epochs = 25
for epoch in range(num_epochs):
    start = time.time()
    train_one_epoch(model, optimizer, train_loader, device, epoch)
    print(f'Tempo da época {epoch + 1}: {time.time() - start:.2f} segundos')
    # Salva o estado do modelo após cada época
    model_path = f"../models/fasterrcnn_resnet50_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo salvo: {model_path}")