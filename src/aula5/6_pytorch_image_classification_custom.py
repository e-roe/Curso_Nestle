import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Dispositivo
device = torch.device("cpu")

# Configurações
data_dir = r"C:\Roe\Stepps\projs\Curso_Nestle\datasets\Animals"
batch_size = 32
num_epochs = 10
learning_rate = 0.0001
num_classes = len(os.listdir(f"{data_dir}/train"))

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Datasets e Loaders
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transforms)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Modelo MobileNetV2
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

model = model.to(device)

# Loss e Otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Listas para plot
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Treinamento
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        results = model(images)
        loss = criterion(results, labels)
        loss.backward()
        optimizer.step()

        a = loss.item()
        running_loss += loss.item()
        _, preds = torch.max(results, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validação
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            results = model(images) # Faz a inferência
            loss = criterion(results, labels)
            val_loss += loss.item()

            _, preds = torch.max(results, 1)
            correct += (preds == labels).sum().item() # Número de acertos
            total += labels.size(0)  # Primeira dimensão do tensor labels que é batch size

    val_loss /= len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Treino Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Plot dos gráficos
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Loss Treino')
plt.plot(epochs, val_losses, label='Loss Validação')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.title('Loss por Época')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Acurácia Treinamento')
plt.plot(epochs, val_accuracies, label='Acurácia Validação')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.title('Accuracy por Época')
plt.legend()

plt.tight_layout()
plt.show()
