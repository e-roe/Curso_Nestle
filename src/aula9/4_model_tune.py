from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Carrega o modelo (pode ser um pré-treinado ou personalizado)
model = YOLO("yolo11n.pt")

data_path = '../../datasets/Veiculos/dataset.yaml'
# Realiza o ajuste de hiperparâmetros
model.tune(
    data=data_path,   # caminho para o dataset
    epochs=5,             # número de épocas para cada experimento
    iterations=5,         # número de combinações de hiperparâmetros testadas
    optimizer="Adam",      # otimizador a ser usado (opcional)
    plots=True             # se True, gera gráficos comparativos
)