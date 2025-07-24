from ultralytics import YOLO

# instalar: pip install tensorrt
# Carregue o modelo treinado (.pt)
model = YOLO('../runs/detect/train4/weights/best.pt')

# Exporte para ONNX
model.export(format='engine')