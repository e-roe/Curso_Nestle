from ultralytics import YOLO
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    print(torch.cuda.is_available())
    model = YOLO('yolo11n.pt')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_path = '../../datasets/Veiculos/dataset.yaml'
    print(type(model))
    model.to(device)
    model.train(data=data_path, epochs=10, batch=4, plots=True, imgsz=640, save_period=-1, save_dir='./runs',
                lr0=0.001, optimizer='AdamW', cos_lr=True, lrf=0.01)
