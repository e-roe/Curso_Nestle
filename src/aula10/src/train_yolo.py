from ultralytics import YOLO
import torch
import os
import sys

if __name__ == '__main__':
    print(torch.cuda.is_available())
    model = YOLO('yolo11n.pt')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    data_path = './data/dataset.yaml'
    print(type(model))
    model.to(device)
    model.train(data=data_path, epochs=100, batch=4, plots=True, imgsz=640, save_period=-1, save_dir='./runs',
                lr0=0.001, optimizer='AdamW')
