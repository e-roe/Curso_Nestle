from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Verifica se há GPU disponível e imprime o resultado
    print(torch.cuda.is_available())

    # Carrega o modelo YOLO a partir do arquivo de pesos
    model = YOLO('yolo11n.pt')

    # Define o dispositivo para execução (GPU se disponível, senão CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Caminho para o arquivo de configuração do dataset
    data_path = './data/dataset.yaml'

    # Mostra o tipo do objeto do modelo
    print(type(model))

    # Move o modelo para o dispositivo selecionado
    model.to(device)

    # Inicia o treinamento do modelo com os parâmetros definidos
    model.train(
        data=data_path,      # Caminho para o arquivo YAML do dataset
        epochs=100,          # Número de épocas de treinamento
        batch=4,             # Tamanho do batch
        plots=True,          # Gera gráficos durante o treinamento
        imgsz=640,           # Tamanho das imagens
        save_period=-1,      # Não salva checkpoints intermediários
        save_dir='./runs',   # Diretório para salvar os resultados
        lr0=0.001,           # Taxa de aprendizado inicial
        optimizer='AdamW'    # Otimizador utilizado
    )