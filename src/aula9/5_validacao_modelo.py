import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

bbox_colors = {'Ambulance': (255, 100, 100), 'Bus': (255, 0, 255), 'Car': (10, 250, 100),
               'Motorcycle': (100, 150, 255), 'Truck': (255, 0, 255)}


def normalize_confusion_matrix(cm, norm='true'):
    """
    Normaliza uma matriz de confusão.

    Parâmetros:
    cm (array-like): Matriz de confusão a ser normalizada.
    norm (str): Tipo de normalização ('true', 'pred', 'all').

    Retorna:
    ndarray: Matriz de confusão normalizada.
    """
    if norm == 'true':
        # Normaliza pelas linhas (rótulos verdadeiros)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    elif norm == 'pred':
        # Normaliza pelas colunas (rótulos previstos)
        cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    elif norm == 'all':
        # Normaliza toda a matriz
        cm_normalized = cm.astype('float') / cm.sum()
    else:
        raise ValueError("Tipo de normalização desconhecido. Use 'true', 'pred' ou 'all'.")

    return cm_normalized


def plot_confusion_matrix(cm, classes, fmt='.2f', title='Confusion Matrix', save_path='confusion_matrix.png'):
    """
    Plota uma matriz de confusão como um mapa de calor.

    Parâmetros:
    cm (array-like): Matriz de confusão a ser plotada.
    classes (list): Lista de nomes das classes para os eixos.
    fmt (str): Formato das anotações (ex.: '.2f' para floats, 'd' para inteiros).
    title (str): Título do gráfico.
    save_path (str): Caminho para salvar o gráfico como um arquivo de imagem.
    """
    if fmt == 'd':
        cm = cm.astype(int)  # Converte para inteiros se o formato for 'd'

    # Define o tamanho da figura
    plt.figure(figsize=(10, 10))

    # Cria o mapa de calor com anotações
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=classes, yticklabels=classes, square=True,
                cbar_kws={"shrink": .8})

    # Configura os rótulos e o título
    plt.title(title, pad=20)
    plt.xlabel('Rótulo Previsto')
    plt.ylabel('Rótulo Verdadeiro')
    plt.xticks(rotation=45)

    # Ajusta as margens manualmente
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

    # Salva o gráfico como uma imagem
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.show()


def validate_model(model, data_path):
    """
    Valida o modelo YOLO em um conjunto de dados e retorna as métricas de validação.

    :param model: O modelo YOLO a ser validado.
    :param data_path: Caminho para o arquivo de configuração do conjunto de dados.
    :return: None
    """

    conf_str = 'veiculos'
    os.makedirs(f'../outputs/{conf_str}', exist_ok=True)
    # Valida o modelo no conjunto de dados de anotação
    metrics = model.val(data=data_path)

    # Extrai a matriz de confusão das métricas de validação
    a = metrics.confusion_matrix.matrix
    os.makedirs(f'../../outputs/{conf_str}', exist_ok=True)
    classes = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
    # Plota e salva a matriz de confusão bruta
    plot_confusion_matrix(a.astype(int), classes, fmt='d', save_path=f'../outputs/{conf_str}/confusion_matrix_{conf_str}.png')

    # Normaliza a matriz de confusão por linha (rótulos verdadeiros) e a plota
    cm_normalized_true = normalize_confusion_matrix(metrics.confusion_matrix.matrix, norm='true')
    plot_confusion_matrix(cm_normalized_true, classes, fmt='.2f',
                          title='Matriz de Confusão Normalizada (Por Linha) Verdadeiro',
                          save_path=f'../../outputs/{conf_str}/confusion_matrix_true_{conf_str}.png')

    # Normaliza a matriz de confusão por coluna (rótulos previstos) e a plota
    cm_normalized_pred = normalize_confusion_matrix(metrics.confusion_matrix.matrix, norm='pred')
    plot_confusion_matrix(cm_normalized_pred, classes, fmt='.2f',
                          title='Matriz de Confusão Normalizada (Por Coluna) Previsto',
                          save_path=f'../../outputs/{conf_str}/confusion_matrix_pred_{conf_str}.png')

    # Normaliza toda a matriz de confusão (matriz inteira) e a plota
    cm_normalized_all = normalize_confusion_matrix(metrics.confusion_matrix.matrix, norm='all')
    plot_confusion_matrix(cm_normalized_all, classes, fmt='.2f',
                          title='Matriz de Confusão Normalizada (Matriz Inteira) Tudo',
                          save_path=f'../../outputs/{conf_str}/confusion_matrix_all_{conf_str}.png')

    # Acessa as métricas de IoU (Interseção sobre União)
    map50 = metrics.box.map50  # mAP em IoU=0.50
    map75 = metrics.box.map75  # mAP em IoU=0.75
    maps = metrics.box.maps  # mAP para cada classe em IoU=0.50:0.95

    # Imprime as métricas de mAP
    print('Precisão Média (mAP) em IoU=0.50:', map50)
    print('Precisão Média (mAP) em IoU=0.75:', map75)
    print('Precisão Média (mAP) em IoU=0.50:0.95:', maps)
    print('Precisão Média (mAP) em IoU=0.50:0.95:', metrics.box.map)


def video_inference(video_path, model):
    """
    Realiza a inferência em um vídeo usando o modelo YOLO.

    Parâmetros:
    video_path (str): Caminho para o arquivo de vídeo.
    model: Modelo YOLO treinado para realizar a inferência.

    Funcionalidade:
    - Abre o vídeo especificado e processa cada frame.
    - Para cada frame, realiza a detecção de objetos usando o modelo.
    - Desenha caixas delimitadoras (bounding boxes) ao redor dos objetos detectados.
    - Exibe o frame processado em uma janela redimensionada.
    """

    cap = cv2.VideoCapture(video_path)  # Abre o vídeo para leitura
    while True:
        ret, frame = cap.read()  # Lê um frame do vídeo
        if not ret:
            break  # Encerra o loop se não houver mais frames

        results = model(frame, save=False)  # Realiza a inferência no frame
        for result in results:
            for box in result.boxes:
                boxc = box.xyxy.cpu().numpy()[0]  # Coordenadas da caixa delimitadora
                confidence = box.conf.cpu().numpy()[0]  # Nível de confiança da detecção
                color = bbox_colors[result.names[int(box.cls[0])]]  # Cor associada à classe detectada
                # Desenha a caixa delimitadora com borda preta e colorida
                cv2.rectangle(frame, (int(boxc[0]), int(boxc[1])), (int(boxc[2]), int(boxc[3])), color, 3)
                # Adiciona o nome da classe e o nível de confiança como texto
                cv2.putText(frame, f"{result.names[int(box.cls[0])]} ({confidence:.2f})",
                            (int(boxc[0]) + 4, int(boxc[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Sai do loop se a tecla 'q' for pressionada
        cv2.imshow('', frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cap.release()  # Libera o vídeo após o processamento

if __name__ == '__main__':
    model = YOLO('./runs/detect/train4/weights/best.pt')

    # Extract class names from the configuration
    classes = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
    data_path = '../../datasets/veiculos/dataset.yaml'
    validate_model(model, data_path)

    video_inference('../../inputs/TrafficPolice.mp4', model)