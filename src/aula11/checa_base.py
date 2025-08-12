import cv2
import os
import matplotlib.pyplot as plt

# Dicionário com as cores para cada classe de bounding box
cores_bboxes = {'helmet': (255, 100, 100), 'vest': (255, 0, 255), 'person': (10, 250, 100), 'ear': (100, 150, 255)}


def checa_anotacoes(root_path, classes, save_video=False):
    import collections
    class_counts = collections.Counter()  # Contador para as classes
    videos = os.listdir(root_path)  # Lista os diretórios de vídeos
    bbox_counter = 0
    frame_counter = 0
    for video in videos:
        files = os.listdir(os.path.join(root_path, video))  # Lista arquivos em cada vídeo
        for file in files:
            if 'txt' not in file:
                continue  # Pula arquivos que não são anotações
            with open(os.path.join(root_path, video, file)) as f:
                lines = f.readlines()
                if len(lines) == 0:
                    print(f'Arquivo {file}, sem anotacoes')
                    continue  # Pula arquivos de anotação vazios
            image = None

            for line in lines:
                ll = line[:].split(' ')
                class_idx = int(ll[0])  # Índice da classe
                class_counts[class_idx] += 1  # Conta a ocorrência da classe
                if image is None:
                    # Lê a imagem correspondente à anotação
                    image = cv2.imread(os.path.join(root_path, video, file[:-3] + 'jpg'))
                    frame_counter += 1

                width = image.shape[1]
                height = image.shape[0]
                cx = float(ll[1])
                cy = float(ll[2])
                w = float(ll[3])
                h = float(ll[4])
                # Converte coordenadas YOLO para pixels
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                color = cores_bboxes[classes[class_idx]]
                # Desenha o retângulo da bounding box
                cv2.rectangle(image, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), (255, 255, 255), 4)
                cv2.rectangle(image, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), color, 4)
                # Escreve o nome da classe na imagem
                cv2.putText(image, classes[class_idx], (int(x1 * width), int(y1 * height) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.25, color,
                            2, lineType=cv2.LINE_AA)
                cv2.putText(image, classes[class_idx], (int(x1 * width) + 3, int(y1 * height) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255),
                            2, lineType=cv2.LINE_AA)
                bbox_counter += 1

            # Mostra a imagem com as anotações
            cv2.imshow('', cv2.resize(image, (0, 0), fx=0.5, fy=0.5))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
    # Plota o histograma das classes
    plt.figure(figsize=(8, 5))
    class_names = [classes[i] for i in range(len(classes))]
    counts = [class_counts[i] for i in range(len(classes))]
    plt.bar(class_names, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Quantidade')
    plt.title('Distribuição de Anotações por Classe')
    plt.show()


if __name__ == '__main__':
    classes = ['helmet', 'vest', 'person', 'ear']
    src_path = r'D:\Datasets\Epi_Geral\base_pratica'
    checa_anotacoes(src_path, classes, save_video=True)