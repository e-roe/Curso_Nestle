import os
import shutil
import random


def divide_videos(base_dir, output_dir, splits=(70, 20, 10)):
    # Garante que as proporções somem 100
    if sum(splits) != 100:
        raise ValueError("As proporções devem somar 100!")

    # Cria as pastas de saída para cada conjunto (treino, validação, teste)
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, 'labels'), exist_ok=True)

    # Lista todas as pastas de vídeos
    videos = [v for v in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, v))]
    random.shuffle(videos)  # Embaralha a lista para garantir aleatoriedade

    total = len(videos)
    n_train = int(splits[0] / 100 * total)
    n_val = int(splits[1] / 100 * total)

    # Divide os vídeos conforme as proporções
    train_videos = videos[:n_train]
    val_videos = videos[n_train:n_train + n_val]
    test_videos = videos[n_train + n_val:]

    split_map = [('train', train_videos), ('val', val_videos), ('test', test_videos)]

    # Para cada divisão, copia as imagens e labels para as pastas correspondentes
    for split_name, split_videos in split_map:
        for video in split_videos:
            video_path = os.path.join(base_dir, video)
            # Lista imagens e labels dentro da pasta do vídeo
            images = [f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))]
            labels = [f for f in os.listdir(video_path) if f.endswith('.txt')]
            # Copia as imagens para a pasta de destino, prefixando com o nome do vídeo
            for img in images:
                shutil.copy(os.path.join(video_path, img),
                            os.path.join(output_dir, split_name, 'images', f"{video}_{img}")
                )
            # Copia os labels para a pasta de destino, prefixando com o nome do vídeo
            for lbl in labels:
                shutil.copy(os.path.join(video_path, lbl),
                            os.path.join(output_dir, split_name, 'labels', f"{video}_{lbl}")
                )

    print("Concluído! Treino:", len(train_videos), "Validação:", len(val_videos), "Teste:", len(test_videos))


if __name__ == "__main__":
    local_base = r'D:\Datasets\Epi_Geral\base_pratica'  # Caminho para a base de vídeos
    destino_base = r'D:\Datasets\Epi_Geral\base_pratica_divs'  # Caminho para salvar as divisões
    divisoes = (70, 30, 0)  # Proporções: treino, validação, teste
    divide_videos(local_base, destino_base, splits=divisoes)