import os
import shutil
import random
import argparse


def split_dataset(source_dir, dest_dir, split):
    """
    Divide um conjunto de dados de imagens e labels em conjuntos de treino, validação e teste.

    :param source_dir: Caminho para o diretório de origem contendo imagens e labels.
    :param dest_dir: Caminho para o diretório de destino para os conjuntos de treino, validação e teste.
    :param split: Tupla contendo a divisão percentual para treino, validação e teste (ex.: (70, 20, 10)).
    """
    # Garante que as porcentagens de divisão somem 100
    if sum(split) != 100:
        raise ValueError("As porcentagens de divisão devem somar 100.")

    # Cria os diretórios de destino
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, subset, 'labels'), exist_ok=True)

    # Obtém todos os arquivos de imagem e label
    images = sorted([f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png'))])
    labels = sorted([f for f in os.listdir(source_dir) if f.endswith('.txt')])

    # Garante que cada imagem tenha um label correspondente
    images = [img for img in images if os.path.splitext(img)[0] + '.txt' in labels]

    # Embaralha os dados
    data = list(zip(images, [os.path.splitext(img)[0] + '.txt' for img in images]))
    random.shuffle(data)

    # Calcula os índices de divisão
    total = len(data)
    train_end = int(split[0] / 100 * total)
    val_end = train_end + int(split[1] / 100 * total)

    # Divide os dados
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Função auxiliar para copiar arquivos
    def copy_files(data, subset):
        for img, lbl in data:
            shutil.copy(os.path.join(source_dir, img), os.path.join(dest_dir, subset, 'images', img))
            shutil.copy(os.path.join(source_dir, lbl), os.path.join(dest_dir, subset, 'labels', lbl))

    # Copia os arquivos para os diretórios correspondentes
    copy_files(train_data, 'train')
    copy_files(val_data, 'val')
    copy_files(test_data, 'test')

    print(f"Divisão do conjunto de dados concluída. Treino: {len(train_data)}, Validação: {len(val_data)}, Teste: {len(test_data)}")


# Uso
# python divide_base.py source_dir source_dir --split qt_treino qt_val qt_teste
# onde
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide um conjunto de dados em conjuntos de treino, validação e teste.")
    parser.add_argument("source_dir", type=str, help="Caminho para o diretório de origem contendo imagens e labels.")
    parser.add_argument("dest_dir", type=str, help="Caminho para o diretório de destino para os conjuntos de treino, validação e teste.")
    parser.add_argument("--split", type=int, nargs=3, default=[70, 20, 10],
                        help="Divisão percentual para treino, validação e teste (padrão: 70 20 10).")

    args = parser.parse_args()

    split_dataset(args.source_dir, args.dest_dir, args.split)