import os
import matplotlib.pyplot as plt

# Caminhos para os diretórios de treino e validação
train_dir = r"C:\Roe\Stepps\projs\Curso_Nestle\datasets\Animals\train"
val_dir = r"C:\Roe\Stepps\projs\Curso_Nestle\datasets\Animals\val"

# Função para contar o número de imagens por classe em um diretório
def count_images_per_class(data_dir):
    class_counts = {}  # Dicionário para armazenar as contagens por classe
    for class_name in os.listdir(data_dir):  # Itera sobre as pastas de classes
        class_path = os.path.join(data_dir, class_name)  # Caminho completo da classe
        if os.path.isdir(class_path):  # Verifica se é um diretório
            class_counts[class_name] = len(os.listdir(class_path))  # Conta as imagens na pasta
    return class_counts

# Obtém as distribuições de classes para os conjuntos de treino e validação
train_class_counts = count_images_per_class(train_dir)
val_class_counts = count_images_per_class(val_dir)

# Exibe as distribuições de classes do conjunto de treino
print("Distribuição de Classes - Treino:")
for class_name, count in train_class_counts.items():
    print(f"{class_name}: {count} imagens")

# Exibe as distribuições de classes do conjunto de validação
print("\nDistribuição de Classes - Validação:")
for class_name, count in val_class_counts.items():
    print(f"{class_name}: {count} imagens")

# Plota a distribuição das classes para treino e validação
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Gráfico para o conjunto de treino
axes[0].bar(train_class_counts.keys(), train_class_counts.values(), color='skyblue')
axes[0].set_title('Distribuição de Classes - Treino')  # Título do gráfico
axes[0].set_xlabel('Classes')  # Rótulo do eixo X
axes[0].set_ylabel('Número de Imagens')  # Rótulo do eixo Y
axes[0].tick_params(axis='x', rotation=45)  # Rotaciona os rótulos do eixo X

# Gráfico para o conjunto de validação
axes[1].bar(val_class_counts.keys(), val_class_counts.values(), color='lightgreen')
axes[1].set_title('Distribuição de Classes - Validação')  # Título do gráfico
axes[1].set_xlabel('Classes')  # Rótulo do eixo X
axes[1].tick_params(axis='x', rotation=45)  # Rotaciona os rótulos do eixo X

# Ajusta o layout para evitar sobreposição e exibe os gráficos
plt.tight_layout()
plt.show()