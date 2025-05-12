import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Definindo a pipeline de Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),  # Espelhamento horizontal
    layers.RandomRotation(0.1),  # Rotação aleatória
    layers.RandomZoom(0.1),  # Zoom aleatório
    layers.RandomContrast(0.1),  # Contraste aleatório
    layers.RandomBrightness(0.2),  # Brilho aleatório
    layers.RandomTranslation(0.1, 0.1),  # Translação aleatória
    layers.RandomHeight(0.1)  # Altura aleatória
])

# Carrega uma imagem exemplo
image = tf.keras.utils.load_img('../../inputs/faces.jpg', target_size=(224, 224))
image = tf.keras.utils.img_to_array(image)
image = tf.expand_dims(image, 0)  # adiciona dimensão batch

# Cria uma figura para o mosaico
plt.figure(figsize=(6, 6))

# Adiciona a imagem original ao mosaico
plt.subplot(3, 3, 1)
plt.title("Original")
plt.imshow(image[0].numpy().astype("uint8"))
plt.axis("off")

# Gera e adiciona 8 imagens aumentadas ao mosaico
for i in range(2, 10):
    augmented_image = data_augmentation(image)
    augmented_image = tf.squeeze(augmented_image, axis=0)  # Remove a dimensão batch
    plt.subplot(3, 3, i)
    plt.imshow(augmented_image.numpy().astype("uint8"))
    plt.axis("off")

# Mostra o mosaico
plt.tight_layout()
plt.show()