import tensorflow as tf
from tensorflow.keras import layers

# Definindo a pipeline de Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),           # Espelhamento horizontal
    layers.RandomRotation(0.1),                # Rotação aleatória
    layers.RandomZoom(0.1),                    # Zoom aleatório
    layers.RandomContrast(0.1)                 # Contraste aleatório
])

# Exemplo: aplicando a transformação a uma imagem
import matplotlib.pyplot as plt

# Carrega uma imagem exemplo
image = tf.keras.utils.load_img('../../inputs/faces.jpg', target_size=(224, 224))
image = tf.keras.utils.img_to_array(image)
image = tf.expand_dims(image, 0)  # adiciona dimensão batch

# Aplica a data augmentation
augmented_image = data_augmentation(image)

# Converte o tensor aumentado para um array NumPy
augmented_image = tf.squeeze(augmented_image, axis=0)  # Remove a dimensão batch

# Mostra original e aumentada
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image[0].numpy().astype("uint8"))

plt.subplot(1, 2, 2)
plt.title("Augmented")
plt.imshow(augmented_image.numpy().astype("uint8"))  # Converte para NumPy antes de exibir
plt.show()
