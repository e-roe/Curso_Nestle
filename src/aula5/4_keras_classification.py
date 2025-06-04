import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Carregando o dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalização dos dados
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Construindo o modelo
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 3. Compilando o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Treinamento
model.fit(x_train, y_train, epochs=5)

# 5. Avaliação
model.evaluate(x_test, y_test)