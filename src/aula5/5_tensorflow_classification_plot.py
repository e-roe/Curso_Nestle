import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Listas para armazenar métricas
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Carrega o dataset MNIST, que contém imagens de dígitos escritos à mão
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Pré-processamento dos dados
# Redimensiona as imagens para vetores de 784 elementos (28x28) e normaliza os valores para [0, 1]
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test  = x_test.reshape(-1, 784).astype(np.float32) / 255.0
# Converte os rótulos para one-hot encoding (10 classes)
y_train = tf.one_hot(y_train, depth=10)
y_test  = tf.one_hot(y_test, depth=10)

# Define os hiperparâmetros do modelo
learning_rate = 0.1  # Taxa de aprendizado para o otimizador
epochs = 5           # Número de épocas de treinamento
batch_size = 128     # Tamanho do lote para o treinamento

# Cria datasets do TensorFlow para treinamento e teste
# Os dados de treinamento são embaralhados e divididos em lotes
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Inicializa os pesos e bias do modelo
# W representa os pesos conectando as entradas às saídas
# b representa os bias adicionados às saídas
W = tf.Variable(tf.random.normal([784, 10], stddev=0.1))
b = tf.Variable(tf.zeros([10]))

# Define o otimizador SGD (Stochastic Gradient Descent)
optimizer = tf.optimizers.SGD(learning_rate)

# Função de perda: calcula a entropia cruzada entre as previsões e os rótulos reais
def compute_loss(logits, labels):
    return tf.reduce_mean(
       tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# Função de acurácia: calcula a porcentagem de previsões corretas
def compute_accuracy(logits, labels):
    preds = tf.argmax(logits, axis=1)  # Obtém as classes previstas
    targets = tf.argmax(labels, axis=1)  # Obtém as classes reais
    return tf.reduce_mean(tf.cast(tf.equal(preds, targets), tf.float32))

# Loop de avaliação e treinamento por época
for epoch in range(epochs):

    running_loss = 0.0
    correct, total = 0, 0

    for step, (x_batch, y_batch) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            logits = tf.matmul(x_batch, W) + b
            loss = compute_loss(logits, y_batch)
        grads = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(grads, [W, b]))

        running_loss += loss.numpy()
        preds = tf.argmax(logits, axis=1)
        targets = tf.argmax(y_batch, axis=1)
        correct += tf.reduce_sum(tf.cast(tf.equal(preds, targets), tf.float32)).numpy()
        total += y_batch.shape[0]

    train_loss = running_loss / len(train_ds)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validação
    total_acc = 0
    num_batches = 0
    val_loss = 0.0
    correct, total = 0, 0

    for x_batch, y_batch in test_ds:
        logits = tf.matmul(x_batch, W) + b
        loss = compute_loss(logits, y_batch)
        val_loss += loss.numpy()

        preds = tf.argmax(logits, axis=1)
        targets = tf.argmax(y_batch, axis=1)
        correct += tf.reduce_sum(tf.cast(tf.equal(preds, targets), tf.float32)).numpy()
        total += y_batch.shape[0]

    val_loss /= len(test_ds)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# Plot dos gráficos
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Loss Treinamento')
plt.plot(epochs_range, val_losses, label='Loss Validação')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.title('Loss por Época')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Acurácia treinamento')
plt.plot(epochs_range, val_accuracies, label='Acurácia Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.title('Acurácia por Época')
plt.legend()

plt.tight_layout()
plt.show()