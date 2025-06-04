import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Geraçâo de dados sintéticos: y = 3x + noise
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 3 * X + np.random.randn(100, 1) * 1.

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Os dados cicam média=0 e desvio padrâo = 1
# Inicializando o modelo de regressão linear com SGD
model = SGDRegressor(max_iter=1, warm_start=True, penalty=None, learning_rate='constant', eta0=0.1)

# Plot the data points before training
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data')
plt.title('Pontos do Dataset Antes do Treinamento')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Configuração do plot
plt.figure(figsize=(10, 6))
epochs = 20  # Número de épocas para o treinamento

for epoch in range(epochs):
    model.partial_fit(X_scaled, y.ravel())  # Treina por uma época
    y_pred = model.predict(X_scaled)  # Predição utilizando o modelo atual

    # Plota os pontos e a linha da regressão
    plt.clf()
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, y_pred, color='red', label=f'Epoch {epoch + 1}')
    plt.title('Linear Regression Adjustment Over Epochs')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.pause(0.5)  # Pausa para visualizar o ajuste da reta

plt.show()