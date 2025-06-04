from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
dados = np.array([[1], [5], [10]])
# Normalização
normalizador = MinMaxScaler()
print(normalizador.fit_transform(dados))

# Padronização
padronizador = StandardScaler()
print(padronizador.fit_transform(dados))
