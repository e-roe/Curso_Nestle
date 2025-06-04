import numpy as np

print('1. Criando Arrays')
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])
print(a)  # [1 2 3]
print()
print(b)  # matriz 2x2
print()

print('2. Performance Vetorizada')
x = np.arange(1000000)
y = x * 2  # vetorizado, muito mais rápido que loops
print(y)
print()

print('3. Operações Matemáticas')
a = np.array([1, 2, 3])
print(np.sum(a))      # 6
print(np.mean(a))     # 2.0
print(np.std(a))      # 0.816…
print(np.sqrt(a))  # [1. 2. 3.]


print('4. Indexação e Slicing')
a = np.array([10, 20, 30, 40])
print(a[1:3])  # [20 30]

print('Informaçoes')
b = np.array([[1, 2], [3, 4], [3, 4]])
print(b.shape)  # (2, 2) - 2 linhas e 2 colunas
print(b.ndim)  # número de dimensões
print(b.dtype)  # tipo de dado
print(b.size)  # número total de elementos

