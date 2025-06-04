from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()  # Carregando o dataset Iris
X = iris.data
y = iris.target

# Dividindo o dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=3)  # Treinando um KNN
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test) # Fazendo inferências

accuracy = accuracy_score(y_test, y_pred) # Calculando acurácia
print(f"Accuracy: {accuracy}")
print(y_test)  # base de teste
print(y_pred)  # Predito pelo modelo
