from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Carregar o dataset de dígitos
digits = load_digits()
X = digits.data
y = digits.target

# Dividir o dataset em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Definir quais hiperparâmetros ajustar
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1], 'kernel': ['rbf', 'linear', 'poly']}

# Criação de um classificador SVM
svm_classifier = SVC()

# GridSearchCV para ajuste dos hiperparâmetros
grid_search = GridSearchCV(svm_classifier, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Obter os melhores hiperparâmetros
best_params = grid_search.best_params_
print(f'Melhores hiperparâmetros: {best_params}')

# Avaliar o modelo com os melhores parâmetros
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Acurácia no Teste:", test_score)

svm_classifier.fit(X_train, y_train)
# Making predictions
y_pred = svm_classifier.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')