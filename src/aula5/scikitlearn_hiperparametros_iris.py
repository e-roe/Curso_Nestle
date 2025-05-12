from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.92, random_state=42)

# Definir o modelo
rf = RandomForestClassifier(random_state=42)

# Definir hiperparâmetros para buscar
param_grid = {
    'n_estimators': [50, 100, 200],  # Número de árvores
    'max_depth': [None, 10, 20],  # Profundidade máxima
    'min_samples_split': [2, 5, 10]  # Critério para dividir nós
}

# Aplicar GridSearchCV para encontrar os melhores hiperparâmetros
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)


rf.fit(X_train, y_train)

# Making predictions
y_pred = rf.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy1: {accuracy}')


# Exibir os melhores hiperparâmetros
print("Melhores Hiperparâmetros:", grid_search.best_params_)
print("Melhor Score:", grid_search.best_score_)

# Avaliar o modelo com os melhores parâmetros
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Acurácia no Teste:", test_score)

