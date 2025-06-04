import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def normalize_confusion_matrix(cm, norm='true'):
    if norm == 'true':
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    elif norm == 'pred':
        cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    elif norm == 'all':
        cm_normalized = cm.astype('float') / cm.sum()
    else:
        raise ValueError("Unknown normalization type. Use 'true', 'pred', or 'all'.")

    return cm_normalized


if __name__ == '__main__':
    # Loading the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    labels =  np.unique(y)
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.92, random_state=42)

    # Training a K-Nearest Neighbors classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)

    # Making predictions
    y_pred = knn_classifier.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

    plt.subplots_adjust(bottom=0.34)
    plt.subplots_adjust(left=0.24)
    plt.xticks(rotation=0)
    plt.title('Matriz de Confusão')
    plt.xlabel('Classe Verdadeira')
    plt.ylabel('Classe Prevista')
    plt.show()

    cm_normalized = normalize_confusion_matrix(cm, norm='true')

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)

    plt.subplots_adjust(bottom=0.34)
    plt.subplots_adjust(left=0.24)
    plt.xticks(rotation=0)
    plt.title('Matriz de Confusão')
    plt.xlabel('Classe Verdadeira')
    plt.ylabel('Classe Prevista')
    plt.show()
