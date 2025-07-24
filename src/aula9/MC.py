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
    y_true = ["gato", "rato", "gato", "gato", "rato", "cachorro", "gato", "rato", "cachorro", "cachorro",  "cachorro",
              "cachorro"]
    y_pred = ["gato", "rato", "rato", "gato", "rato", "cachorro", "cachorro", "rato", "cachorro", "cachorro", "cachorro",
              "cachorro"]

    labels = ["gato", "rato", "cachorro"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

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
