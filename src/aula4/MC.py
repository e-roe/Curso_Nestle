from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


y_true = ["gato", "rato", "gato", "gato", "rato", "cachorro", "gato", "rato", "cachorro", "cachorro",  "cachorro", "cachorro"]
y_pred = ["gato", "rato", "gato", "gato", "cachorro", "cachorro", "gato", "rato", "cachorro", "cachorro", "cachorro", "gato"]

labels = ["gato", "rato", "cachorro"]
cm = confusion_matrix(y_true, y_true, labels=labels)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# disp.plot()

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["gato", "rato", "cachorro"],
                yticklabels=["gato", "rato", "cachorro"])

plt.subplots_adjust(bottom=0.34)
plt.subplots_adjust(left=0.24)
plt.xticks(rotation=80)
plt.title('Confusion Matrix')
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.show()

