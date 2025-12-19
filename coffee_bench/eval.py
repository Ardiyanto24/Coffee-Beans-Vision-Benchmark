import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def predict_classes(model, gen):
    probs = model.predict(gen, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = gen.classes
    return y_true, y_pred, probs

def make_report(y_true, y_pred, class_names):
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    return rep, cm

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

def plot_history(history, title_prefix=""):
    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    if "loss" in hist:
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, hist["loss"], label="train_loss")
        if "val_loss" in hist:
            plt.plot(epochs, hist["val_loss"], label="val_loss")
        plt.title(f"{title_prefix} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if "accuracy" in hist:
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, hist["accuracy"], label="train_acc")
        if "val_accuracy" in hist:
            plt.plot(epochs, hist["val_accuracy"], label="val_acc")
        plt.title(f"{title_prefix} Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
