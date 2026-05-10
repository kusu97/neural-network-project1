"""Direction 5: confusion matrices, misclassified examples, and CNN kernels."""
import gzip
import os
import pickle
from struct import unpack

import matplotlib
import mynn as nn
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_mnist_test():
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images / images.max(), labels


def confusion_matrix(labels, preds, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels, preds):
        matrix[label, pred] += 1
    return matrix


def plot_confusion(matrix, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_misclassified(images, labels, preds, save_path, max_items=16):
    wrong_idx = np.where(labels != preds)[0][:max_items]
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for ax, idx in zip(axes.reshape(-1), wrong_idx):
        ax.imshow(images[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'T:{labels[idx]} P:{preds[idx]}', fontsize=9)
        ax.axis('off')
    for ax in axes.reshape(-1)[len(wrong_idx):]:
        ax.axis('off')
    fig.suptitle('CNN misclassified test examples')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_kernels(model, save_path):
    kernels = model.conv.W[:, 0]
    fig, axes = plt.subplots(2, 4, figsize=(6, 3))
    for ax, kernel in zip(axes.reshape(-1), kernels):
        ax.imshow(kernel, cmap='coolwarm')
        ax.axis('off')
    fig.suptitle('CNN first-layer convolution kernels')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


os.makedirs('figs', exist_ok=True)
os.makedirs('analysis_results', exist_ok=True)

test_imgs, test_labs = load_mnist_test()

mlp = nn.models.Model_MLP()
mlp.load_model(r'.\best_models\part_a_mlp\best_model.pickle')
mlp_logits = mlp(test_imgs)
mlp_preds = np.argmax(mlp_logits, axis=1)
mlp_matrix = confusion_matrix(test_labs, mlp_preds)

cnn = nn.models.Model_CNN()
cnn.load_model(r'.\best_models\part_b_cnn\best_model.pickle')
cnn.eval()
cnn_logits = cnn(test_imgs)
cnn_preds = np.argmax(cnn_logits, axis=1)
cnn_matrix = confusion_matrix(test_labs, cnn_preds)

plot_confusion(mlp_matrix, 'MLP baseline confusion matrix', r'./figs/part_c_mlp_confusion_matrix.png')
plot_confusion(cnn_matrix, 'CNN confusion matrix', r'./figs/part_c_cnn_confusion_matrix.png')
plot_misclassified(test_imgs, test_labs, cnn_preds, r'./figs/part_c_cnn_misclassified_examples.png')
plot_kernels(cnn, r'./figs/part_c_cnn_kernels.png')

summary = {
    'mlp_accuracy': float(np.mean(mlp_preds == test_labs)),
    'cnn_accuracy': float(np.mean(cnn_preds == test_labs)),
    'mlp_confusion_matrix': mlp_matrix,
    'cnn_confusion_matrix': cnn_matrix,
    'mlp_per_class_accuracy': np.diag(mlp_matrix) / mlp_matrix.sum(axis=1),
    'cnn_per_class_accuracy': np.diag(cnn_matrix) / cnn_matrix.sum(axis=1),
    'cnn_error_count': int(np.sum(cnn_preds != test_labs)),
}

with open(r'./analysis_results/part_c_error_analysis.pickle', 'wb') as f:
    pickle.dump(summary, f)

print(f"MLP test accuracy: {summary['mlp_accuracy']:.4f}")
print(f"CNN test accuracy: {summary['cnn_accuracy']:.4f}")
print(f"CNN errors: {summary['cnn_error_count']}")
print("CNN per-class accuracy:", np.round(summary['cnn_per_class_accuracy'], 4))
