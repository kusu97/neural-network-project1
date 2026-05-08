"""Train the Part B CNN model on MNIST."""
import gzip
import os
import pickle
from struct import unpack

import matplotlib
import mynn as nn
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from draw_tools.plot import plot


np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

if os.path.exists('idx.pickle'):
    with open('idx.pickle', 'rb') as f:
        idx = pickle.load(f)
else:
    idx = np.random.permutation(np.arange(num))
    with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

cnn_model = nn.models.Model_CNN(lambda_list=[1e-4, 1e-4, 1e-4])
optimizer = nn.optimizer.SGD(init_lr=0.1, model=cnn_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[2000, 3200], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max() + 1)

runner = nn.runner.RunnerM(
    cnn_model,
    optimizer,
    nn.metric.accuracy,
    loss_fn,
    batch_size=128,
    scheduler=scheduler,
)

runner.train(
    [train_imgs, train_labs],
    [valid_imgs, valid_labs],
    num_epochs=10,
    log_iters=100,
    eval_iters=100,
    save_dir=r'./best_models/part_b_cnn',
)

os.makedirs('figs', exist_ok=True)
fig, axes = plt.subplots(1, 2)
axes.reshape(-1)
fig.set_tight_layout(1)
plot(runner, axes)
fig.suptitle('Part B CNN learning curves')
fig.savefig(r'./figs/part_b_cnn_learning_curve.png', dpi=200)

with open(r'./best_models/part_b_cnn/history.pickle', 'wb') as f:
    pickle.dump(
        {
            'train_loss': runner.train_loss,
            'dev_loss': runner.dev_loss,
            'train_scores': runner.train_scores,
            'dev_scores': runner.dev_scores,
            'best_dev_accuracy': runner.best_score,
            'architecture': 'Conv(1,8,3,stride=2,pad=1)-ReLU-Linear(1568,128)-ReLU-Linear(128,10)',
            'batch_size': 128,
            'epochs': 10,
            'initial_lr': 0.1,
            'milestones': [2000, 3200],
            'gamma': 0.5,
        },
        f,
    )

print(f"Best validation accuracy: {runner.best_score:.4f}")
