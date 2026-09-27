from myproject.model.activations import Dense, ReLU, Tanh, ConvolutionalLayer, MaxPool, ReshapeLayer, Dropout, BatchNorm
from myproject.loss.loss_functions import CrossEntropy
from myproject.model.neural_network import NeuralNetwork
from myproject.data_scripts.data_preprocessing import process_CIFAR10
from myproject.optimizer.optimizer import SGDWithMomentum
from myproject.training.trainer import Trainer
from myproject.utils.io import Logger
from myproject.scheduler.scheduler import CosineAnnealingLR
import torch

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected device: {device}")

train_images, train_targets, test_images, test_targets = process_CIFAR10(device = device)
print(train_images.shape, test_images.shape)

num_epochs = 50

# 32x32 -> pool -> 16x16 -> pool -> 8x8
model = NeuralNetwork([
    ConvolutionalLayer(32, 3, 3, padding = 1),
    BatchNorm(32),
    ReLU(),
    ConvolutionalLayer(32, 32, 3, padding = 1),
    BatchNorm(32),
    ReLU(),
    MaxPool(2, 2),
    ConvolutionalLayer(64, 32, 3, padding = 1),
    BatchNorm(64),
    ReLU(),
    ConvolutionalLayer(64, 64, 3, padding = 1),
    BatchNorm(64),
    ReLU(),
    MaxPool(2, 2),
    ReshapeLayer(),
    Dense(256, 64 * 8 * 8),
    BatchNorm(256),
    Tanh(),
    Dropout(0.5),
    Dense(10, 256)
])

model.to(device)

optimizer = SGDWithMomentum(model.parameters(), lr = 0.05, momentum = 0.9, weight_decay = 5e-4)

scheduler = CosineAnnealingLR(optimizer, num_epochs = num_epochs, min_lr = 1e-4)

trainer = Trainer(model, CrossEntropy(), Logger(), scheduler, augment = True)

trainer.train_model(
    optimizer = optimizer,
    train_data = train_images,
    train_targets = train_targets,
    eval_data = test_images,
    eval_targets = test_targets,
    num_epochs = num_epochs,
    batch_size = 64,
    eval = True
)
