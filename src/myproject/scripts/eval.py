from myproject.utils.io import build_NeuralNetwork
from myproject.training.trainer import Trainer
from myproject.loss.loss_functions import CrossEntropy
from myproject.utils.io import Logger
from myproject.data_scripts.data_preprocessing import process_MNIST
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected device: {device}")

train_images, train_targets, test_images, test_targets = process_MNIST(device = device, ConvolutionalLayer=True)

model = build_NeuralNetwork("run_2026-01-28_12-02-23")
model.to(device)

trainer = Trainer(
    model = model,
    logger = Logger(),
    loss_fn = CrossEntropy(),
)

trainer.eval_model(
    eval_data = test_images,
    eval_targets = test_targets,
    batch_size = 64
)