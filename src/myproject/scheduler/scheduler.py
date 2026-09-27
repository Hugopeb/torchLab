import math
import torch

class ReducelrOnPlateau:
    def __init__(self, optimizer, patience = 2, factor = 0.5, min_lr = 1e-4, threshold = 1e-4):
        self.optimizer = optimizer
        self.threshold = threshold
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.best = float("inf")
        self.patience_counter = 0

    def step(self, loss):
        if loss is None:
            raise ValueError("Scheduler loss is None")

        if loss < self.best - self.threshold:
            self.best = loss
            self.patience_counter = 0

        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
            print(f"Scheduler: Reducing lr to {new_lr}")
            self.optimizer.lr = new_lr
            self.patience_counter = 0

    def get_config(self):
        return {
            "type": "ReducelrOnPlateau",
            "patience": self.patience,
            "factor": self.factor,
            "min_lr": self.min_lr,
            "threshold": self.threshold
        }



class CosineAnnealingLR:
    """
    Decays lr from its initial value to min_lr following half a cosine over
    num_epochs. step() is called once per epoch; the loss is accepted to keep
    the same interface as ReducelrOnPlateau but ignored.
    """
    def __init__(self, optimizer, num_epochs, min_lr = 0.0):
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.epoch = 0

    def step(self, loss = None):
        self.epoch = min(self.epoch + 1, self.num_epochs)
        cos = (1 + math.cos(math.pi * self.epoch / self.num_epochs)) / 2
        self.optimizer.lr = self.min_lr + (self.base_lr - self.min_lr) * cos

    def get_config(self):
        return {
            "type": "CosineAnnealingLR",
            "base_lr": self.base_lr,
            "num_epochs": self.num_epochs,
            "min_lr": self.min_lr
        }
