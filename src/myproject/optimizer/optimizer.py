import torch


class SGD:
    def __init__(self, parameters, lr = 0.01):
        self.lr = lr
        self.parameters = parameters

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def get_config(self):
        return {
            "type": "SGD",
            "lr": self.lr
        }


class SGDWithMomentum:
    def __init__(self, parameters, lr=0.01, momentum=0.8):
        self.lr = lr
        self.momentum = momentum
        self.parameters = parameters

        self.velocity = [torch.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            # model.to(device) may run after the optimizer was built, keep velocity next to the params
            if self.velocity[i].device != p.data.device:
                self.velocity[i] = self.velocity[i].to(p.data.device)

            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * p.grad
            p.data += self.velocity[i]

    def get_config(self):
        return {
            "type": "SGDWithMomentum",
            "lr": self.lr,
            "momentum": self.momentum
        }
