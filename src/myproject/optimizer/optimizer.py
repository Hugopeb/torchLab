import torch


def apply_weight_decay(p, weight_decay):
    '''
    Returns the gradient with L2 weight decay added (grad + weight_decay * w).
    Only weight tensors (ndim > 1) are decayed, biases are left untouched.
    '''
    if weight_decay == 0.0 or p.data.ndim <= 1:
        return p.grad
    return p.grad + weight_decay * p.data


class SGD:
    def __init__(self, parameters, lr = 0.01, weight_decay = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay
        self.parameters = parameters

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * apply_weight_decay(p, self.weight_decay)

    def get_config(self):
        return {
            "type": "SGD",
            "lr": self.lr,
            "weight_decay": self.weight_decay
        }


class SGDWithMomentum:
    def __init__(self, parameters, lr=0.01, momentum=0.8, weight_decay = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.parameters = parameters

        self.velocity = [torch.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            # model.to(device) may run after the optimizer was built, keep velocity next to the params
            if self.velocity[i].device != p.data.device:
                self.velocity[i] = self.velocity[i].to(p.data.device)

            grad = apply_weight_decay(p, self.weight_decay)
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grad
            p.data += self.velocity[i]

    def get_config(self):
        return {
            "type": "SGDWithMomentum",
            "lr": self.lr,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay
        }
