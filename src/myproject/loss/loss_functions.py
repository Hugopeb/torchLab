import torch

class CrossEntropy():
    """
    CrossEntropy loss module.

    This class implements a standard cross-entropy loss for multi-class
    classification using logits as input.

    Methods
    -------
    forward(logits, output)
        Computes the average cross-entropy loss over the batch.
        - logits: raw model outputs (unnormalized scores) of shape (batch_size, number_of_classes)
        - output: ground-truth labels of shape (batch_size,)
        Returns:
        - scalar loss value (mean over batch)

    backwards(output)
        Computes the gradient of the loss with respect to the input logits.
        The gradient is derived from the softmax + cross-entropy formulation:
        dL/dz = (softmax(z) - y) / batch_size
        where y is the one-hot encoded label vector.

        The method reuses cached softmax log-probabilities computed during
        forward pass for efficiency.

    Notes
    -----
    - Assumes `forward` is called before `backwards`.
    - Uses `torch.log_softmax` for numerical stability.
    - Gradient is returned as a tensor of shape (batch_size, number_of_classes).
    """
    def forward(self, preds, targets):
        self.targets = targets
        self.batch_size = targets.shape[0]
        self.log_probs = torch.log_softmax(preds, dim = 1)

        avg_batch_CE = -self.log_probs[torch.arange(self.batch_size), self.targets].mean()

        return avg_batch_CE
    
    def backwards(self):
        predicted_probs = torch.exp(self.log_probs)
        
        grad_preds = predicted_probs.clone()
        grad_preds[torch.arange(self.batch_size), self.targets] -= 1

        # The loss is the batch mean, so its gradient carries 1 / batch_size
        return grad_preds / self.batch_size


class MSE:
    '''
    Mean Squared Error loss with mean reduction over all elements
    (same behavior as torch.nn.MSELoss(reduction="mean")).
    '''
    def forward(self, preds, targets):
        self.diff = preds - targets
        loss = torch.mean(self.diff**2)

        return loss

    def backwards(self):
        grad_preds = (2 / self.diff.numel()) * self.diff

        return grad_preds