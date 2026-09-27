import torch    
from datetime import datetime
import torch.nn.functional as F

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def get_batches(data, targets, batch_size=64, shuffle=True):
    """
    Yields batches of data and targets.

    Args:
        data (torch.Tensor): Input data tensor of shape (N, ...).
        targets (torch.Tensor): Target tensor of shape (N, ...).
        batch_size (int): Batch size.
        shuffle (bool): Shuffle before batching.

    Yields:
        Tuple[torch.Tensor, torch.Tensor]: Batch of (data, targets).
    """

    if not isinstance(data, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("data and targets must be torch tensors")

    if data.shape[0] != targets.shape[0]:
        raise ValueError("data and targets must have the same length")

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    num_samples = data.shape[0]
    indices = torch.arange(num_samples)

    if shuffle:
        indices = indices[torch.randperm(num_samples)]

    for start in range(0, num_samples, batch_size):
        batch_idx = indices[start:start + batch_size]

        x_batch = data[batch_idx]
        y_batch = targets[batch_idx]

        yield x_batch, y_batch



def ensure_conv_input(input):
    '''
    Ensure tensor has shape (batch_size, input_channels, height, width).
    If input is (batch_size, height, width), add input channel dimension.

    Args:
        x: input tensor

    Returns:
        Tensor with shape (batch_size, channel_inputs, height, width)

    Raises:
        ValueError if input ndim is not 3 or 4.
    '''
    if input.ndim == 3:
        return input.unsqueeze(1)
    
    if input.ndim == 4:
        return input
        
    else:
        raise ValueError(
            f"Unexpected image shape. Expected image.ndim = 3 or 4 but got image.ndim = {input.ndim}"
        )

class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = None
        

def random_crop_and_flip(batch, padding=4):
    """
    batch: torch.Tensor of shape (N, C, H, W)
           already on the correct device
    returns: augmented batch
    """
    if batch.ndim == 3:
        batch = batch.unsqueeze(1)

    N, C, H, W = batch.shape
    device = batch.device

    batch_padded = F.pad(batch, (padding, padding, padding, padding), mode='reflect')

    max_offset = 2 * padding

    top = torch.randint(0, max_offset + 1, (N, 1), device=device)
    left = torch.randint(0, max_offset + 1, (N, 1), device=device)

    # Per-sample row/col indices into the padded image, gathered in one go
    rows = (top + torch.arange(H, device=device))[:, None, :, None]       # (N, 1, H, 1)
    cols = (left + torch.arange(W, device=device))[:, None, None, :]      # (N, 1, 1, W)
    batch_idx = torch.arange(N, device=device)[:, None, None, None]
    chan_idx = torch.arange(C, device=device)[None, :, None, None]

    cropped = batch_padded[batch_idx, chan_idx, rows, cols]

    flip_mask = torch.rand(N, device=device) < 0.5
    cropped[flip_mask] = torch.flip(cropped[flip_mask], dims=[3])

    return cropped
