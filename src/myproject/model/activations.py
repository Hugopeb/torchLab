from torch.nn.functional import conv2d, fold, pad
import torch

from myproject.utils.utils import ensure_conv_input, Parameter

class Dense:
    """
    Fully connected layer (linear transformation).

    Attributes:
        input_size (int): Number of input features.
        output_size (int): Number of output neurons.
        weights (torch.Tensor): Weight matrix of shape (output_size, input_size).
        bias (torch.Tensor): Bias vector of shape (1, output_size).
    """
    def __init__(self, output_size, input_size):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = Parameter(
            torch.randn(output_size, input_size) * 0.01
        )
        self.bias = Parameter(
            torch.zeros(output_size) 
        )

    def parameters(self):
        return [self.weights, self.bias]
    
    def forward(self, input):
        self.input = input
        self.batch_size = self.input.shape[0]
        self.output = self.input @ self.weights.data.T + self.bias.data

        return self.output

    def backwards(self, grad_output):
        '''
        Computes the gradient of the loss function w.r.t weights, bias and input
        of the Dense layer. The gradients w.r.t weights and the bias are stored
        as attributes of self.weights and self.bias. 

        The method returns the gradient of the loss function w.r.t input or the layer.
        '''
        # grad_output already carries the 1 / batch_size factor from the loss
        self.weights.grad = grad_output.T @ self.input
        self.bias.grad = grad_output.sum(dim=0)

        self.grad_input = grad_output @ self.weights.data

        return self.grad_input

    def get_config(self):
        return {
            "type": "Dense",
            "output_size": int(self.output_size),
            "input_size": int(self.input_size) 
        }
    
    def state_dict(self):
        return {
            "weights": self.weights.data,
            "bias": self.bias.data
        }
    
    def load_state_dict(self, state):
        self.weights.data = state["weights"]
        self.bias.data = state["bias"]


    def stats(self):
        '''
        Returns a list with two dictionaries, one for the weights
        and one for the bias. They both contain the mean, std, max
        and min values of these parameters 
        '''
        weights_stats = {
            "mean": self.weights.data.mean().item(),
            "std": self.weights.data.std().item(),
            "max": self.weights.data.max().item(),
            "min": self.weights.data.min().item()
        }

        bias_stats = {
            "mean": self.bias.data.mean().item(),
            "std": self.bias.data.std().item(),
            "max": self.bias.data.max().item(),
            "min": self.bias.data.min().item()
        }

        return weights_stats, bias_stats
    
    def to(self, device):
        self.weights.data = self.weights.data.to(device)
        self.bias.data = self.bias.data.to(device)

        return self
    
    
class ReLU:
    """
    Rectified Linear Unit activation function.

    The layer outputs max(0, x) element-wise.  
    During the forward pass, a mask is stored to enable efficient backward pass.

    Attributes:
        mask (torch.Tensor): Boolean tensor indicating where input > 0.
    """
    def forward(self, input):
        self.mask = input > 0.0
        return self.mask * input
    
    def backwards(self, grad_output):
        return grad_output * self.mask

    def get_config(self):
        return {
            "type": "ReLU"
        }


class BatchNorm:
    """
    Batch normalization over the channel dimension.

    Works after a ConvolutionalLayer (input (N, C, H, W), statistics over N, H, W)
    or after a Dense layer (input (N, C), statistics over N).

    In training mode the batch mean and (biased) variance are used and the running
    estimates are updated with an exponential moving average (the running variance
    uses the unbiased estimate, as torch does). In evaluation mode the running
    estimates are used instead.

    Attributes:
        num_features (int): Number of channels C.
        weights (Parameter): Scale gamma, shape (C,).
        bias (Parameter): Shift beta, shape (C,).
        running_mean, running_var (torch.Tensor): Estimates used at evaluation.
        momentum (float): Weight of the current batch in the running estimates.
        training (bool): Set by NeuralNetwork.train() / NeuralNetwork.eval().
    """
    def __init__(self, num_features, momentum = 0.1, eps = 1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.training = True

        self.weights = Parameter(torch.ones(num_features))
        self.bias = Parameter(torch.zeros(num_features))

        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def parameters(self):
        return [self.weights, self.bias]

    def _shape(self, input):
        '''
        Returns the dims to reduce over and the shape that broadcasts a
        per-channel (C,) tensor against the input.
        '''
        if input.ndim == 4:
            return (0, 2, 3), (1, -1, 1, 1)
        if input.ndim == 2:
            return (0,), (1, -1)
        raise ValueError(f"BatchNorm expects input.ndim = 2 or 4 but got input.ndim = {input.ndim}")

    def forward(self, input):
        self.dims, self.view = self._shape(input)

        if self.training:
            mean = input.mean(dim = self.dims)
            var = input.var(dim = self.dims, unbiased = False)

            m = input.numel() // self.num_features
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var * m / max(m - 1, 1)
        else:
            mean = self.running_mean
            var = self.running_var

        self.inv_std = (var + self.eps).rsqrt().view(self.view)
        self.x_hat = (input - mean.view(self.view)) * self.inv_std

        return self.weights.data.view(self.view) * self.x_hat + self.bias.data.view(self.view)

    def backwards(self, grad_output):
        '''
        Gradients w.r.t gamma, beta and the input.

        In training mode mean and variance depend on the input, which gives
            dx = inv_std / m * (m * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
        with dx_hat = grad_output * gamma and the sums over the reduced dims.
        In evaluation mode the statistics are constants, so dx = dx_hat * inv_std.
        '''
        self.weights.grad = (grad_output * self.x_hat).sum(dim = self.dims)
        self.bias.grad = grad_output.sum(dim = self.dims)

        grad_x_hat = grad_output * self.weights.data.view(self.view)

        if not self.training:
            return grad_x_hat * self.inv_std

        m = grad_output.numel() // self.num_features
        sum_grad = grad_x_hat.sum(dim = self.dims, keepdim = True)
        sum_grad_x_hat = (grad_x_hat * self.x_hat).sum(dim = self.dims, keepdim = True)

        return self.inv_std / m * (m * grad_x_hat - sum_grad - self.x_hat * sum_grad_x_hat)

    def get_config(self):
        return {
            "type": "BatchNorm",
            "num_features": int(self.num_features),
            "momentum": self.momentum,
            "eps": self.eps
        }

    def state_dict(self):
        return {
            "weights": self.weights.data,
            "bias": self.bias.data,
            "running_mean": self.running_mean,
            "running_var": self.running_var
        }

    def load_state_dict(self, state):
        self.weights.data = state["weights"]
        self.bias.data = state["bias"]
        self.running_mean = state["running_mean"]
        self.running_var = state["running_var"]

    def stats(self):
        '''
        Returns the mean, std, max and min of gamma (weights) and beta (bias)
        '''
        return tuple(
            {
                "mean": t.data.mean().item(),
                "std": t.data.std().item(),
                "max": t.data.max().item(),
                "min": t.data.min().item()
            }
            for t in (self.weights, self.bias)
        )

    def to(self, device):
        self.weights.data = self.weights.data.to(device)
        self.bias.data = self.bias.data.to(device)
        self.running_mean = self.running_mean.to(device)
        self.running_var = self.running_var.to(device)

        return self


class Dropout:
    """
    Inverted dropout.

    During training each activation is zeroed with probability p and the
    survivors are scaled by 1 / (1 - p), so the expected output matches the
    input and no rescaling is needed at evaluation time, where the layer is
    the identity.

    Attributes:
        p (float): Probability of dropping an activation.
        training (bool): Set by NeuralNetwork.train() / NeuralNetwork.eval().
        mask (torch.Tensor): Scaled keep-mask stored for the backward pass.
    """
    def __init__(self, p = 0.5):
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.training = True

    def forward(self, input):
        if not self.training or self.p == 0.0:
            self.mask = None
            return input

        self.mask = (torch.rand_like(input) >= self.p) / (1.0 - self.p)
        return input * self.mask

    def backwards(self, grad_output):
        if self.mask is None:
            return grad_output
        return grad_output * self.mask

    def get_config(self):
        return {
            "type": "Dropout",
            "p": self.p
        }


class Tanh:
    """
    Hyperbolic tangent activation function.

    During forward pass, the output is stored to compute gradients efficiently
    in the backward pass.

    Attributes:
        output (torch.Tensor): Stores tanh(x) for use in backward.
    """
    def forward(self, input):
        self.output = torch.tanh(input)
        return self.output
    
    def backwards(self, grad_output):
        grad_input = grad_output * (1 - self.output**2)
        return grad_input

    def get_config(self):
        return {
            "type": "Tanh"
        }
    

class ConvolutionalLayer:
    def __init__(self, output_channels, input_channels, kernel_size, stride = 1, padding = 0):
        self.stride = stride
        self.padding = padding
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size

        # Scale by 1 / sqrt(fan_in) so pre-activations don't saturate Tanh
        fan_in = self.input_channels * self.kernel_size * self.kernel_size
        self.weights = Parameter(
            torch.randn(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size) / fan_in**0.5
        )
        self.bias = Parameter(
            torch.zeros(self.output_channels)
        ) 

    def parameters(self):
        return [self.weights, self.bias]

    def forward(self, input): 
        """
        Forward pass for convolution.

        Input shape: (batch, in_channels, H, W)
        Output shape: (batch, out_channels, H_out, W_out)

        Uses custom conv2d implementation.

        Stores input and computed output for backprop.
        """
        self.input = ensure_conv_input(input)
        self.input_size = self.input.shape[2]
        self.batch_size = self.input.shape[0]

        self.output = conv2d(
            self.input,
            self.weights.data,
            self.bias.data,
            self.stride,
            self.padding
        )

        return self.output
    
    def backwards(self, grad_output):
        """
        Backprop for convolution.

        grad_output shape: (batch, out_channels, H_out, W_out)

        Computes:
        - grad_bias: gradient wrt bias
        - grad_weights: gradient wrt filters
        - grad_input: gradient wrt input (propagated to previous layer)

        Notes:
        - Weight gradient computed using convolution between input and grad_output.
        - Input gradient computed using flipped weights (standard conv backprop).
        - For stride > 1, grad_output is used as a dilated kernel for the weight
          gradient and is dilated with zeros before computing the input gradient.
        """
        k = self.kernel_size
        s = self.stride
        H, W = self.input.shape[2:]

        self.bias.grad = grad_output.sum(dim = (0, 2, 3))

        input_permuted = self.input.permute(1,0,2,3)
        grad_output_permuted = grad_output.permute(1,0,2,3)

        # When (H + 2p - k) is not a multiple of the stride the result is
        # larger than the kernel, the extra rows/cols are discarded
        self.weights.grad = conv2d(
            input_permuted,
            grad_output_permuted,
            padding = self.padding,
            dilation = s
        )[:, :, :k, :k].permute(1,0,2,3)

        weights_flipped = self.weights.data.flip(dims = [-1, -2]).permute(1,0,2,3)

        # Insert (stride - 1) zeros between grad_output elements
        if s > 1:
            N, C, H_out, W_out = grad_output.shape
            grad_output_dilated = torch.zeros(
                (N, C, s * (H_out - 1) + 1, s * (W_out - 1) + 1),
                device = grad_output.device
            )
            grad_output_dilated[:, :, ::s, ::s] = grad_output
            grad_output = grad_output_dilated

        # Full convolution gives the gradient w.r.t the padded input,
        # of size s * (H_out - 1) + k along each spatial dim
        grad_input_padded = conv2d(
            grad_output,
            weights_flipped,
            padding = k - 1
        )

        # Padded input rows/cols skipped by the stride receive zero gradient
        p = self.padding
        grad_input_padded = pad(
            grad_input_padded,
            (0, W + 2*p - grad_input_padded.shape[3], 0, H + 2*p - grad_input_padded.shape[2])
        )

        # Discard the gradient that falls on the padding
        self.grad_input = grad_input_padded[:, :, p:p + H, p:p + W]

        return self.grad_input

    def get_config(self):
        return {
            "type": "ConvolutionalLayer",
            "output_channels": int(self.output_channels),
            "input_channels": int(self.input_channels),
            "kernel_size": int(self.kernel_size),
            "stride": int(self.stride),
            "padding": int(self.padding)
        }
    
    def state_dict(self):
        return {
            "weights": self.weights.data,
            "bias": self.bias.data
        }
        
    def load_state_dict(self, state):
        self.weights.data = state["weights"]
        self.bias.data = state["bias"]

    
    def stats(self):
        '''
        Returns a list of dictionaries with the mean, std, max and min
        values of the weights and bias of the layer. .item() assures 
        values are JSONserializable.
        '''
        weights_stats = {
            "mean": self.weights.data.mean().item(),
            "std": self.weights.data.std().item(),
            "max": self.weights.data.max().item(),
            "min": self.weights.data.min().item()
        }

        bias_stats = {
            "mean": self.bias.data.mean().item(),
            "std": self.bias.data.std().item(),
            "max": self.bias.data.max().item(),
            "min": self.bias.data.min().item()
        }

        return weights_stats, bias_stats
    
    def to(self, device):
        self.weights.data = self.weights.data.to(device)
        self.bias.data = self.bias.data.to(device)

        return self


class ReshapeLayer:
    """
    Flatten layer used to bridge convolutional layers and fully connected layers.

    Converts:
        (batch, C, H, W) → (batch, C * H * W)

    During backprop, reshapes gradients back to the original input shape.
    """
    def forward(self, x):
        self.input_shape = x.shape

        self.output_shape = x.reshape(x.size(0), -1)
        return self.output_shape

    def backwards(self, grad_output):
        return grad_output.reshape(self.input_shape)

    def get_config(self):
        return {
            "type": "ReshapeLayer"
        }


class MaxPool:
    """
    2D Max Pooling layer (no learnable parameters).

    Uses torch.unfold to extract sliding windows and applies max
    over each window. Backprop routes gradients only to max locations.
    """
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, input):
        """
        Forward pass.

        input shape:  (N, C, H, W)
        output shape: (N, C, out_h, out_w)

        Stores argmax indices for use during backprop.
        """
        self.input_shape = input.shape
        k = self.kernel_size
        s = self.stride

        # (N, C, out_h, out_w, k, k)
        input_unfold = input.unfold(2, k, s).unfold(3, k, s)

        input_unfold_flat = input_unfold.reshape(
            *input_unfold.shape[:-2], k * k
        )

        self.argmax = input_unfold_flat.argmax(dim=-1)

        output = input_unfold_flat.max(dim=-1).values
        return output

    def backwards(self, grad_output):
        """
        Backward pass.

        grad_output shape: (N, C, out_h, out_w)

        Gradient is routed only to the max element of each pooling window.
        """
        N, C, H, W = self.input_shape
        k = self.kernel_size
        s = self.stride
        out_h, out_w = grad_output.shape[2:]

        # Gradient per window, same structure as the forward unfold
        # (reshaping an unfold view makes a copy, so we can't scatter
        # into grad_input directly)
        grad_windows = torch.zeros(
            (N, C, out_h, out_w, k * k), device = grad_output.device
        )

        grad_windows.scatter_(
            dim=-1,
            index=self.argmax.unsqueeze(-1),
            src=grad_output.unsqueeze(-1)
        )

        # fold expects (N, C * k * k, out_h * out_w) and places each window
        # back in (H, W), summing where windows overlap
        grad_windows = grad_windows.permute(0, 1, 4, 2, 3).reshape(
            N, C * k * k, out_h * out_w
        )
        grad_input = fold(
            grad_windows,
            output_size = (H, W),
            kernel_size = k,
            stride = s
        )

        return grad_input
    
    def get_config(self):
        return {
            "type": "MaxPool",
            "kernel_size": self.kernel_size,
            "stride": self.stride
        }


class GAP:
    """
    Global Average Pooling (GAP) layer.

    Replaces fully connected layers at the end of CNNs by averaging
    each feature map into a single value.

    Input shape:  (batch, channels, H, W)
    Output shape: (batch, channels)
    """
    def forward(self, input):
        self.input = input

        output = input.mean(dim=(-1, -2))
        return output

    def backwards(self, grad_output):
        """
        Backward pass.

        grad_output shape: (batch, channels)

        Each spatial location contributed equally to the mean,
        so the gradient is distributed uniformly over H*W.
        """
        B, C, H, W = self.input.shape

        grad_input = grad_output[:, :, None, None] / (H * W)

        grad_input = grad_input.expand(B, C, H, W)
        return grad_input

    def get_config(self):
        return {
            "type": "GAP"
        }


        
