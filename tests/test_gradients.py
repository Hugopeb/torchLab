import pytest
import torch
import torch.nn.functional as F

from myproject.model.activations import Dense, ReLU, Tanh, ConvolutionalLayer, ReshapeLayer, MaxPool, GAP
from myproject.model.neural_network import NeuralNetwork
from myproject.loss.loss_functions import CrossEntropy, MSE
from myproject.optimizer.optimizer import SGDWithMomentum

'''
Gradient checks against torch.autograd.

Every forward pass in the framework is built from differentiable torch ops,
so if the input and the parameters are marked with requires_grad, autograd
computes the exact gradient of our own forward. The manual backwards pass
must match it.
'''

# TF32 convolutions on cuda are not precise enough for these tolerances
torch.backends.cudnn.allow_tf32 = False

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
ATOL = 1e-5
RTOL = 1e-4


@pytest.fixture(autouse = True)
def seed():
    torch.manual_seed(0)


def check_layer_gradients(layer, input):
    '''
    Runs forward on the layer, backpropagates a random grad_output both
    with autograd and with layer.backwards, and compares grad_input and
    the gradient of every parameter.
    '''
    params = layer.parameters() if hasattr(layer, "parameters") else []

    input = input.clone().requires_grad_()
    for p in params:
        p.data.requires_grad_()

    output = layer.forward(input)
    grad_output = torch.randn_like(output)
    (output * grad_output).sum().backward()

    with torch.no_grad():
        grad_input = layer.backwards(grad_output)

    assert grad_input.shape == input.shape
    assert torch.allclose(grad_input, input.grad, atol = ATOL, rtol = RTOL)

    for p in params:
        # Shape check matters: broadcasting can hide a wrong gradient shape
        assert p.grad.shape == p.data.shape
        assert torch.allclose(p.grad, p.data.grad, atol = ATOL, rtol = RTOL)


@pytest.mark.parametrize("device", DEVICES)
def test_dense(device):
    layer = Dense(7, 12).to(device)
    check_layer_gradients(layer, torch.randn(5, 12, device = device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("activation", [ReLU, Tanh])
def test_activations(device, activation):
    check_layer_gradients(activation(), torch.randn(5, 3, 6, 6, device = device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "padding, stride, size",
    [
        (0, 1, 8),
        (1, 1, 8),
        (2, 1, 8),
        (0, 2, 8),
        (1, 2, 9),   # (H + 2p - k) not a multiple of the stride
        (2, 3, 10),
        (3, 1, 8),   # padding > kernel_size - 1
        (4, 2, 9),
    ]
)
def test_convolutional_layer(device, padding, stride, size):
    layer = ConvolutionalLayer(4, 3, 3, stride = stride, padding = padding).to(device)
    # Non-zero bias so the forward matches a general convolution
    layer.bias.data = torch.randn(4, device = device)

    check_layer_gradients(layer, torch.randn(5, 3, size, size, device = device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "kernel_size, stride, size",
    [
        (2, 2, 8),
        (2, 2, 9),   # last row/col not covered by any window
        (3, 2, 9),   # overlapping windows
    ]
)
def test_maxpool(device, kernel_size, stride, size):
    layer = MaxPool(kernel_size, stride)
    input = torch.randn(5, 3, size, size, device = device)

    assert torch.allclose(layer.forward(input), F.max_pool2d(input, kernel_size, stride))
    check_layer_gradients(layer, input)


@pytest.mark.parametrize("device", DEVICES)
def test_gap(device):
    check_layer_gradients(GAP(), torch.randn(5, 3, 6, 6, device = device))


@pytest.mark.parametrize("device", DEVICES)
def test_reshape_layer(device):
    check_layer_gradients(ReshapeLayer(), torch.randn(5, 3, 6, 6, device = device))


@pytest.mark.parametrize("device", DEVICES)
def test_cross_entropy(device):
    loss_fn = CrossEntropy()
    preds = torch.randn(6, 10, device = device, requires_grad = True)
    targets = torch.randint(0, 10, (6,), device = device)

    loss = loss_fn.forward(preds, targets)
    loss.backward()

    assert torch.allclose(loss, F.cross_entropy(preds, targets))
    assert torch.allclose(loss_fn.backwards(), preds.grad, atol = ATOL, rtol = RTOL)


@pytest.mark.parametrize("device", DEVICES)
def test_mse(device):
    loss_fn = MSE()
    preds = torch.randn(6, 4, device = device, requires_grad = True)
    targets = torch.randn(6, 4, device = device)

    loss = loss_fn.forward(preds, targets)
    loss.backward()

    assert torch.allclose(loss, F.mse_loss(preds, targets))
    assert torch.allclose(loss_fn.backwards(), preds.grad, atol = ATOL, rtol = RTOL)


@pytest.mark.parametrize("device", DEVICES)
def test_full_network(device):
    '''
    End to end check with a small version of the train.py architecture:
    every parameter gradient after loss + model.backwards must match autograd.
    '''
    model = NeuralNetwork([
        ConvolutionalLayer(4, 1, 3, padding = 1),
        Tanh(),
        ConvolutionalLayer(4, 4, 3, stride = 2),
        ReLU(),
        MaxPool(2, 2),
        ReshapeLayer(),
        Dense(16, 36),
        Tanh(),
        Dense(10, 16)
    ]).to(device)
    loss_fn = CrossEntropy()

    input = torch.randn(8, 1, 14, 14, device = device)
    targets = torch.randint(0, 10, (8,), device = device)

    params = model.parameters()
    for p in params:
        p.data.requires_grad_()

    loss = loss_fn.forward(model.forward(input), targets)
    loss.backward()

    with torch.no_grad():
        model.backwards(loss_fn.backwards())

    for p in params:
        assert p.grad.shape == p.data.shape
        assert torch.allclose(p.grad, p.data.grad, atol = ATOL, rtol = RTOL)



@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs cuda")
def test_momentum_after_model_to_device():
    '''
    The optimizer can be built before model.to(device): velocity must follow
    the parameters to their new device.
    '''
    model = NeuralNetwork([Dense(3, 4)])
    optimizer = SGDWithMomentum(model.parameters(), lr = 0.1)
    model.to("cuda")

    loss_fn = CrossEntropy()
    loss_fn.forward(model.forward(torch.randn(2, 4, device = "cuda")), torch.tensor([0, 2], device = "cuda"))
    model.backwards(loss_fn.backwards())
    optimizer.step()

    for p in model.parameters():
        assert p.data.device.type == "cuda"
