import pytest
import torch
import torch.nn.functional as F

from myproject.model.activations import Dense, ReLU, Tanh, ConvolutionalLayer, ReshapeLayer, MaxPool, GAP, Dropout, BatchNorm
from myproject.model.neural_network import NeuralNetwork
from myproject.loss.loss_functions import CrossEntropy, MSE
from myproject.optimizer.optimizer import SGD, SGDWithMomentum
from myproject.utils.utils import random_crop_and_flip, Parameter

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
@pytest.mark.parametrize("shape", [(6, 4, 5, 5), (6, 4)])
@pytest.mark.parametrize("training", [True, False])
def test_batchnorm(device, shape, training):
    layer = BatchNorm(4).to(device)
    # Non trivial gamma, beta and running stats so every term of the gradient matters
    layer.weights.data = torch.randn(4, device = device)
    layer.bias.data = torch.randn(4, device = device)
    layer.running_mean = torch.randn(4, device = device)
    layer.running_var = torch.rand(4, device = device) + 0.5
    layer.training = training
    input = torch.randn(*shape, device = device) * 2 + 1

    ref = F.batch_norm(
        input, layer.running_mean.clone(), layer.running_var.clone(),
        layer.weights.data, layer.bias.data, training = training
    )
    assert torch.allclose(layer.forward(input), ref, atol = ATOL, rtol = RTOL)

    check_layer_gradients(layer, input)


@pytest.mark.parametrize("device", DEVICES)
def test_batchnorm_running_stats(device):
    layer = BatchNorm(3).to(device)
    running_mean = torch.zeros(3, device = device)
    running_var = torch.ones(3, device = device)

    for _ in range(3):
        input = torch.randn(8, 3, 4, 4, device = device) * 3 + 2
        layer.forward(input)
        F.batch_norm(input, running_mean, running_var, training = True, momentum = 0.1)

    assert torch.allclose(layer.running_mean, running_mean, atol = ATOL)
    assert torch.allclose(layer.running_var, running_var, atol = ATOL)


@pytest.mark.parametrize("device", DEVICES)
def test_dropout(device):
    layer = Dropout(p = 0.3)
    input = torch.randn(5, 3, 6, 6, device = device)

    # The random mask is drawn in forward, so reseed to get the same one in both passes
    torch.manual_seed(1)
    check_layer_gradients(layer, input)

    # Roughly p of the activations dropped, survivors scaled by 1 / (1 - p)
    output = layer.forward(torch.ones(1000, 100, device = device))
    kept = output != 0
    assert abs(1 - kept.float().mean().item() - 0.3) < 0.01
    assert torch.allclose(output[kept], torch.full_like(output[kept], 1 / 0.7))


@pytest.mark.parametrize("device", DEVICES)
def test_dropout_eval_is_identity(device):
    model = NeuralNetwork([Dense(4, 4), Dropout(p = 0.5)]).to(device)
    model.eval()
    input = torch.randn(3, 4, device = device)
    grad = torch.randn(3, 4, device = device)

    assert torch.equal(model.layers[1].forward(input), input)
    assert torch.equal(model.layers[1].backwards(grad), grad)

    model.train()
    assert model.layers[1].training


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
        BatchNorm(4),
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


@pytest.mark.parametrize("optimizer_cls", [SGD, SGDWithMomentum])
def test_weight_decay_matches_torch(optimizer_cls):
    '''
    Weight decay only on weights (ndim > 1), compared with torch.optim.SGD
    using one param group with decay for the weights and one without for the bias.
    '''
    weights = Parameter(torch.randn(4, 3))
    bias = Parameter(torch.randn(4))
    ref_w = weights.data.clone().requires_grad_()
    ref_b = bias.data.clone().requires_grad_()

    momentum = {"momentum": 0.9} if optimizer_cls is SGDWithMomentum else {}
    optimizer = optimizer_cls([weights, bias], lr = 0.1, weight_decay = 0.05, **momentum)
    ref = torch.optim.SGD(
        [{"params": [ref_w], "weight_decay": 0.05}, {"params": [ref_b], "weight_decay": 0.0}],
        lr = 0.1, **momentum
    )

    for _ in range(3):
        grad_w, grad_b = torch.randn(4, 3), torch.randn(4)
        weights.grad, bias.grad = grad_w, grad_b
        ref_w.grad, ref_b.grad = grad_w.clone(), grad_b.clone()
        optimizer.step()
        ref.step()

    # torch applies lr to the velocity instead of inside it; same trajectory for a constant lr
    assert torch.allclose(weights.data, ref_w.data, atol = ATOL)
    assert torch.allclose(bias.data, ref_b.data, atol = ATOL)


@pytest.mark.parametrize("device", DEVICES)
def test_random_crop_and_flip(device):
    batch = torch.randn(16, 3, 8, 8, device = device)
    padded = F.pad(batch, (4, 4, 4, 4), mode = "reflect")
    out = random_crop_and_flip(batch)

    assert out.shape == batch.shape
    # Every output image must be some 8x8 window of its padded input, possibly flipped
    for i in range(16):
        windows = padded[i].unfold(1, 8, 1).unfold(2, 8, 1).permute(1, 2, 0, 3, 4).reshape(-1, 3, 8, 8)
        candidates = torch.cat([windows, torch.flip(windows, dims = [3])])
        assert (candidates == out[i]).flatten(1).all(dim = 1).any()
