from myproject.model.activations import ConvolutionalLayer
import torch

class NeuralNetwork():
    """
    A simple modular neural network container.

    This class represents a sequential model that applies a list of layers
    in order. It supports forward pass, backward pass, parameter
    updates, and exporting the architecture configuration.

    Attributes:
        layers (list): List of layer objects that implement `forward`,
        `backwards`, and optionally `update_parameters` and `get_config`.

    Methods:
        forward(input): Runs the input through all layers sequentially.
        backwards(grad_input): Backpropagates gradients through layers.
        update_parameters(lr): Calls `update_parameters` on each layer if present.
        get_architecture(): Returns a list of layer configuration dictionaries.
        state_dict(): Returns a dictionary of layer weights and bias
        load_state_dict(state): Loads a set of weight and bias previously trained
        stats(epoch): Returns a list of layer weights/bias stats: mean, std, max, min
        train() / eval(): Switch layers like Dropout between training and evaluation
    """
    def __init__(self, architecture = []):
        self.layers = []
        for layer in architecture:
            self.layers.append(layer)

    def __call__(self, input):
        return self.forward(input)     

    def parameters(self):
        parameters = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                parameters += layer.parameters()
            else:
                pass

        return parameters

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backwards(self, grad_input):
        for layer in list(reversed(self.layers)):
            grad_input = layer.backwards(grad_input)

    def get_architecture(self):
        return [layer.get_config() for layer in self.layers]
    
    def state_dict(self):
        state = {}
        for i, layer in enumerate(self.layers):
            try:
                state[f"{type(layer).__name__}_{i}"] = layer.state_dict()
            except AttributeError:
                pass
        return state
    
    def load_state_dict(self, state):
        for i, layer in enumerate(self.layers):
            key = f"{type(layer).__name__}_{i}"
            if key in state:
                layer.load_state_dict(state[key])

    
    def stats(self, epoch):
        stats = []
        for i, layer in enumerate(self.layers):
            try:
                stats.append({
                    "epoch": epoch,
                    f"layer": f"{type(layer).__name__}_{i}",
                    "weights": layer.stats()[0],
                    "bias": layer.stats()[1]
                })

            except AttributeError:
                pass
        
        return stats
    
    def get_conv_layers(self):
        '''
        Returns a list of the ConvolutionalLayer instances inside the moodel.
        If there's no ConvolutionalLayer it returns an empty list.
        '''
        return [l for l in self.layers if isinstance(l, ConvolutionalLayer)]
    
    def get_conv_filters(self):
        """
        Returns the convolutional filter weights prepared for visualization.

        This method collects all convolutional layers in the network and extracts
        their kernel weights. For each convolutional layer, the kernels are reduced
        along the input-channel dimension (via averaging) to produce
        two-dimensional filter representations suitable for visualization.

        If there are no instances of ConvolutionalLayer it returns an empty dict

        Returns:
            dict: A dictionary mapping layer identifiers (e.g., "ConvolutionalLayer_0")
            to arrays of collapsed convolutional kernels shaped (N, 1, H, W)
        """

        state = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ConvolutionalLayer):
                state[f"ConvolutionalLayer_{i}"] = layer.weights.data.mean(dim = 1, keepdim = True)

        return state
    
    def train(self):
        '''
        Puts layers with train/eval behaviour (e.g. Dropout) in training mode.
        '''
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True
        return self

    def eval(self):
        '''
        Puts layers with train/eval behaviour (e.g. Dropout) in evaluation mode.
        '''
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False
        return self

    def to(self, device):
        for layer in self.layers:
            if hasattr(layer, "to"):
                layer.to(device)

        return self