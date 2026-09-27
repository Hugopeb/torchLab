import os
import json
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from myproject.model.neural_network import NeuralNetwork
from myproject.model.activations import ReshapeLayer
from myproject.model.activations import ConvolutionalLayer, Dense, Tanh, ReLU, MaxPool, GAP
from myproject.config import RUNS_DIR
from myproject.utils.utils import get_timestamp


class Logger:
    '''
    Utility class for managing experiment logging and artifact persistence.

    This class creates a uniquely timestamped run directory and provides
    methods to save model-related artifacts, including the model architecture,
    trained weights, and experiment configuration. 
    Directory structure:
        run_<timestamp>/
            ├── architecture.json
            ├── weights.pth
            ├──  config.json
            └── metrics
                ├──ConvolutionalLayers.png
                └──stats.jsonl
    '''
    def __init__(self):
        self.timestamp = get_timestamp()

        self.run_dir = os.path.join(
            RUNS_DIR, 
            f"run_{self.timestamp}"
        )

        self.metrics_dir = os.path.join(
            self.run_dir,
            "metrics"
        )

        os.makedirs(self.run_dir, exist_ok = True)
        os.makedirs(self.metrics_dir, exist_ok = True)

    def log_architecture(self, model):
        '''
        Takes the model as argument and saves the model's 
        architecture inside run_dir as architecture.json
        
        '''
        path = os.path.join(self.run_dir, "architecture.json")
        architecture = model.get_architecture()

        with open(path, "w") as f:
            json.dump(architecture, f, indent=2)

    def log_config(self, num_epochs, batch_size, optimizer):
        '''
        Saves the run configuration: num_epochs, batch_size
        and learning_rate, (possible random seed in future)
        '''
        opt_config = optimizer.get_config()

        config = {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": opt_config["lr"],
            "optimizer": opt_config["type"],
        }

        if "momentum" in opt_config:
            config["momentum"] = opt_config["momentum"]

        path = os.path.join(self.run_dir, "config.json")

        with open(path, "w") as f:
            json.dump(config, f, indent = 2)

    def log_model(self, model):
        '''
        Save the final weights and bias (state_dict) for each trainable
        layer of the net as a .pth file. 
        '''
        path = os.path.join(self.run_dir, "weights.pth")
        state_dict = model.state_dict()

        torch.save(state_dict, path)

    def log_stats(self, model, epoch):
        '''
        Saves the mean, std, max and min values of the weights and the
        bias for each layer at a given epoch as stats.jsonl.
        '''
        path = os.path.join(self.metrics_dir, "stats.jsonl")
        stats = model.stats(epoch)

        with open(path, "a") as f:
            for state in stats:
                f.write(json.dumps(state) + "\n")


    def log_conv_filters(self, model):
        """
        Save visualization of convolutional filters for each conv layer.

        Assumptions:
            model.get_conv_filters() returns a dict where:
            key: str, layer name (e.g., "ConvolutionalLayer_0")
            value: torch.Tensor of shape (N, 1, H, W) representing N filters

        Output:
            Saves one PNG per layer under self.metrics_dir.
        """
        state = model.get_conv_filters()

        if not state:
            print("No convolutional layers found.")
            return

        for key in state.keys():
            filters = state[key]         

            grid = make_grid(filters, nrow=4, normalize=True, padding=1)

            plt.figure(figsize=(6, 6))
            plt.imshow(grid.detach().cpu().permute(1, 2, 0).numpy())
            plt.axis("off")
            plt.title(key)
            plt.savefig(os.path.join(self.metrics_dir, f"{key}.png"))
            plt.close()

    def log_train_metrics(self, train_metrics):
        '''
        Logs train metrics (epoch, avg_epoch_CE, train_time) 
        for each epoch of the training loop. If eval = True during 
        training it also loads (accuracy, eval_time) over the test
        dataset.
        
        train_metrics: Dictionary of the train metrics listed above.
        '''
        path = os.path.join(self.metrics_dir, "train_metrics.jsonl")

        with open(path, "a") as f:
            f.write(json.dumps(train_metrics) + "\n")

            

def build_NeuralNetwork(model_dir):
    '''
    Reconstructs a neural network model from saved architecture and weights.

    This function loads a serialized model architecture definition
    (`architecture.json`) and corresponding trained parameters
    (`weights.pth`) from a specified model directory. It dynamically
    instantiates each layer based on the architecture configuration,
    assembles them into a NeuralNetwork object, and restores the model's
    learned state.

    Args:
        model_dir (str or Path):
            Name of the model run directory located under RUNS_DIR.

    Returns:
        NeuralNetwork:
            A fully constructed neural network with learned weights loaded.

    Raises:
        ValueError:
            If an unknown layer type is encountered in the architecture
            configuration.
        FileNotFoundError:
            If the architecture or weight files are missing.
    '''
    folder_path = RUNS_DIR / model_dir
    json_path = folder_path / "architecture.json"
    state_path = folder_path / "weights.pth"

    # Load on CPU so weights saved on cuda can be restored anywhere, use model.to(device) after
    state = torch.load(state_path, map_location = "cpu")

    with open(json_path, "r") as f:
        config = json.load(f)

    layers = []

    for layer_cfg in config:
        layer_type = layer_cfg["type"]

        if layer_type == "ConvolutionalLayer":
            layers.append(
                ConvolutionalLayer(
                    output_channels = layer_cfg["output_channels"],
                    input_channels = layer_cfg["input_channels"],
                    kernel_size = layer_cfg["kernel_size"],
                    # Runs saved before stride/padding were logged used the defaults
                    stride = layer_cfg.get("stride", 1),
                    padding = layer_cfg.get("padding", 0)
                )
            )

        elif layer_type == "ReshapeLayer":
            layers.append(ReshapeLayer())

        elif layer_type == "Dense":
            layers.append(
                Dense(
                    output_size = layer_cfg["output_size"],
                    input_size = layer_cfg["input_size"]
                )
            )

        elif layer_type == "ReLU":
            layers.append(ReLU())

        elif layer_type == "Tanh":
            layers.append(Tanh())

        elif layer_type == "MaxPool":
            layers.append(
                MaxPool(
                    kernel_size = layer_cfg["kernel_size"],
                    stride = layer_cfg["stride"]
                )
            )

        elif layer_type == "GAP":
            layers.append(GAP())

        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
    
    model = NeuralNetwork(layers)

    model.load_state_dict(state)

    return model


