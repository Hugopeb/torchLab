import time
import torch
from myproject.utils.utils import get_batches, random_crop_and_flip


class Trainer:
    """
    Trainer encapsulates the training and evaluation logic for a neural network model.

    This class is responsible for:
    - Executing the training loop (forward, loss, backward, parameter update)
    - Executing the evaluation loop (forward + metric computation only)
    - Tracking timing and aggregate metrics per epoch
    - Coordinating logging and model artifact persistence
    """
    def __init__(self, model, loss_fn, logger, scheduler = None):
        self.model = model
        self.logger = logger
        self.loss_fn = loss_fn
        self.scheduler = scheduler

    def train_epoch(self, optimizer, train_data, train_targets, batch_size):
        """
        Run one full training epoch over the training dataset.

        This method:
        - Iterates over the dataset in mini-batches
        - Performs forward pass, loss computation, backward pass, and parameter updates
        - Accumulates the average cross-entropy loss over the epoch
        - Measures wall-clock training time

        Returns:
        - avg_epoch_CE: mean cross-entropy loss across all batches
        - epoch_time: elapsed time for the epoch (seconds)
        """
        num_batches = round(train_data.shape[0] / batch_size)
        start = time.perf_counter()
        total_epoch_CE = 0
        batch_index = 0


        for x_batch, y_batch in get_batches(train_data, train_targets, batch_size = batch_size):
            # x_batch = random_crop_and_flip(x_batch)
            output = self.model.forward(x_batch)
            avg_batch_CE = self.loss_fn.forward(output, y_batch)
            grad_input = self.loss_fn.backwards()
            self.model.backwards(grad_input)
            optimizer.step()
            total_epoch_CE += avg_batch_CE

            batch_index += 1
            if batch_index % 100 == 0:
                print(f"Finished batch: {batch_index} / {num_batches}")

        end = time.perf_counter()
        train_time = end - start
        avg_epoch_CE = total_epoch_CE / num_batches

        if self.scheduler is not None:
            self.scheduler.step(avg_epoch_CE)
        
        return avg_epoch_CE, train_time
    
    def eval_epoch(self, eval_data, eval_targets, batch_size):
        """
        Run one evaluation epoch on the validation dataset.

        This method:
        - Performs forward passes only (no gradient computation)
        - Computes classification accuracy using argmax over logits
        - Measures wall-clock evaluation time

        Returns:
        - accuracy: fraction of correctly classified samples
        - eval_time: elapsed time for evaluation (seconds)
        """
        start = time.perf_counter()

        total_samples = len(eval_targets) 
        total_correct = 0
        
        for x_batch, y_batch in get_batches(eval_data, eval_targets, batch_size=batch_size):
            output = self.model.forward(x_batch)
            
            preds = output.argmax(dim = 1)
            total_correct += (preds == y_batch).sum().item()

        end = time.perf_counter()

        accuracy = total_correct / total_samples
        eval_time = end - start

        return accuracy, eval_time


    def train_model(self, optimizer, train_data, train_targets, eval_data, eval_targets, num_epochs, batch_size = 64, eval = True):
        """
        Training loop coordinating training, evaluation, and logging.

        For each epoch:
        - Trains the model for one epoch
        - Logs training metrics and layer statistics
        - Evaluates the model on validation data
        - Logs evaluation metrics

        After training:
        - Persists model artifacts (filters, architecture, weights)

        This method acts as the main entry point for model training.
        """
        for epoch in range(num_epochs):
            avg_epoch_CE, train_time = self.train_epoch(
                optimizer = optimizer,
                train_data = train_data,
                train_targets = train_targets,
                batch_size = batch_size,
            )

            train_metrics = {
                "epoch": epoch,
                "avg_epoch_CE": avg_epoch_CE.item(),
                "train_time": train_time
            }

            if eval:
                accuracy, eval_time = self.eval_epoch(
                    eval_data = eval_data,
                    eval_targets = eval_targets,
                    batch_size = batch_size
                )

                train_metrics["accuracy"] = accuracy
                train_metrics["eval_time"] = eval_time

            print(train_metrics)

            self.logger.log_train_metrics(train_metrics)
            self.logger.log_stats(self.model, epoch)

        print(f"Logging in {self.logger.run_dir}")
        
        self.logger.log_model(self.model)
        self.logger.log_conv_filters(self.model)
        self.logger.log_architecture(self.model)

        self.logger.log_config(
            num_epochs, 
            batch_size, 
            optimizer,
        )
        
    def eval_model(self, eval_data, eval_targets, batch_size):
        '''
        Loads the model's accuracy and evaluation time without training.
        '''
        accuracy, eval_time = self.eval_epoch(
            eval_data = eval_data,
            eval_targets = eval_targets,
            batch_size = batch_size
        )
        print(f"Accuracy of the model: {accuracy: .4f}")
        print(f"Evaluation time: {eval_time: 4f} seconds")





            


        
    

