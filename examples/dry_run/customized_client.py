import logging
import math
import os
import sys
import psutil  # New dependency for resource monitoring
from datetime import datetime

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from fedscale.cloud.execution.torch_client import TorchClient


class Customized_Client(TorchClient):
    """Enhanced client component in Federated Learning with resource-aware and adaptive training."""

    def train(self, client_data, model, conf):
        """Enhanced training function with resource grading and failure recovery."""

        # Initialize device and configurations
        device = conf.cuda_device if conf.use_cuda else torch.device('cpu')
        client_id = conf.client_id

        logging.info(f"[CLIENT {client_id}] Training started at {datetime.now()}")
        model = model.to(device=device)
        model.train()

        trained_unique_samples = min(len(client_data.dataset), conf.local_steps * conf.batch_size)

        optimizer = SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)
        criterion = CrossEntropyLoss().to(device=device)

        epoch_train_loss = 1e-4
        completed_steps = 0

        # Resource-aware batch size and local step adjustments
        batch_size = self.adjust_batch_size(conf.batch_size)
        local_steps = self.adjust_local_steps(conf.local_steps)

        logging.info(f"[CLIENT {client_id}] Adjusted batch size: {batch_size}, Adjusted local steps: {local_steps}")

        # Initialize failure recovery mechanism
        snapshot_path = f"model_snapshot_client_{client_id}.pt"
        if os.path.exists(snapshot_path):
            model.load_state_dict(torch.load(snapshot_path))
            logging.info(f"[CLIENT {client_id}] Resumed from snapshot.")

        # Training loop
        while completed_steps < local_steps:
            try:
                data, target = self.get_training_data(client_data, batch_size, device)
                output = model(data)
                loss = criterion(output, target)

                epoch_train_loss = (1. - conf.loss_decay) * epoch_train_loss + conf.loss_decay * loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                completed_steps += 1

                # Save snapshot for recovery
                if completed_steps % 10 == 0:
                    torch.save(model.state_dict(), snapshot_path)
                    logging.info(f"[CLIENT {client_id}] Snapshot saved after {completed_steps} steps.")

            except Exception as ex:
                logging.error(f"[CLIENT {client_id}] Error during training: {ex}")
                break

        # Clean up snapshot after successful training
        if os.path.exists(snapshot_path):
            os.remove(snapshot_path)

        # Collect results
        results = self.collect_results(model, client_id, epoch_train_loss, completed_steps, trained_unique_samples)

        logging.info(f"[CLIENT {client_id}] Training completed. Results: {results}")

        return results

    def adjust_batch_size(self, batch_size):
        """Adjust batch size based on system memory availability."""
        memory_info = psutil.virtual_memory()
        if memory_info.available < 2 * 1024**3:  # Less than 2GB available
            return max(1, batch_size // 2)
        return batch_size

    def adjust_local_steps(self, local_steps):
        """Adjust local steps based on CPU usage."""
        cpu_usage = psutil.cpu_percent()
        if cpu_usage > 80:  # High CPU usage
            return max(1, local_steps // 2)
        return local_steps

    def get_training_data(self, client_data, batch_size, device):
        """Simulate getting training data."""
        data = torch.rand(batch_size, 3, 256, 256, device=device)
        target = torch.randint(0, 10, (batch_size,), device=device)
        return data, target

    def collect_results(self, model, client_id, epoch_train_loss, completed_steps, trained_unique_samples):
        """Collect training results and prepare for return."""
        state_dicts = model.state_dict()
        model_param = {p: state_dicts[p].data.cpu().numpy() for p in state_dicts}

        results = {
            'client_id': client_id,
            'moving_loss': epoch_train_loss,
            'trained_size': completed_steps * trained_unique_samples,
            'success': completed_steps > 0,
            'update_weight': model_param,
            'utility': math.sqrt(epoch_train_loss) * float(trained_unique_samples),
            'wall_duration': 0
        }

        return results

    def monitor_system_resources(self):
        """Monitor system resource usage and log periodically."""
        memory_info = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent()
        logging.info(f"System Memory Usage: {memory_info.percent}%, CPU Usage: {cpu_usage}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Simulate configuration and client data for testing
    class Config:
        def __init__(self):
            self.cuda_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.use_cuda = torch.cuda.is_available()
            self.client_id = 1
            self.local_steps = 50
            self.batch_size = 32
            self.learning_rate = 0.01
            self.loss_decay = 0.9

    client_data = None  # Replace with actual client dataset
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 128 * 128, 10)
    )

    conf = Config()
    client = Customized_Client()
    results = client.train(client_data, model, conf)
    print("Training Results:", results)


