import logging
import math
import os
import sys
import psutil  # New: To monitor resource usage
import time    # New: To measure runtime performance
from datetime import datetime

import numpy as np
import torch
from clip_norm import clip_grad_norm_
from torch.autograd import Variable

from fedscale.cloud.execution.torch_client import TorchClient


class Customized_Client(TorchClient):
    """
    Enhanced Client Component in Federated Learning
    Includes resource monitoring, dynamic differential privacy adjustment, and
    task allocation based on client resources.
    """

    def monitor_resources(self):
        """
        Monitors the system resource usage (CPU, memory, GPU).
        Logs the resource usage and returns a dictionary of current stats.
        """
        usage_stats = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory()._asdict(),
            'disk': psutil.disk_usage('/')._asdict(),
        }
        if torch.cuda.is_available():
            usage_stats['gpu'] = {
                'utilization': torch.cuda.utilization(),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_cached': torch.cuda.memory_reserved(),
            }
        logging.info(f"Resource usage: {usage_stats}")
        return usage_stats

    def dynamic_privacy_adjustment(self, conf, usage_stats):
        """
        Dynamically adjusts differential privacy noise levels based on resource usage.
        """
        noise_factor = conf.noise_factor
        memory_usage = usage_stats['memory']['percent']
        if memory_usage > 80:  # Adjust based on memory usage
            noise_factor *= 1.2  # Increase noise for high resource usage
            logging.warning("High memory usage detected. Increasing privacy noise.")
        elif memory_usage < 50:
            noise_factor *= 0.8  # Decrease noise for low resource usage
            logging.info("Low memory usage detected. Decreasing privacy noise.")
        conf.noise_factor = noise_factor
        logging.info(f"Dynamic noise factor adjusted to: {noise_factor}")
        return conf

    def task_allocation(self, client_data, conf):
        """
        Allocates tasks to the client based on available resources.
        """
        usage_stats = self.monitor_resources()
        if usage_stats['cpu_percent'] > 80:
            conf.batch_size = max(1, conf.batch_size // 2)
            logging.warning("High CPU usage detected. Reducing batch size.")
        elif usage_stats['memory']['percent'] > 70:
            conf.batch_size = max(1, conf.batch_size // 2)
            logging.warning("High memory usage detected. Reducing batch size.")
        else:
            conf.batch_size = min(conf.batch_size * 2, len(client_data.dataset))
            logging.info("Sufficient resources available. Increasing batch size.")
        logging.info(f"Adjusted batch size: {conf.batch_size}")
        return conf

    def train(self, client_data, model, conf):
        """
        Enhanced training loop with dynamic resource monitoring,
        privacy adjustment, and task allocation.
        """
        client_id = conf.client_id
        logging.info(f"Starting training (CLIENT: {client_id}) at {datetime.now()}...")

        tokenizer, device = conf.tokenizer, conf.device
        last_model_params = [p.data.clone() for p in model.parameters()]
        model = model.to(device=device)
        model.train()

        trained_unique_samples = min(
            len(client_data.dataset), conf.local_steps * conf.batch_size
        )
        self.global_model = None

        if conf.gradient_policy == 'fed-prox':
            self.global_model = [param.data.clone() for param in model.parameters()]

        optimizer = self.get_optimizer(model, conf)
        criterion = self.get_criterion(conf)
        error_type = None

        while self.completed_steps < conf.local_steps:
            try:
                # Monitor resources and adjust configurations dynamically
                usage_stats = self.monitor_resources()
                conf = self.dynamic_privacy_adjustment(conf, usage_stats)
                conf = self.task_allocation(client_data, conf)
                self.train_step(client_data, conf, model, optimizer, criterion)
            except Exception as ex:
                error_type = ex
                break

        delta_weight = []
        for param in model.parameters():
            delta_weight.append(
                (param.data.cpu() - last_model_params[len(delta_weight)])
            )

        clip_grad_norm_(delta_weight, max_norm=conf.clip_threshold)

        # Recover model weights
        idx = 0
        for param in model.parameters():
            param.data = last_model_params[idx] + delta_weight[idx]
            idx += 1

        sigma = conf.noise_factor * conf.clip_threshold
        state_dicts = model.state_dict()
        model_param = {
            p: np.asarray(
                state_dicts[p].data.cpu().numpy()
                + torch.normal(mean=0, std=sigma, size=state_dicts[p].data.shape)
                .cpu()
                .numpy()
            )
            for p in state_dicts
        }

        results = {
            "client_id": client_id,
            "moving_loss": self.epoch_train_loss,
            "trained_size": self.completed_steps * conf.batch_size,
            "success": self.completed_steps > 0,
        }
        results["utility"] = math.sqrt(self.loss_squared) * float(trained_unique_samples)

        if error_type is None:
            logging.info(
                f"Training of (CLIENT: {client_id}) completes at {datetime.now()}, {results}"
            )
        else:
            logging.info(
                f"Training of (CLIENT: {client_id}) failed at {datetime.now()} as {error_type}"
            )

        results["update_weight"] = model_param
        results["wall_duration"] = time.time()

        return results

