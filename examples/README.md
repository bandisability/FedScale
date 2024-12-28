# Enhancements and Optimizations for `clip_norm.py`

## Introduction

This document describes the enhancements and optimizations implemented in the `clip_norm.py` file for improving gradient clipping efficiency in distributed and large-scale federated learning setups. The updated features ensure improved scalability, performance, and usability.

## Key Enhancements

### 1. Distributed Gradient Clipping
- **Description**: Introduced distributed gradient clipping to handle large-scale federated learning scenarios efficiently.
- **Benefit**: Reduces communication overhead and enhances performance when training on multiple nodes.

### 2. Dynamic Gradient Clipping Thresholds
- **Description**: Added support for dynamic adjustment of gradient clipping thresholds based on training performance metrics.
- **Benefit**: Prevents over-clipping, ensuring better model convergence and performance.

### 3. Enhanced Logging and Monitoring
- **Description**: Integrated detailed logging to track gradient norms, clipping coefficients, and threshold adjustments.
- **Benefit**: Provides real-time insights into the gradient clipping process for debugging and monitoring.

### 4. Robust Error Handling
- **Description**: Implemented robust handling for non-finite gradient norms (e.g., NaN, Inf) with user-configurable error handling.
- **Benefit**: Ensures stable training in the presence of numerical instabilities.

### 5. Centralized and Distributed Support
- **Description**: Updated the code to handle both centralized single-node and distributed multi-node scenarios.
- **Benefit**: Provides flexibility for users to deploy the algorithm in various setups.

### 6. Refactored Code
- **Description**: Improved readability and maintainability with clear type annotations, detailed comments, and structured functions.
- **Benefit**: Simplifies future modifications and enhances code quality.

## Example Usage

Below is an example demonstrating the usage of the updated `clip_grad_norm_` function with dynamic threshold adjustment:

```python
from clip_norm import clip_grad_norm_

# Example usage for dynamic gradient clipping
parameters = model.parameters()
dynamic_threshold = calculate_dynamic_threshold(metrics)
clip_grad_norm_(
    parameters,
    max_norm=dynamic_threshold,
    norm_type=2.0,
    error_if_nonfinite=True
)

# Enhancements and Optimizations for `customized_client.py`

## Key Features Added

### 1. Client Resource Monitoring
- **Description**: Tracks CPU, memory, and GPU utilization during training.
- **Benefit**: Ensures the training process adapts to client resource constraints.

### 2. Dynamic Differential Privacy Adjustment
- **Description**: Adjusts the differential privacy noise level dynamically based on resource usage.
- **Benefit**: Balances privacy and performance in resource-constrained environments.

### 3. Adaptive Task Allocation
- **Description**: Modifies batch size and local steps dynamically based on available resources.
- **Benefit**: Ensures efficient resource utilization and avoids overloading the client.

### 4. Enhanced Logging
- **Description**: Logs resource usage, batch size adjustments, and training progress in detail.
- **Benefit**: Provides visibility into the training process for debugging and monitoring.

## Usage Example

```python
# Example usage of the enhanced Customized_Client
client = Customized_Client()
client_data = load_client_data()
model = load_model()
conf = load_config()

results = client.train(client_data, model, conf)
print("Training Results:", results)

## Customized Executor Enhancements

### Features Added
1. **Asynchronous Task Execution**:
   - Introduced `ThreadPoolExecutor` to enable concurrent task execution.
   - Tasks are now executed in parallel, significantly improving task throughput and resource utilization.

2. **Execution Logging**:
   - Logs detailed information for each task, including:
     - Start and end times
     - Execution duration
     - Errors encountered
   - Logs are saved in `executor_task_log.txt` for performance monitoring and debugging.

3. **Differential Privacy Logging**:
   - Added detailed logs for differential privacy computations, including:
     - Privacy budget per task
     - Noise distribution used
   - Logs provide transparency for privacy configurations and enable diagnostic analysis.

4. **Dynamic Task Scheduling and Resource Optimization**:
   - Dynamically calculates differential privacy budgets based on task priority.
   - Assigns noise distributions (e.g., Gaussian, Laplace) dynamically to enhance privacy protection.
   - Tasks with lower priorities are assigned higher noise for better privacy protection.

5. **Performance Bottleneck Analysis**:
   - Logs task execution times to help identify and analyze performance bottlenecks.
   - Enables informed decisions for further optimization.

---

### Example Log Output
The executor generates the following log details during execution:

# Building Your Own FL Algorithm with FedScale

You can use FedScale to implement your own FL algorithm(s), for optimization, client selection, etc.
In this tutorial, we focus on the API you need to implement an FL algorithm.
Implementations for several existing FL algorithms are included as well.

Check the [instructions](../README.md) to set up your environment and [instructions](../benchmark/dataset/README.md) to download datasets.


## Algorithm API
Federated algorithms have four main components in most cases:

- A server-to-client broadcast step;
- A local client update step;
- A client-to-server upload step; and
- A server-side aggregation step.

To modify each of these steps for your FL algorithm, you can customize your server and client respectively.
Here we provide several examples to cover different components in FedScale.

## Aggregation Algorithm

FedScale uses Federated averaging as the default aggregation algorithm.
[FedAvg](https://arxiv.org/abs/1602.05629) is a communication efficient algorithm, where clients keep their data locally for privacy protection; a central parameter server is used to communicate between clients.
Each participant locally performs E epochs of stochastic gradient descent (SGD) during every round.
The participants then communicate their model updates to the central server, where they are averaged.

The aggregation algorithm in FedScale is mainly reflected in two code segments.

1. **Client updates**: FedScale calls `training_handler` in [cloud/execution/executor.py](../fedscale/cloud/execution/executor.py) to initiate client training.
The following code segment from [cloud/execution/clients/client.py](../fedscale/cloud/execution/clients/client.py) shows how the client trains the model and updates the gradient (when implementing FedProx).


```
class Client(object):
   """Basic client component in Federated Learning"""
   def __init__(self, conf):
       self.optimizer = ClientOptimizer()
       ...
      
   def train(self, client_data, model, conf):
       # Prepare for training
       ...
       # Conduct local training
       while completed_steps < conf.local_steps:
           try:
               for data_pair in client_data:
                   # Forward Pass
                   ...
                  
                   optimizer.zero_grad()
                   loss.backward()
                   optimizer.step()
                   self.optimizer.update_client_weight(conf, model, global_model if global_model is not None else None  )

                   completed_steps += 1
                   if completed_steps == conf.local_steps:
                       break
                
       # Collect training results
       return results
```
```
class ClientOptimizer(object):
   def __init__(self, sample_seed):
       pass
  
   def update_client_weight(self, conf, model, global_model = None):
       if conf.gradient_policy == 'fed-prox':
           for idx, param in enumerate(model.parameters()):
               param.data += conf.learning_rate * conf.proxy_mu * (param.data - global_model[idx])

```

2. **Server aggregates**: In the server-side, FedScale calls `round_weight_handler` in [cloud/aggregation/aggregator.py](../fedscale/cloud/aggregation/aggregator.py) to do the aggregation at the end of each round.
In the function `round_weight_handler`, you can customize your aggregator optimizer in [cloud/aggregation/optimizers.py](../fedscale/cloud/optimizers.py).
The following code segment shows how FedYoGi and FedAvg aggregate the participant gradients.

```
class TorchServerOptimizer(object):

   def __init__(self, mode, args, device, sample_seed=233):
       self.mode = mode
       if mode == 'fed-yogi':
           from utils.yogi import YoGi
           self.gradient_controller = YoGi(eta=args.yogi_eta, tau=args.yogi_tau, beta=args.yogi_beta, beta2=args.yogi_beta2)
      
       ...
      
   def update_round_gradient(self, last_model, current_model, target_model):
      
       if self.mode == 'fed-yogi':
           """
           "Adaptive Federated Optimizations",
           Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konecný, Sanjiv Kumar, H. Brendan McMahan,
           ICLR 2021.
           """
           last_model = [x.to(device=self.device) for x in last_model]
           current_model = [x.to(device=self.device) for x in current_model]

           diff_weight = self.gradient_controller.update([pb-pa for pa, pb in zip(last_model, current_model)])

           for idx, param in enumerate(target_model.parameters()):
               param.data = last_model[idx] + diff_weight[idx]

       elif self.mode == 'fed-avg':
           # The default optimizer, FedAvg, has been applied in aggregator.py on the fly
           pass
```

## Client Selection

FedScale uses random selection among all available clients by default.
However, you can customize the client selector by modifying the `client_manager` in [cloud/aggregation/aggregator.py](../fedscale/cloud/aggregation/aggregator.py),
which is defined in [/cloud/client_manager.py](../fedscale/cloud/client_manager.py).

Upon every device checking in or reporting results, FedScale aggregator calls `client_manager.registerClient(...)` or `client_manager.registerScore(...)` to record the necessary client information that could help you with the selection decision.
At the beginning of the round, FedScale aggregator calls `client_manager.resampleClients(...)` to select the training participants.

For example, [Oort](https://www.usenix.org/conference/osdi21/presentation/lai) is a client selector
that considers both statistical and system utility to improve the model time-to-accuracy performance.
You can find more details of Oort implementation in [../thirdparty/oort/oort.py](../thirdparty/oort/oort.py) and [../fedscale/cloud/client_manager.py](../fedscale/cloud/client_manager.py).

## Other Examples

You can find more FL algorithm examples in this directory, most of which involve simply customizing the `cloud/aggregation/aggregator.py`, `cloud/execution/executor.py`, and/or `cloud/execution/clients/client.py`. 
