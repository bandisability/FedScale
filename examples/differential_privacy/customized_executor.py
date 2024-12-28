# -*- coding: utf-8 -*-

import os
import sys
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from customized_client import Customized_Client
from fedscale.cloud.execution.executor import Executor
import fedscale.cloud.config_parser as parser


class Customized_Executor(Executor):
    """Customized Executor for managing task execution with advanced features."""

    def __init__(self, args):
        super(Customized_Executor, self).__init__(args)
        self.logger = logging.getLogger("CustomizedExecutor")
        self.task_pool = ThreadPoolExecutor(max_workers=self.args.max_concurrent_tasks)
        self.log_task_details()
        self.task_execution_log = {}

    def get_client_trainer(self, conf):
        return Customized_Client(conf)

    def log_task_details(self):
        """Logs task execution details to identify performance bottlenecks."""
        log_path = os.path.join(self.args.log_path, "executor_task_log.txt")
        logging.basicConfig(
            filename=log_path, level=logging.INFO, format="%(asctime)s - %(message)s"
        )
        self.logger.info("Task logging initialized at %s", log_path)

    def execute_task(self, task):
        """Executes a task and records its execution details."""
        start_time = time.time()
        self.logger.info("Starting task execution: %s", task)
        try:
            result = super().execute_task(task)
            self.logger.info("Task %s executed successfully.", task)
        except Exception as e:
            self.logger.error("Task %s failed with error: %s", task, str(e))
            result = None
        end_time = time.time()
        self.log_task_execution(task, start_time, end_time)
        return result

    def log_task_execution(self, task, start_time, end_time):
        """Logs execution time and details of the task."""
        execution_time = end_time - start_time
        self.task_execution_log[task] = execution_time
        self.logger.info(
            "Task %s completed in %.2f seconds.", task, execution_time
        )

    def execute_tasks_concurrently(self, tasks):
        """Executes multiple tasks concurrently using ThreadPoolExecutor."""
        futures = {self.task_pool.submit(self.execute_task, task): task for task in tasks}
        results = []
        for future in as_completed(futures):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                self.logger.error("Task %s raised an exception: %s", task, str(e))
        return results

    def log_privacy_details(self, task, noise_budget, noise_distribution):
        """Logs detailed differential privacy information."""
        self.logger.info(
            "Task %s Privacy Details: Budget=%s, Distribution=%s",
            task,
            noise_budget,
            noise_distribution,
        )

    def run(self):
        """Overriding the default run method to include extended logging and privacy details."""
        self.logger.info("Customized Executor started.")
        tasks = self.load_tasks()
        self.logger.info("Loaded %d tasks for execution.", len(tasks))
        results = self.execute_tasks_concurrently(tasks)
        self.logger.info("Completed all tasks. Results: %s", results)
        self.logger.info("Executor shutting down.")

    def load_tasks(self):
        """Loads tasks and assigns necessary privacy budgets and noise distribution."""
        tasks = super().load_tasks()
        for task in tasks:
            # Assign privacy budget and noise distribution
            noise_budget = self.calculate_privacy_budget(task)
            noise_distribution = self.assign_noise_distribution(task)
            self.log_privacy_details(task, noise_budget, noise_distribution)
        return tasks

    def calculate_privacy_budget(self, task):
        """Calculates the differential privacy budget for a task."""
        return 1.0 / (1 + task["priority"])  # Example: Lower priority => higher noise

    def assign_noise_distribution(self, task):
        """Assigns the noise distribution for differential privacy."""
        return "Gaussian" if task["priority"] < 5 else "Laplace"


if __name__ == "__main__":
    executor = Customized_Executor(parser.args)
    executor.run()


