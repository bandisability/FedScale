# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import threading
import logging
from concurrent.futures import ThreadPoolExecutor

from customized_client import Customized_Client
from fedscale.cloud.execution.executor import Executor
import fedscale.cloud.config_parser as parser

class Customized_Executor(Executor):
    """
    Enhanced Executor for Federated Learning with priority queue, 
    asynchronous task execution, and detailed logging.
    """

    def __init__(self, args):
        super(Customized_Executor, self).__init__(args)
        self.task_queue = asyncio.PriorityQueue()
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.logger = logging.getLogger("CustomizedExecutor")
        self.logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    def get_client_trainer(self, conf):
        return Customized_Client(conf)

    async def submit_task(self, priority, task, *args):
        """Submits a task to the priority queue."""
        await self.task_queue.put((priority, task, args))
        self.logger.info(f"Task submitted with priority {priority}")

    async def execute_tasks(self):
        """Executes tasks asynchronously based on priority."""
        while not self.task_queue.empty():
            priority, task, args = await self.task_queue.get()
            self.logger.info(f"Executing task with priority {priority}")
            try:
                await task(*args)
            except Exception as e:
                self.logger.error(f"Task execution failed: {e}")
            finally:
                self.task_queue.task_done()

    def monitor_performance(self):
        """Monitors system performance and logs resource usage."""
        try:
            import psutil
            while True:
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                self.logger.info(
                    f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_info.percent}%"
                )
        except ImportError:
            self.logger.warning("psutil module not installed. Skipping performance monitoring.")

    def log_task_performance(self, task_name, start_time, end_time):
        """Logs task performance details."""
        duration = end_time - start_time
        self.logger.info(f"Task {task_name} completed in {duration:.2f} seconds")

    def handle_task_failure(self, task_name, reason):
        """Handles task failure and logs the details."""
        self.logger.error(f"Task {task_name} failed due to: {reason}")

    def dashboard_api(self):
        """Simulates a RESTful API for accessing task progress."""
        self.logger.info("Simulating RESTful API for task monitoring...")

    def run(self):
        """Overrides the default run method to include enhanced functionalities."""
        self.logger.info("Customized Executor starting...")
        # Start a performance monitoring thread
        monitoring_thread = threading.Thread(target=self.monitor_performance, daemon=True)
        monitoring_thread.start()
        # Example task submission
        asyncio.run(self.submit_task(1, self.example_task, "High Priority Task"))
        asyncio.run(self.submit_task(2, self.example_task, "Low Priority Task"))
        asyncio.run(self.execute_tasks())
        self.logger.info("Customized Executor shutting down.")

    async def example_task(self, task_name):
        """An example task to demonstrate asynchronous execution."""
        self.logger.info(f"Task {task_name} started.")
        start_time = asyncio.get_event_loop().time()
        await asyncio.sleep(2)  # Simulate task duration
        end_time = asyncio.get_event_loop().time()
        self.log_task_performance(task_name, start_time, end_time)

if __name__ == "__main__":
    executor = Customized_Executor(parser.args)
    executor.run()
