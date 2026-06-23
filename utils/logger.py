"""Logging utilities for training."""

import os
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Training logger with TensorBoard support.

    Logs scalars, histograms, images, and metrics
    to both console and TensorBoard.
    """

    def __init__(self, log_dir, experiment_name="fpt_experiment"):
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0

    def log_scalar(self, tag, value, step=None):
        """Log a scalar value."""
        if step is None:
            step = self.global_step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, scalars, step=None):
        """Log multiple scalars.
        Args:
            scalars: dict of {tag: value}
        """
        for tag, value in scalars.items():
            self.log_scalar(tag, value, step)

    def log_images(self, tag, images, step=None):
        """Log images.
        Args:
            images: torch tensor (N, C, H, W) in [0, 1]
        """
        if step is None:
            step = self.global_step
        self.writer.add_images(tag, images, step)

    def log_histogram(self, tag, values, step=None):
        """Log a histogram of values."""
        if step is None:
            step = self.global_step
        self.writer.add_histogram(tag, values, step)

    def step(self):
        """Increment global step counter."""
        self.global_step += 1

    def close(self):
        self.writer.close()