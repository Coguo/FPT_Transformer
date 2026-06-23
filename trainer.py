"""Training loop for FPT model.

Implements two-phase training from the paper:
  Phase 1 - FPT: Dice loss, SGD, cosine annealing, warmup (300 epochs)
  Phase 2 - IBR: BCE + CE, SGD, cosine annealing, warmup (200 epochs, optional)
"""

import os
import time
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models import FPTNet
from losses import FPTLoss, IBRLoss
from utils.metrics import compute_all_metrics
from utils.visualization import save_prediction_grid, plot_metrics_curves
from utils.logger import Logger


def _create_optimizer(model, config, phase="fpt"):
    """Create SGD optimizer with config parameters."""
    if phase == "fpt":
        lr = config["lr"]
        params = model.get_trainable_params(phase="fpt")
    else:
        lr = config["lr"]
        params = model.get_trainable_params(phase="ibr")

    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )


def _create_scheduler(optimizer, epochs, warmup_epochs, scheduler_type="cosine"):
    """Create learning rate scheduler.

    Cosine annealing from initial lr to 0.
    Warmup is handled separately inside the training loop.
    """
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=0
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs // 3, gamma=0.1
        )
    return scheduler


def _adjust_learning_rate(optimizer, epoch, warmup_epochs, base_lr):
    """Linear warmup: lr starts from 0 and linearly increases to base_lr."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    epoch, device, scaler=None, logger=None,
                    log_interval=20, use_amp=True):
    """Train for one epoch.

    Returns:
        average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        if logger and batch_idx % log_interval == 0:
            logger.log_scalar("train/batch_loss", loss.item())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / num_batches

    if scheduler is not None:
        # Cosine scheduler steps after warmup is done
        if epoch >= scheduler.last_epoch + 1:
            scheduler.step()

    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, device, compute_bf_flag=True):
    """Validate model on test set.

    Returns:
        dict with loss, iou, f1, bf
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_masks = []

    pbar = tqdm(loader, desc="Validation")
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        all_preds.append(logits.cpu())
        all_masks.append(masks.cpu())

    avg_loss = total_loss / len(loader)

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Compute metrics
    metrics = compute_all_metrics(all_preds, all_masks,
                                  compute_bf_flag=compute_bf_flag)
    metrics["loss"] = avg_loss

    return metrics


class Trainer:
    """Full trainer for FPT model with two-phase training."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            config["training"]["device"]
            if torch.cuda.is_available() and config["training"]["device"] == "cuda"
            else "cpu"
        )
        print(f"Using device: {self.device}")

        # Model
        self.model = FPTNet(
            num_classes=1,
            pretrained=config["model"]["pretrained"],
            decoder_type=config["model"]["decoder"]["type"],
        ).to(self.device)

        # Log model size
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters()
                          if p.requires_grad)
        print(f"Model: {n_params:,} total params, {n_trainable:,} trainable")

        # Logger
        self.logger = Logger(
            config["training"]["log_dir"],
            experiment_name=f"fpt_{config['model']['decoder']['type']}",
        )

        # AMP scaler
        self.scaler = GradScaler() if config["training"]["mixed_precision"] else None
        self.use_amp = config["training"]["mixed_precision"] and self.device.type == "cuda"

        # Metrics history
        self.train_metrics_history = []
        self.val_metrics_history = []

        # Checkpoint dir
        self.save_dir = config["training"]["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

    def train_fpt_phase(self, train_loader, val_loader):
        """Phase 1: FPT training with Dice loss.

        Matches the paper:
        - Dice Loss
        - SGD (momentum=0.9, weight_decay=0.0005, lr=0.003)
        - Cosine annealing (0.9 -> 0)
        - Warmup 10 epochs
        - 300 epochs
        """
        cfg = self.config["training"]["fpt"]
        epochs = cfg["epochs"]
        warmup = cfg["warmup_epochs"]

        criterion = FPTLoss(dice_weight=1.0, bce_weight=0.0).to(self.device)
        optimizer = _create_optimizer(self.model, cfg, phase="fpt")
        scheduler = _create_scheduler(
            optimizer, epochs, warmup, cfg["scheduler"]
        )

        print(f"\n{'='*50}")
        print(f"Phase 1: FPT Training")
        print(f"  Loss: Dice")
        print(f"  Optimizer: SGD (lr={cfg['lr']}, momentum={cfg['momentum']})")
        print(f"  Scheduler: Cosine (warmup {warmup} epochs)")
        print(f"  Epochs: {epochs}")
        print(f"{'='*50}\n")

        start_time = time.time()
        best_f1 = 0.0

        for epoch in range(1, epochs + 1):
            # Warmup
            _adjust_learning_rate(optimizer, epoch - 1, warmup, cfg["lr"])

            # Train
            train_loss = train_one_epoch(
                self.model, train_loader, criterion, optimizer,
                scheduler, epoch, self.device, self.scaler,
                self.logger, self.config["training"]["log_interval"],
                self.use_amp,
            )

            # Log current lr
            current_lr = optimizer.param_groups[0]["lr"]
            self.logger.log_scalar("train/lr", current_lr, epoch)

            # Validate every epoch
            val_metrics = validate(
                self.model, val_loader, criterion, self.device,
                compute_bf_flag=True,
            )

            # Log metrics
            self.logger.log_scalars({
                "train/loss": train_loss,
                "val/loss": val_metrics["loss"],
                "val/iou": val_metrics["iou"],
                "val/f1": val_metrics["f1"],
                "val/bf": val_metrics.get("bf", 0.0),
            }, step=epoch)

            # Save history
            self.train_metrics_history.append({"loss": train_loss})
            self.val_metrics_history.append(val_metrics)

            # Progress
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"IoU: {val_metrics['iou']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | "
                f"BF: {val_metrics.get('bf', 0.0):.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed:.0f}s"
            )

            # Save best model
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                self._save_checkpoint("best_fpt.pth", epoch, val_metrics)

        # Save final model
        self._save_checkpoint("final_fpt.pth", epochs, val_metrics)

        # Plot curves
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.save_dir, "fpt_training_curves.png"),
        )

        print(f"\nFPT Training Complete. Best F1: {best_f1:.4f}")

    def train_ibr_phase(self, train_loader, val_loader):
        """Phase 2: IBR fine-tuning (optional).

        Trains only the decoder head with BCE + CE loss.
        Requires direction ground truth in the dataloader.

        Paper config:
        - BCE + CE loss
        - SGD (momentum=0.9, weight_decay=0.0005, lr=0.001)
        - Cosine annealing, 20 epoch warmup
        - 200 epochs
        """
        cfg = self.config["training"]["ibr"]
        if not cfg["enabled"]:
            print("IBR phase disabled. Skipping.")
            return

        epochs = cfg["epochs"]
        warmup = cfg["warmup_epochs"]

        # Switch decoder to IBR mode if needed
        if self.config["model"]["decoder"]["type"] != "ibr":
            print(
                "Warning: decoder_type is not 'ibr', "
                "but IBR training requested. "
                "Skipping IBR phase."
            )
            return

        criterion = IBRLoss(
            bce_weight=cfg["bce_weight"],
            ce_weight=cfg["ce_weight"],
            dice_weight=1.0,
        ).to(self.device)

        optimizer = _create_optimizer(self.model, cfg, phase="ibr")
        scheduler = _create_scheduler(
            optimizer, epochs, warmup, cfg["scheduler"]
        )

        print(f"\n{'='*50}")
        print(f"Phase 2: IBR Fine-tuning")
        print(f"  Loss: BCE + CE")
        print(f"  Optimizer: SGD (lr={cfg['lr']})")
        print(f"  Epochs: {epochs}")
        print(f"{'='*50}\n")

        start_time = time.time()
        best_f1 = 0.0

        for epoch in range(1, epochs + 1):
            _adjust_learning_rate(optimizer, epoch - 1, warmup, cfg["lr"])

            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} [IBR]")

            for batch in pbar:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                boundaries = batch["boundary"].to(self.device)
                directions = batch["direction"].to(self.device)

                optimizer.zero_grad()

                seg_pred, boundary_pred, direction_pred = self.model(images)
                losses = criterion(
                    (seg_pred, boundary_pred, direction_pred),
                    (masks, boundaries, directions),
                )

                losses["total"].backward()
                optimizer.step()

                running_loss += losses["total"].item()
                pbar.set_postfix({"loss": f"{losses['total'].item():.4f}"})

            # Validate
            self.model.eval()
            val_loss = 0.0
            all_preds, all_masks = [], []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device)
                    boundaries = batch["boundary"].to(self.device)
                    directions = batch["direction"].to(self.device)

                    seg_pred, boundary_pred, direction_pred = self.model(images)
                    losses = criterion(
                        (seg_pred, boundary_pred, direction_pred),
                        (masks, boundaries, directions),
                    )
                    val_loss += losses["total"].item()
                    all_preds.append(seg_pred.cpu())
                    all_masks.append(masks.cpu())

            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            all_preds = torch.cat(all_preds, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            metrics = compute_all_metrics(all_preds, all_masks)

            elapsed = time.time() - start_time
            print(
                f"IBR Epoch {epoch}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"IoU: {metrics['iou']:.4f} | F1: {metrics['f1']:.4f} | "
                f"Time: {elapsed:.0f}s"
            )

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                self._save_checkpoint("best_ibr.pth", epoch, metrics)

        self._save_checkpoint("final_ibr.pth", epochs, metrics)
        print(f"IBR Training Complete. Best F1: {best_f1:.4f}")

    def _save_checkpoint(self, filename, epoch, metrics):
        """Save model checkpoint."""
        path = os.path.join(self.save_dir, filename)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint

    def close(self):
        self.logger.close()