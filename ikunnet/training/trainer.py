"""Trainer for contrastive learning on mask embeddings."""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ikunnet.models.encoder import CNNEncoder
from ikunnet.models.projection_head import ProjectionHead
from ikunnet.models.vicreg_loss import VICRegLoss
from ikunnet.training.dataset import MaskDataset
from ikunnet.training.augmentation import MaskTransform


class Trainer:
    """
    Trainer for mask embedding representation learning with VICReg.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_head: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        variance_weight: float = 1.0,
        covariance_weight: float = 1.0,
        std_target: float = 1.0,
        device: str = 'cuda',
        save_dir: str = 'checkpoints'
    ):
        """
        Args:
            encoder: CNN encoder
            projection_head: Projection head
            train_loader: Training dataloader
            val_loader: Optional validation dataloader
            lr: Learning rate
            weight_decay: Weight decay for Adam
            variance_weight: Weight for variance loss term
            covariance_weight: Weight for covariance loss term
            std_target: Target standard deviation for VICReg
            device: 'cuda' or 'cpu'
            save_dir: Directory to save checkpoints
        """
        self.encoder = encoder.to(device)
        self.projection_head = projection_head.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # VICReg loss function
        self.criterion = VICRegLoss(
            variance_weight=variance_weight,
            covariance_weight=covariance_weight,
            std_target=std_target
        )

        # Optimizer (only for encoder and projection_head)
        self.optimizer = optim.Adam(
            list(encoder.parameters()) + list(projection_head.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 100  # Assume 100 epochs
        )

        # Mixed precision training
        self.scaler = GradScaler() if device == 'cuda' else None

        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.console = Console()

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            avg_loss: Average loss for this epoch
        """
        self.encoder.train()
        self.projection_head.train()

        total_loss = 0.0
        num_batches = 0

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )

        with progress:
            task = progress.add_task(f"Epoch {epoch}", total=len(self.train_loader))

            for batch in self.train_loader:
                # batch is (masks,) for VICReg
                masks = batch

                # Skip empty batches (can happen in online mode)
                if masks.size(0) == 0:
                    progress.update(task, advance=1)
                    continue

                masks = masks.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass with mixed precision
                if self.scaler is not None:
                    with autocast():
                        # Encode
                        h = self.encoder(masks)

                        # Project to embedding space
                        z = self.projection_head(h)

                        # Compute VICReg loss
                        loss = self.criterion(z)

                    # Backward pass with mixed precision
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Encode
                    h = self.encoder(masks)

                    # Project
                    z = self.projection_head(h)

                    # Loss
                    loss = self.criterion(z)

                    # Backward
                    loss.backward()
                    self.optimizer.step()

                # Update learning rate
                self.scheduler.step()

                # Track metrics
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                # Update progress bar with batch size
                batch_size = masks.size(0)
                progress.update(task, advance=1, description=f"Epoch {epoch} | B: {batch_size} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches

        return avg_loss

    def validate(self) -> float:
        """
        Validate the model.

        Returns:
            avg_loss: Average validation loss
        """
        if self.val_loader is None:
            return 0.0

        self.encoder.eval()
        self.projection_head.eval()

        total_loss = 0.0
        num_batches = 0

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )

        with torch.no_grad():
            with progress:
                task = progress.add_task("Validating", total=len(self.val_loader))

                for batch in self.val_loader:
                    # Validation dataset returns single view, duplicate for loss computation
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        # Training mode: (view1, view2)
                        view1, view2 = batch
                    else:
                        # Validation mode: single view, use it twice
                        view1 = batch
                        view2 = batch

                    view1 = view1.to(self.device)
                    view2 = view2.to(self.device)

                    # Encode
                    h1 = self.encoder(view1)
                    h2 = self.encoder(view2)

                    # Project
                    z1 = self.projection_head(h1)
                    z2 = self.projection_head(h2)

                    # Loss
                    loss = self.criterion(z1, z2)

                    total_loss += loss.item()
                    num_batches += 1
                    progress.update(task, advance=1)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return avg_loss

    def save_checkpoint(self, epoch: int, loss: float):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            loss: Current loss
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'projection_head_state_dict': self.projection_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'global_step': self.global_step
        }

        path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        # Also save latest
        latest_path = self.save_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        self.console.print(f"[green]Checkpoint saved: {path}[/green]")

    def train(self, num_epochs: int, save_every: int = 10, start_epoch: int = 1):
        """
        Train for multiple epochs.

        Args:
            num_epochs: Total number of epochs to train (will train from start_epoch to num_epochs)
            save_every: Save checkpoint every N epochs
            start_epoch: Starting epoch number (for resuming training)
        """
        self.console.print("\n[bold cyan]Starting training...[/bold cyan]")
        self.console.print(f"Epochs: {start_epoch} -> {num_epochs}")
        self.console.print(f"Device: {self.device}")
        self.console.print(f"Checkpoint dir: {self.save_dir}\n")

        # Create metrics table
        table = Table(title="Training Progress")
        table.add_column("Epoch", justify="right")
        table.add_column("Train Loss", justify="right")
        table.add_column("Val Loss", justify="right")
        table.add_column("LR", justify="right", style="cyan")

        best_loss = float('inf')

        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log
            table.add_row(str(epoch), f"{train_loss:.4f}", f"{val_loss:.4f}", f"{current_lr:.2e}")
            self.console.print(table)

            # Save checkpoint
            if epoch % save_every == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, train_loss)

            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                best_path = self.save_dir / "checkpoint_best.pt"
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': self.encoder.state_dict(),
                    'projection_head_state_dict': self.projection_head.state_dict(),
                    'loss': best_loss
                }, best_path)

        self.console.print(f"\n[green]Training completed![/green]")
        self.console.print(f"Best loss: {best_loss:.4f}")
