"""Train ColorTransformerClassifier on ImageNet with dual loss."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ikunnet.models.color_transformer_classifier import ColorTransformerClassifier


def get_transforms(image_size=640):
    """Get training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    return train_transform, val_transform


def create_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    image_size: int = 640,
    num_workers: int = 0
):
    """Create training and validation dataloaders."""
    train_transform, val_transform = get_transforms(image_size)

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset, val_dataset


def train_epoch(
    model: ColorTransformerClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    console: Console,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_loss_clf = 0.0
    total_loss_vicreg = 0.0
    correct = 0
    total = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )

    with progress:
        task = progress.add_task(f"Epoch {epoch}", total=len(dataloader))

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                output = model(images, labels=labels)
                loss = output['loss']
                loss_clf = output['loss_clf']
                loss_vicreg = output['loss_vicreg']

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track metrics
            total_loss += loss.item()
            total_loss_clf += loss_clf.item()
            total_loss_vicreg += loss_vicreg.item()

            # Accuracy
            preds = output['logits'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]

            # Update progress
            progress.update(
                task,
                advance=1,
                description=f"Epoch {epoch} | Loss: {loss.item():.4f} | Clf: {loss_clf.item():.4f} | VIC: {loss_vicreg.item():.4f}"
            )

    avg_loss = total_loss / len(dataloader)
    avg_loss_clf = total_loss_clf / len(dataloader)
    avg_loss_vicreg = total_loss_vicreg / len(dataloader)
    accuracy = correct / total

    return {
        'loss': avg_loss,
        'loss_clf': avg_loss_clf,
        'loss_vicreg': avg_loss_vicreg,
        'accuracy': accuracy
    }


@torch.no_grad()
def validate(
    model: ColorTransformerClassifier,
    dataloader: DataLoader,
    device: str,
    console: Console
) -> dict:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )

    with progress:
        task = progress.add_task("Validating", total=len(dataloader))

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images, labels=labels)
            loss = output['loss']

            total_loss += loss.item()

            preds = output['logits'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]

            progress.update(task, advance=1)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def main():
    parser = argparse.ArgumentParser(description="Train ColorTransformerClassifier with dual loss")

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/imagenet1k/imagenet-mini',
                       help='ImageNet data directory')
    parser.add_argument('--image-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers')

    # Model arguments
    parser.add_argument('--num-classes', type=int, default=1000,
                       help='Number of output classes')
    parser.add_argument('--encoder-dim', type=int, default=256,
                       help='Encoder feature dimension')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads in pooling')

    # Color separator arguments
    parser.add_argument('--interval-size', type=int, default=40,
                       help='Color interval size for ColorSeparator')
    parser.add_argument('--min-pixels', type=int, default=100,
                       help='Minimum pixel count for color groups')
    parser.add_argument('--separator-device', type=str, default='cuda', choices=['cpu', 'cuda'],
                       help='Device for ColorSeparator')

    # Positional encoding
    parser.add_argument('--use-sin-encoding', action='store_true', default=True,
                       help='Use sinusoidal RGB encoding (default: True)')
    parser.add_argument('--use-mlp-encoding', action='store_true',
                       help='Use MLP RGB encoding instead of sinusoidal')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')

    # VICReg arguments
    parser.add_argument('--variance-weight', type=float, default=1.0,
                       help='Weight for variance loss term in VICReg')
    parser.add_argument('--covariance-weight', type=float, default=1.0,
                       help='Weight for covariance loss term in VICReg')
    parser.add_argument('--std-target', type=float, default=1.0,
                       help='Target standard deviation for VICReg')

    # System arguments
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--save-dir', type=str, default='checkpoints_classifier',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    # Check data directory
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    if not os.path.exists(train_dir):
        print(f"Error: Training data directory not found: {train_dir}")
        print("Please provide the ImageNet dataset directory with train/ val subdirectories.")
        return

    console = Console()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Print config
    console.print("\n[bold cyan]Color Transformer Classifier Training[/bold cyan]")
    console.print(f"Data dir: {args.data_dir}")
    console.print(f"Batch size: {args.batch_size}")
    console.print(f"Image size: {args.image_size}")
    console.print(f"Encoder dim: {args.encoder_dim}")
    console.print(f"Num classes: {args.num_classes}")
    console.print(f"Interval size: {args.interval_size}")
    console.print(f"Min pixels: {args.min_pixels}")
    console.print(f"Separator device: {args.separator_device}")
    console.print(f"Device: {args.device}")

    # Determine encoding type
    use_sin_encoding = not args.use_mlp_encoding
    encoding_type = "Sinusoidal" if use_sin_encoding else "MLP"
    console.print(f"RGB Encoding: {encoding_type}\n")

    # Create dataloaders
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )

    console.print(f"Train samples: {len(train_dataset)}")
    console.print(f"Val samples: {len(val_dataset)}")
    console.print(f"Train batches: {len(train_loader)}")
    console.print(f"Val batches: {len(val_loader)}\n")

    # Create model
    model = ColorTransformerClassifier(
        num_classes=args.num_classes,
        encoder_dim=args.encoder_dim,
        num_heads=args.num_heads,
        use_sin_encoding=use_sin_encoding,
        interval_size=args.interval_size,
        min_pixel_count=args.min_pixels,
        device=args.separator_device
    ).to(args.device)

    # Update VICReg loss weights
    model.vicreg_loss.variance_weight = args.variance_weight
    model.vicreg_loss.covariance_weight = args.covariance_weight
    model.vicreg_loss.std_target = args.std_target

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Total parameters: {total_params:,}")
    console.print(f"Trainable parameters: {trainable_params:,}\n")

    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        console.print(f"Resuming from [yellow]{args.resume}[/yellow]")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        console.print(f"Resumed from epoch {checkpoint.get('epoch', 0)}\n")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    # Scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.device == 'cuda' else None

    # Metrics table
    table = Table(title="Training Progress")
    table.add_column("Epoch", justify="right")
    table.add_column("Train Loss", justify="right")
    table.add_column("Train Acc", justify="right")
    table.add_column("Val Loss", justify="right")
    table.add_column("Val Acc", justify="right")
    table.add_column("LR", justify="right", style="cyan")

    best_val_acc = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, args.device, console, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, args.device, console)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        table.add_row(
            str(epoch),
            f"{train_metrics['loss']:.4f}",
            f"{train_metrics['accuracy']:.4f}",
            f"{val_metrics['loss']:.4f}",
            f"{val_metrics['accuracy']:.4f}",
            f"{current_lr:.2e}"
        )
        console.print(table)

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
            model.save_checkpoint(
                checkpoint_path,
                epoch,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                val_acc=val_metrics['accuracy']
            )

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_path = save_dir / "checkpoint_best.pt"
            model.save_checkpoint(
                best_path,
                epoch,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                val_acc=val_metrics['accuracy']
            )
            console.print(f"[green]New best val acc: {best_val_acc:.4f}[/green]")

    console.print(f"\n[green]Training completed![/green]")
    console.print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
