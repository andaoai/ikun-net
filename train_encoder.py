"""Train encoder for mask embedding with contrastive learning."""

import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split

from ikunnet.models.encoder import CNNEncoder
from ikunnet.models.projection_head import ProjectionHead
from ikunnet.training.dataset import MaskDataset
from ikunnet.training.augmentation import SimpleTwoViewTransform, MaskTransform
from ikunnet.training.trainer import Trainer
from ikunnet.training.online_dataset import ImageMaskDataset


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    val_split: float = 0.1
):
    """
    Create training and validation dataloaders.

    Args:
        data_dir: Directory containing masks
        batch_size: Batch size
        num_workers: Number of workers for data loading
        image_size: Image size after augmentation
        val_split: Validation split ratio

    Returns:
        train_loader, val_loader
    """
    # Training dataset with two identical views (no augmentation)
    train_dataset = MaskDataset(
        mask_dir=data_dir,
        two_view_transform=SimpleTwoViewTransform(image_size=image_size)
    )

    # Validation dataset with single-view transform
    val_dataset = MaskDataset(
        mask_dir=data_dir,
        transform=MaskTransform(image_size=image_size)
    )

    # Split into train/val
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size

    if val_size > 0:
        train_dataset, _ = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

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
    ) if val_size > 0 else None

    return train_loader, val_loader


def create_online_dataloaders(
    data_dir: str,
    images_per_batch: int,
    interval_size: int,
    min_pixel_count: int,
    separator_device: str,
    loss_type: str = 'contrastive',
    num_workers: int = 0
):
    """
    Create online mask extraction dataloader.

    Args:
        data_dir: ImageNet dataset directory
        images_per_batch: Number of images to process per batch
        interval_size: Color interval size for ColorSeparator
        min_pixel_count: Minimum pixel count for color groups
        separator_device: Device for ColorSeparator ('cpu' or 'cuda')
        loss_type: 'contrastive' (returns 2 views) or 'uniformity' (returns 1 view)
        num_workers: Number of workers (must be 0 for ColorSeparator)

    Returns:
        train_loader, val_loader (val_loader is None for online mode)
    """
    train_dataset = ImageMaskDataset(
        image_dir=data_dir,
        images_per_batch=images_per_batch,
        interval_size=interval_size,
        min_pixel_count=min_pixel_count,
        device=separator_device
    )

    # Collate function - extract masks on-the-fly
    def collate_fn(batch):
        separator = train_dataset.get_separator()
        all_masks = []

        for image_paths in batch:
            for img_path in image_paths:
                # Extract masks using ColorSeparator
                results = separator.separate_colors(img_path)
                masks = results['masks']
                color_groups = results['color_groups']

                # Convert masks to tensors and resize
                for group in color_groups:
                    mask = masks[group.group_id]
                    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)
                    # Resize from 640x640 to 224x224 to save memory
                    mask_tensor = torch.nn.functional.interpolate(
                        mask_tensor.unsqueeze(0),
                        size=(224, 224),
                        mode='nearest'
                    ).squeeze(0)
                    all_masks.append(mask_tensor)

        # Handle empty batch
        if len(all_masks) == 0:
            if loss_type == 'contrastive':
                return torch.zeros(0, 1, 224, 224), torch.zeros(0, 1, 224, 224)
            else:  # uniformity
                return torch.zeros(0, 1, 224, 224)

        # Stack all masks into one batch
        masks_batch = torch.stack(all_masks, dim=0)  # (total_masks, 1, 224, 224)

        # Return format based on loss_type
        if loss_type == 'contrastive':
            # Create two identical views for contrastive learning
            view1 = masks_batch.clone()
            view2 = masks_batch.clone()
            return view1, view2
        else:  # uniformity
            # Return single view for uniformity loss
            return masks_batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Each item is already a batch of images
        shuffle=True,
        num_workers=num_workers,  # Must be 0 for ColorSeparator
        collate_fn=collate_fn,
        pin_memory=False  # Disable pin_memory for online mode
    )

    return train_loader, None  # No validation set for online mode


def main():
    parser = argparse.ArgumentParser(description="Train mask encoder with contrastive learning")

    # Mode selection
    parser.add_argument('--online', action='store_true',
                       help='Use online mask extraction from ImageNet images')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/masks_dataset/masks',
                       help='Directory containing data (masks for offline, ImageNet for online)')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size after augmentation (offline mode only)')

    # Online mode arguments
    parser.add_argument('--images-per-batch', type=int, default=4,
                       help='Number of images to process per batch (online mode)')
    parser.add_argument('--interval-size', type=int, default=40,
                       help='Color interval size for ColorSeparator (online mode)')
    parser.add_argument('--min-pixels', type=int, default=100,
                       help='Minimum pixel count for color groups (online mode)')
    parser.add_argument('--separator-device', type=str, default='cuda', choices=['cpu', 'cuda'],
                       help='Device for ColorSeparator (online mode)')

    # Model arguments
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--feature-dim', type=int, default=256,
                       help='Encoder feature dimension before projection')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for NT-Xent loss (contrastive) or uniformity loss')
    parser.add_argument('--loss-type', type=str, default='contrastive',
                       choices=['contrastive', 'uniformity'],
                       help='Loss function type: contrastive or uniformity')

    # System arguments
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    # Check data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        if args.online:
            print("Please provide the ImageNet dataset directory.")
        else:
            print("Please run prepare_masks.py first to extract masks from ImageNet.")
        return

    # Create dataloaders based on mode
    if args.online:
        print("Using online mask extraction mode")
        print(f"Loading images from: {args.data_dir}")
        print(f"Images per batch: {args.images_per_batch}")
        print(f"Interval size: {args.interval_size}")
        print(f"Separator device: {args.separator_device}")

        # Force num_workers to 0 for online mode
        num_workers = 0
        if args.num_workers != 0:
            print("Warning: num_workers is forced to 0 for online mode")

        train_loader, val_loader = create_online_dataloaders(
            data_dir=args.data_dir,
            images_per_batch=args.images_per_batch,
            interval_size=args.interval_size,
            min_pixel_count=args.min_pixels,
            separator_device=args.separator_device,
            loss_type=args.loss_type,
            num_workers=num_workers
        )
    else:
        print("Using offline mode (pre-computed masks)")
        print(f"Loading masks from: {args.data_dir}")

        train_loader, val_loader = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size
        )

    print(f"Training batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")

    # Create models
    encoder = CNNEncoder(embedding_dim=args.embedding_dim)
    projection_head = ProjectionHead(
        feature_dim=args.feature_dim,
        embedding_dim=args.embedding_dim
    )

    # Create trainer
    trainer = Trainer(
        encoder=encoder,
        projection_head=projection_head,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        loss_type=args.loss_type,
        device=args.device,
        save_dir=args.save_dir
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        trainer.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        trainer.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        # Only load optimizer state if it exists (checkpoint_best.pt doesn't have it)
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.global_step = checkpoint.get('global_step', 0)
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {checkpoint.get('epoch', 0)}")

    # Train
    trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        start_epoch=start_epoch
    )


if __name__ == '__main__':
    main()
