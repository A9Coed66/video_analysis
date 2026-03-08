"""
04_train_dprnn.py — Huấn luyện DPRNN-TasNet cho bài toán tách nguồn 2 người nói.

Kết nối MixDataset (custom_dataset.py) với kiến trúc DPRNN-TasNet từ repo JusperLee.
Hỗ trợ: SI-SNR loss, checkpoint save/load, resume training, GPU fallback CPU.

Usage:
    python 04_train_dprnn.py --data_dir data_2mix --epochs 50 --batch_size 4
    python 04_train_dprnn.py --data_dir data_2mix --resume_checkpoint checkpoints/epoch_05_loss_-8.1234.pth
"""

import argparse
import os
import sys
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from custom_dataset import MixDataset

# DPRNN-TasNet default config
DEFAULT_NUM_BLOCKS = 6
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_ENCODER_DIM = 64
DEFAULT_KERNEL_SIZE = 2
DEFAULT_NUM_SPK = 2


def parse_args() -> argparse.Namespace:
    """Parse CLI args: data_dir, epochs, batch_size, lr, device, resume_checkpoint."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện DPRNN-TasNet cho speech separation 2-mix"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_2mix",
        help="Đường dẫn tới thư mục data_2mix (default: data_2mix)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Số epoch huấn luyện (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (default: 4, phù hợp GPU 16GB)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Mặc định: tự động chọn GPU nếu có",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Đường dẫn checkpoint để resume training",
    )
    return parser.parse_args()


def build_model(
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    encoder_dim: int = DEFAULT_ENCODER_DIM,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
) -> nn.Module:
    """Khởi tạo DPRNN-TasNet từ repo JusperLee.

    Import model qua sys.path từ thư mục Dual-Path-RNN-Pytorch.

    Args:
        num_blocks: Số DPRNN blocks (default: 6).
        hidden_size: Kích thước hidden state (default: 128).
        encoder_dim: Encoder output dimension N (default: 64).
        kernel_size: Encoder kernel size L (default: 2).

    Returns:
        DPRNN-TasNet model (nn.Module).

    Raises:
        ImportError: Nếu repo Dual-Path-RNN-Pytorch chưa được clone.
    """
    repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dual_path_RNN")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        from model.model_rnn import Dual_RNN_model  # noqa: E402
    except ImportError as e:
        raise ImportError(
            "Không tìm thấy repo Dual-Path-RNN-Pytorch. "
            "Hãy clone repo: git clone https://github.com/JusperLee/Dual-Path-RNN-Pytorch.git"
        ) from e

    model = Dual_RNN_model(
        in_channels=encoder_dim,
        out_channels=encoder_dim,
        hidden_channels=hidden_size,
        kernel_size=kernel_size,
        rnn_type="LSTM",
        norm="ln",
        dropout=0,
        bidirectional=False,
        num_layers=num_blocks,
        K=100,
        num_spks=DEFAULT_NUM_SPK,
    )
    return model


def si_snr_loss(estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Tính SI-SNR loss (negative SI-SNR để minimize).

    Công thức:
        s_target = <s_hat, s> / ||s||^2 * s
        e_noise = s_hat - s_target
        SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        loss = -mean(SI-SNR)

    Args:
        estimated: Tín hiệu ước lượng, shape [..., T].
        target: Tín hiệu mục tiêu, shape [..., T].

    Returns:
        Negative SI-SNR loss (scalar tensor).
    """
    # Zero-mean normalization
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # s_target = <s_hat, s> / ||s||^2 * s
    dot = torch.sum(estimated * target, dim=-1, keepdim=True)
    s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True)
    s_target = (dot / (s_target_energy + 1e-8)) * target

    # e_noise = s_hat - s_target
    e_noise = estimated - s_target

    # SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    si_snr = 10 * torch.log10(
        torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + 1e-8)
    )

    # Negative mean SI-SNR (to minimize)
    return -si_snr.mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train 1 epoch, trả về average loss.

    Args:
        model: DPRNN-TasNet model.
        dataloader: DataLoader cho tập train.
        optimizer: Optimizer (Adam).
        device: Device (cuda/cpu).

    Returns:
        Average loss cho epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (mixture, sources) in enumerate(dataloader):
        mixture = mixture.to(device)   # [B, T]
        sources = sources.to(device)   # [B, 2, T]

        optimizer.zero_grad()

        # Forward pass: model nhận mixture, trả về list of estimated sources
        estimated_list = model(mixture)  # list of [B, T] tensors
        estimated = torch.stack(estimated_list, dim=1)  # [B, num_spk, T]

        # Cắt theo min length (encoder/decoder có thể thay đổi T)
        min_t = min(estimated.shape[-1], sources.shape[-1])
        estimated = estimated[..., :min_t]
        sources = sources[..., :min_t]

        # Tính loss trung bình trên cả 2 sources
        loss = 0.0
        for spk in range(sources.shape[1]):
            loss = loss + si_snr_loss(estimated[:, spk, :], sources[:, spk, :])
        loss = loss / sources.shape[1]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Đánh giá trên validation set, trả về average loss.

    Args:
        model: DPRNN-TasNet model.
        dataloader: DataLoader cho tập validation.
        device: Device (cuda/cpu).

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for mixture, sources in dataloader:
            mixture = mixture.to(device)
            sources = sources.to(device)

            estimated_list = model(mixture)
            estimated = torch.stack(estimated_list, dim=1)  # [B, num_spk, T]

            # Cắt theo min length
            min_t = min(estimated.shape[-1], sources.shape[-1])
            estimated = estimated[..., :min_t]
            sources = sources[..., :min_t]

            loss = 0.0
            for spk in range(sources.shape[1]):
                loss = loss + si_snr_loss(estimated[:, spk, :], sources[:, spk, :])
            loss = loss / sources.shape[1]

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: str,
) -> None:
    """Lưu checkpoint gồm model_state, optimizer_state, epoch, val_loss, config.

    Tạo thư mục checkpoints/ nếu chưa tồn tại.
    Tên file: checkpoints/epoch_{epoch:02d}_loss_{val_loss:.4f}.pth

    Args:
        model: DPRNN-TasNet model.
        optimizer: Optimizer.
        epoch: Số epoch hiện tại.
        val_loss: Validation loss.
        path: Đường dẫn lưu checkpoint.
    """
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": {
            "num_blocks": DEFAULT_NUM_BLOCKS,
            "hidden_size": DEFAULT_HIDDEN_SIZE,
            "encoder_dim": DEFAULT_ENCODER_DIM,
            "kernel_size": DEFAULT_KERNEL_SIZE,
        },
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load checkpoint để resume training.

    Args:
        path: Đường dẫn tới file checkpoint.
        model: DPRNN-TasNet model để load weights.
        optimizer: Optimizer để load state.

    Returns:
        Epoch number để resume (epoch tiếp theo).

    Raises:
        FileNotFoundError: Nếu checkpoint không tồn tại.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint không tồn tại: {path}")

    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    val_loss = checkpoint.get("val_loss", None)
    print(f"  Loaded checkpoint: epoch {epoch}, val_loss: {val_loss}")

    return epoch


def main():
    """Main training loop: MixDataset → DataLoader → train loop."""
    args = parse_args()

    # --- Device setup ---
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        warnings.warn(
            "GPU không khả dụng. Sử dụng CPU — huấn luyện sẽ rất chậm!",
            stacklevel=2,
        )
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Config: batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}")

    # --- Dataset & DataLoader ---
    print("\nĐang tạo DataLoader...")
    train_dataset = MixDataset(data_dir=args.data_dir, split="tr")
    val_dataset = MixDataset(data_dir=args.data_dir, split="cv")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # --- Model ---
    print("\nĐang khởi tạo DPRNN-TasNet...")
    model = build_model()
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Resume ---
    start_epoch = 0
    if args.resume_checkpoint is not None:
        print(f"\nResume từ checkpoint: {args.resume_checkpoint}")
        start_epoch = load_checkpoint(args.resume_checkpoint, model, optimizer)

    # --- Training loop ---
    best_val_loss = float("inf")
    best_epoch = -1

    print(f"\nBắt đầu huấn luyện từ epoch {start_epoch + 1}...")
    print("=" * 60)

    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print("-" * 40)

            try:
                # Train
                train_loss = train_one_epoch(model, train_loader, optimizer, device)
                print(f"  Train Loss: {train_loss:.4f}")

                # Validate
                val_loss = validate(model, val_loader, device)
                print(f"  Val Loss:   {val_loss:.4f}")

            except torch.cuda.OutOfMemoryError:
                print(
                    f"\n[ERROR] CUDA Out of Memory!"
                    f"\n  Gợi ý: Giảm batch_size (hiện tại: {args.batch_size}) "
                    f"hoặc hidden_size (hiện tại: {DEFAULT_HIDDEN_SIZE})"
                )
                raise

            # Save checkpoint
            ckpt_path = (
                f"checkpoints_premium/epoch_{epoch + 1:02d}_loss_{val_loss:.4f}.pth"
            )
            save_checkpoint(model, optimizer, epoch + 1, val_loss, ckpt_path)

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                print(f"  ★ New best val loss: {best_val_loss:.4f} (epoch {best_epoch})")

    except KeyboardInterrupt:
        print("\n\nHuấn luyện bị dừng bởi người dùng.")
        # Save final checkpoint on interrupt
        ckpt_path = f"checkpoints_premium/epoch_{epoch + 1:02d}_loss_interrupted.pth"
        save_checkpoint(model, optimizer, epoch + 1, float("inf"), ckpt_path)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("TỔNG KẾT HUẤN LUYỆN")
    print("=" * 60)
    if best_epoch >= 0:
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Best epoch:           {best_epoch}")
    else:
        print("  Không có epoch nào hoàn tất.")
    print("=" * 60)


if __name__ == "__main__":
    main()
