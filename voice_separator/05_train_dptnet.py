"""
05_train_dptnet.py — Huấn luyện DPTNet (Dual-Path Transformer Network) cho bài toán tách nguồn 2 người nói.

Kết nối MixDataset (custom_dataset.py) với kiến trúc DPTNet từ repo qdPMx/DPTNet.
Hỗ trợ: SI-SNR loss, TransformerOptimizer (warmup + decay), checkpoint save/load, resume training, GPU fallback CPU.

Usage:
    python 05_train_dptnet.py --data_dir data_2mix --epochs 50 --batch_size 4
    python 05_train_dptnet.py --data_dir data_2mix --resume_checkpoint checkpoints_dptnet/epoch_05_loss_-8.1234.pth
"""

import argparse
import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from custom_dataset import MixDataset

# DPTNet default config
DEFAULT_ENC_DIM = 256
DEFAULT_FEATURE_DIM = 64
DEFAULT_HIDDEN_DIM = 128
DEFAULT_NUM_LAYERS = 6
DEFAULT_SEGMENT_SIZE = 250
DEFAULT_NUM_SPK = 2
DEFAULT_WIN_LEN = 2

# TransformerOptimizer config
DEFAULT_WARMUP_STEPS = 4000
DEFAULT_K = 0.2


def parse_args() -> argparse.Namespace:
    """Parse CLI args: data_dir, epochs, batch_size, lr, device, resume_checkpoint."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện DPTNet cho speech separation 2-mix"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data_2mix",
        help="Đường dẫn tới thư mục data_2mix (default: data_2mix)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Số epoch huấn luyện (default: 50)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size (default: 2, phù hợp GPU 16GB)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda/cpu). Mặc định: tự động chọn GPU nếu có",
    )
    parser.add_argument(
        "--resume_checkpoint", type=str, default=None,
        help="Đường dẫn checkpoint để resume training",
    )
    parser.add_argument(
        "--use_transformer_optim", action="store_true",
        help="Sử dụng TransformerOptimizer (warmup + decay) thay vì Adam thuần",
    )
    return parser.parse_args()


def build_model(
    enc_dim: int = DEFAULT_ENC_DIM,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    num_layers: int = DEFAULT_NUM_LAYERS,
    segment_size: int = DEFAULT_SEGMENT_SIZE,
    nspk: int = DEFAULT_NUM_SPK,
    win_len: int = DEFAULT_WIN_LEN,
) -> nn.Module:
    """Khởi tạo DPTNet từ repo DPTNet.

    Import model qua sys.path từ thư mục DPTNet.

    Returns:
        DPTNet_base model (nn.Module).

    Raises:
        ImportError: Nếu repo DPTNet chưa được clone.
    """
    repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DPTNet")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        from models import DPTNet_base  # noqa: E402
    except ImportError as e:
        raise ImportError(
            "Không tìm thấy repo DPTNet. "
            "Hãy clone repo vào thư mục DPTNet cùng cấp với script này."
        ) from e

    model = DPTNet_base(
        enc_dim=enc_dim,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        layer=num_layers,
        segment_size=segment_size,
        nspk=nspk,
        win_len=win_len,
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
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    dot = torch.sum(estimated * target, dim=-1, keepdim=True)
    s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True)
    s_target = (dot / (s_target_energy + 1e-8)) * target

    e_noise = estimated - s_target

    si_snr = 10 * torch.log10(
        torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + 1e-8)
    )

    return -si_snr.mean()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    epoch: int,
    use_transformer_optim: bool = False,
) -> float:
    """Train 1 epoch, trả về average loss.

    Args:
        model: DPTNet model.
        dataloader: DataLoader cho tập train.
        optimizer: Optimizer (Adam hoặc TransformerOptimizer).
        device: Device (cuda/cpu).
        epoch: Epoch hiện tại (dùng cho TransformerOptimizer).
        use_transformer_optim: Có dùng TransformerOptimizer không.

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

        # Forward pass: DPTNet trả về [B, nspk, T] trực tiếp
        estimated = model(mixture)  # [B, num_spk, T]

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

        if use_transformer_optim:
            optimizer.step(epoch)
        else:
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
        model: DPTNet model.
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

            estimated = model(mixture)  # [B, num_spk, T]

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
    optimizer,
    epoch: int,
    val_loss: float,
    path: str,
    use_transformer_optim: bool = False,
) -> None:
    """Lưu checkpoint gồm model_state, optimizer_state, epoch, val_loss, config."""
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if not use_transformer_optim
            else optimizer.optimizer.state_dict()
        ),
        "val_loss": val_loss,
        "use_transformer_optim": use_transformer_optim,
        "config": {
            "enc_dim": DEFAULT_ENC_DIM,
            "feature_dim": DEFAULT_FEATURE_DIM,
            "hidden_dim": DEFAULT_HIDDEN_DIM,
            "num_layers": DEFAULT_NUM_LAYERS,
            "segment_size": DEFAULT_SEGMENT_SIZE,
            "num_spk": DEFAULT_NUM_SPK,
            "win_len": DEFAULT_WIN_LEN,
        },
    }
    if use_transformer_optim:
        checkpoint["transformer_optim_state"] = {
            "step_num": optimizer.step_num,
            "epoch": optimizer.epoch,
        }

    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    use_transformer_optim: bool = False,
) -> int:
    """Load checkpoint để resume training.

    Returns:
        Epoch number để resume (epoch tiếp theo).

    Raises:
        FileNotFoundError: Nếu checkpoint không tồn tại.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint không tồn tại: {path}")

    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if use_transformer_optim:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "transformer_optim_state" in checkpoint:
            optimizer.step_num = checkpoint["transformer_optim_state"]["step_num"]
            optimizer.epoch = checkpoint["transformer_optim_state"]["epoch"]
    else:
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
    print(f"Optimizer: {'TransformerOptimizer (warmup)' if args.use_transformer_optim else 'Adam'}")

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
    print("\nĐang khởi tạo DPTNet...")
    model = build_model()
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # --- Optimizer ---
    if args.use_transformer_optim:
        # Import TransformerOptimizer từ repo DPTNet
        repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DPTNet")
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from others.optimizer_dptnet import TransformerOptimizer

        base_optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
        optimizer = TransformerOptimizer(
            base_optimizer,
            k=DEFAULT_K,
            d_model=DEFAULT_FEATURE_DIM,
            warmup_steps=DEFAULT_WARMUP_STEPS,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Resume ---
    start_epoch = 0
    if args.resume_checkpoint is not None:
        print(f"\nResume từ checkpoint: {args.resume_checkpoint}")
        start_epoch = load_checkpoint(
            args.resume_checkpoint, model, optimizer, args.use_transformer_optim
        )

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
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, device,
                    epoch=epoch + 1,
                    use_transformer_optim=args.use_transformer_optim,
                )
                print(f"  Train Loss: {train_loss:.4f}")

                val_loss = validate(model, val_loader, device)
                print(f"  Val Loss:   {val_loss:.4f}")

            except torch.cuda.OutOfMemoryError:
                print(
                    f"\n[ERROR] CUDA Out of Memory!"
                    f"\n  Gợi ý: Giảm batch_size (hiện tại: {args.batch_size}) "
                    f"hoặc segment_size (hiện tại: {DEFAULT_SEGMENT_SIZE})"
                )
                raise

            # Save checkpoint
            ckpt_path = (
                f"checkpoints_dptnet/epoch_{epoch + 1:02d}_loss_{val_loss:.4f}.pth"
            )
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss, ckpt_path,
                use_transformer_optim=args.use_transformer_optim,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                print(f"  ★ New best val loss: {best_val_loss:.4f} (epoch {best_epoch})")

    except KeyboardInterrupt:
        print("\n\nHuấn luyện bị dừng bởi người dùng.")
        ckpt_path = f"checkpoints_dptnet/epoch_{epoch + 1:02d}_loss_interrupted.pth"
        save_checkpoint(
            model, optimizer, epoch + 1, float("inf"), ckpt_path,
            use_transformer_optim=args.use_transformer_optim,
        )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("TỔNG KẾT HUẤN LUYỆN DPTNET")
    print("=" * 60)
    if best_epoch >= 0:
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Best epoch:           {best_epoch}")
    else:
        print("  Không có epoch nào hoàn tất.")
    print("=" * 60)


if __name__ == "__main__":
    main()
