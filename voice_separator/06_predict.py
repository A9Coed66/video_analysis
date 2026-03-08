"""
06_predict.py — Tách 2 giọng nói từ file audio sử dụng DPRNN-TasNet đã train.

Usage:
    python 06_predict.py --input mix.wav --output output_dir/
    python 06_predict.py --input mix.wav --checkpoint checkpoints/epoch_50_loss_-0.6588.pth
    python 06_predict.py --input_dir folder_of_mixes/ --output output_dir/
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# DPRNN config (phải khớp với 04_train_dprnn.py)
DEFAULT_NUM_BLOCKS = 6
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_ENCODER_DIM = 64
DEFAULT_KERNEL_SIZE = 2
DEFAULT_NUM_SPK = 2
SAMPLE_RATE = 16000


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load DPRNN-TasNet model từ checkpoint.

    Args:
        checkpoint_path: Đường dẫn tới file .pth checkpoint.
        device: Device để load model.

    Returns:
        Model đã load weights, ở eval mode.
    """
    repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dual-Path-RNN-Pytorch")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from model.model_rnn import Dual_RNN_model

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    model = Dual_RNN_model(
        in_channels=config.get("encoder_dim", DEFAULT_ENCODER_DIM),
        out_channels=config.get("encoder_dim", DEFAULT_ENCODER_DIM),
        hidden_channels=config.get("hidden_size", DEFAULT_HIDDEN_SIZE),
        kernel_size=config.get("kernel_size", DEFAULT_KERNEL_SIZE),
        rnn_type="LSTM",
        norm="ln",
        dropout=0,
        bidirectional=False,
        num_layers=config.get("num_blocks", DEFAULT_NUM_BLOCKS),
        K=100,
        num_spks=DEFAULT_NUM_SPK,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"Model loaded: epoch {epoch}, val_loss {val_loss}")
    return model


def separate(
    model: torch.nn.Module,
    audio: np.ndarray,
    device: torch.device,
    chunk_size: int = 64000,
) -> list[np.ndarray]:
    """Tách audio thành 2 nguồn.

    Xử lý audio dài bằng cách chia thành chunks rồi ghép lại.

    Args:
        model: DPRNN-TasNet model (eval mode).
        audio: Input audio, 1D numpy array, 16kHz mono float32.
        device: Device.
        chunk_size: Kích thước mỗi chunk (default: 64000 = 4s @ 16kHz).

    Returns:
        List gồm 2 numpy arrays [s1, s2], mỗi array cùng length với input.
    """
    original_len = len(audio)

    # Pad audio nếu ngắn hơn chunk_size
    if len(audio) < chunk_size:
        audio = np.pad(audio, (0, chunk_size - len(audio)))

    # Chia thành chunks
    n_chunks = (len(audio) + chunk_size - 1) // chunk_size
    padded_len = n_chunks * chunk_size
    if len(audio) < padded_len:
        audio = np.pad(audio, (0, padded_len - len(audio)))

    sources = [[], []]

    with torch.no_grad():
        for i in range(n_chunks):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            x = torch.from_numpy(chunk).float().unsqueeze(0).to(device)  # [1, T]

            estimated_list = model(x)  # list of [1, T'] tensors

            for spk_idx, est in enumerate(estimated_list):
                out = est.squeeze(0).cpu().numpy()
                # Pad/trim to chunk_size
                if len(out) < chunk_size:
                    out = np.pad(out, (0, chunk_size - len(out)))
                else:
                    out = out[:chunk_size]
                sources[spk_idx].append(out)

    # Ghép chunks và trim về original length
    s1 = np.concatenate(sources[0])[:original_len]
    s2 = np.concatenate(sources[1])[:original_len]

    return [s1, s2]


def load_audio(path: str) -> np.ndarray:
    """Load audio file, chuyển về mono 16kHz float32.

    Args:
        path: Đường dẫn file audio (.wav, .mp3, .flac, ...).

    Returns:
        1D numpy array, float32.
    """
    audio, sr = sf.read(path, dtype="float32")

    # Stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample nếu cần
    if sr != SAMPLE_RATE:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        except ImportError:
            warnings.warn(
                f"File {path} có sample rate {sr}Hz, cần librosa để resample về {SAMPLE_RATE}Hz. "
                f"pip install librosa",
                stacklevel=2,
            )

    return audio


def find_best_checkpoint(checkpoint_dir: str = "checkpoints") -> str | None:
    """Tìm checkpoint có val_loss thấp nhất.

    Args:
        checkpoint_dir: Thư mục chứa checkpoints.

    Returns:
        Đường dẫn tới best checkpoint, hoặc None nếu không tìm thấy.
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return None

    best_path = None
    best_loss = float("inf")

    for f in ckpt_path.glob("epoch_*_loss_*.pth"):
        name = f.stem  # epoch_50_loss_-0.6588
        try:
            loss_str = name.split("loss_")[1]
            loss_val = float(loss_str)
            if loss_val < best_loss:
                best_loss = loss_val
                best_path = str(f)
        except (IndexError, ValueError):
            continue

    return best_path


def main():
    parser = argparse.ArgumentParser(
        description="Tách 2 giọng nói từ audio sử dụng DPRNN-TasNet"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Đường dẫn file audio đầu vào",
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Thư mục chứa nhiều file audio để xử lý batch",
    )
    parser.add_argument(
        "--output", type=str, default="separated_output",
        help="Thư mục output (default: separated_output/)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Đường dẫn checkpoint. Mặc định: tự tìm best checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda/cpu). Mặc định: tự chọn",
    )
    args = parser.parse_args()

    if args.input is None and args.input_dir is None:
        parser.error("Cần chỉ định --input hoặc --input_dir")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Checkpoint
    ckpt_path = args.checkpoint or find_best_checkpoint()
    if ckpt_path is None:
        print("Không tìm thấy checkpoint. Hãy train model trước (04_train_dprnn.py)")
        return
    print(f"Checkpoint: {ckpt_path}")

    # Load model
    model = load_model(ckpt_path, device)

    # Collect input files
    input_files = []
    if args.input:
        input_files.append(args.input)
    if args.input_dir:
        in_dir = Path(args.input_dir)
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
            input_files.extend(str(f) for f in in_dir.glob(ext))

    if not input_files:
        print("Không tìm thấy file audio nào.")
        return

    # Output dir
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nXử lý {len(input_files)} file(s)...")
    print("=" * 60)

    for i, fpath in enumerate(input_files):
        fname = Path(fpath).stem
        print(f"\n[{i+1}/{len(input_files)}] {fpath}")

        audio = load_audio(fpath)
        print(f"  Duration: {len(audio)/SAMPLE_RATE:.2f}s ({len(audio)} samples)")

        s1, s2 = separate(model, audio, device)

        # Save
        s1_path = out_dir / f"{fname}_spk1.wav"
        s2_path = out_dir / f"{fname}_spk2.wav"
        sf.write(str(s1_path), s1, SAMPLE_RATE, subtype="FLOAT")
        sf.write(str(s2_path), s2, SAMPLE_RATE, subtype="FLOAT")
        print(f"  -> {s1_path}")
        print(f"  -> {s2_path}")

    print("\n" + "=" * 60)
    print(f"Hoàn tất! Output: {out_dir}/")


if __name__ == "__main__":
    main()
