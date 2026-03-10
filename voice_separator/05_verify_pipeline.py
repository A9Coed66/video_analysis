"""
05_verify_pipeline.py — Script xác minh toàn bộ DPRNN-2mix training pipeline.

Kiểm tra từng bước: segments → mixtures → dataloader → training,
in PASS/FAIL/SKIP cho mỗi bước.
"""

from pathlib import Path

import numpy as np
import soundfile as sf


# Constants
SAMPLE_RATE = 16000
SEGMENT_SAMPLES = 64000  # 4s at 16kHz
MIX_TOLERANCE = 1e-5


def verify_segments(segments_dir: str = "segments") -> bool:
    """Kiểm tra tất cả file trong segments/: 16kHz, mono, 64000 samples.

    Args:
        segments_dir: Đường dẫn tới thư mục segments.

    Returns:
        True nếu tất cả file đạt yêu cầu, False nếu có file lỗi.
    """
    segments_path = Path(segments_dir)
    if not segments_path.exists():
        return None  # SKIP

    wav_files = list(segments_path.rglob("*.wav"))
    if not wav_files:
        print(f"  Không tìm thấy file WAV trong {segments_dir}/")
        return False

    total = len(wav_files)
    errors = []

    for wav_file in wav_files:
        try:
            info = sf.info(str(wav_file))
            if info.samplerate != SAMPLE_RATE:
                errors.append(f"  {wav_file}: sr={info.samplerate} (expected {SAMPLE_RATE})")
            if info.channels != 1:
                errors.append(f"  {wav_file}: channels={info.channels} (expected 1)")
            if info.frames != SEGMENT_SAMPLES:
                errors.append(f"  {wav_file}: frames={info.frames} (expected {SEGMENT_SAMPLES})")
        except Exception as e:
            errors.append(f"  {wav_file}: không đọc được — {e}")

    if errors:
        print(f"  Kiểm tra {total} files, {len(errors)} lỗi:")
        for err in errors[:10]:  # Chỉ in tối đa 10 lỗi
            print(err)
        if len(errors) > 10:
            print(f"  ... và {len(errors) - 10} lỗi khác")
        return False

    print(f"  Kiểm tra {total} files — tất cả 16kHz, mono, {SEGMENT_SAMPLES} samples")
    return True


def verify_mixtures(data_2mix_dir: str = "data_2mix") -> bool:
    """Kiểm tra số file mix == s1 == s2, và mix ≈ s1 + s2.

    Cho mỗi split (tr, cv, tt):
    - Đếm file trong mix/, s1/, s2/ — phải bằng nhau
    - Lấy mẫu một số file, kiểm tra mix ≈ s1 + s2 (tolerance 1e-5)

    Args:
        data_2mix_dir: Đường dẫn tới thư mục data_2mix.

    Returns:
        True nếu tất cả kiểm tra đạt, False nếu có lỗi.
    """
    data_path = Path(data_2mix_dir)
    if not data_path.exists():
        return None  # SKIP

    splits = ["tr", "cv", "tt"]
    all_ok = True

    for split in splits:
        split_dir = data_path / split
        if not split_dir.exists():
            print(f"  Split '{split}' không tồn tại — bỏ qua")
            continue

        mix_dir = split_dir / "mix"
        s1_dir = split_dir / "s1"
        s2_dir = split_dir / "s2"

        # Kiểm tra thư mục con tồn tại
        for d in (mix_dir, s1_dir, s2_dir):
            if not d.exists():
                print(f"  Thư mục {d} không tồn tại")
                all_ok = False
                continue

        mix_files = sorted(mix_dir.glob("*.wav"))
        s1_files = sorted(s1_dir.glob("*.wav"))
        s2_files = sorted(s2_dir.glob("*.wav"))

        n_mix = len(mix_files)
        n_s1 = len(s1_files)
        n_s2 = len(s2_files)

        # Kiểm tra số file bằng nhau
        if not (n_mix == n_s1 == n_s2):
            print(f"  Split '{split}': file count mismatch — mix={n_mix}, s1={n_s1}, s2={n_s2}")
            all_ok = False
            continue

        if n_mix == 0:
            print(f"  Split '{split}': không có file WAV")
            all_ok = False
            continue

        print(f"  Split '{split}': {n_mix} files (mix == s1 == s2 ✓)")

        # Lấy mẫu tối đa 5 file để kiểm tra mix ≈ s1 + s2
        sample_count = min(5, n_mix)
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(n_mix, size=sample_count, replace=False)

        for idx in sample_indices:
            try:
                mix_audio, _ = sf.read(str(mix_files[idx]), dtype="float32")
                s1_audio, _ = sf.read(str(s1_files[idx]), dtype="float32")
                s2_audio, _ = sf.read(str(s2_files[idx]), dtype="float32")

                reconstructed = s1_audio + s2_audio
                max_diff = np.max(np.abs(mix_audio - reconstructed))

                if max_diff > MIX_TOLERANCE:
                    print(f"  Split '{split}', file {mix_files[idx].name}: "
                          f"mix != s1 + s2 (max diff = {max_diff:.2e})")
                    all_ok = False
            except Exception as e:
                print(f"  Split '{split}', file {mix_files[idx].name}: lỗi đọc — {e}")
                all_ok = False

    return all_ok


def verify_dataloader(data_2mix_dir: str = "data_2mix") -> bool:
    """Kiểm tra DataLoader lấy được ít nhất 1 batch.

    Khởi tạo MixDataset + DataLoader, thử lấy 1 batch.

    Args:
        data_2mix_dir: Đường dẫn tới thư mục data_2mix.

    Returns:
        True nếu lấy được ít nhất 1 batch, False nếu lỗi.
    """
    data_path = Path(data_2mix_dir)
    if not data_path.exists():
        return None  # SKIP

    # Tìm split có dữ liệu
    available_split = None
    for split in ["tr", "cv", "tt"]:
        split_dir = data_path / split
        if split_dir.exists() and (split_dir / "mix").exists():
            mix_files = list((split_dir / "mix").glob("*.wav"))
            if mix_files:
                available_split = split
                break

    if available_split is None:
        print("  Không tìm thấy split nào có dữ liệu")
        return False

    try:
        import torch
        from torch.utils.data import DataLoader

        from custom_dataset import MixDataset

        dataset = MixDataset(data_dir=data_2mix_dir, split=available_split)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

        # Lấy 1 batch
        batch = next(iter(dataloader))
        mixture, sources = batch

        print(f"  Split '{available_split}': dataset size = {len(dataset)}")
        print(f"  Batch mixture shape: {mixture.shape}, dtype: {mixture.dtype}")
        print(f"  Batch sources shape: {sources.shape}, dtype: {sources.dtype}")

        # Kiểm tra shape và dtype
        if mixture.dtype != torch.float32:
            print(f"  mixture dtype sai: {mixture.dtype} (expected float32)")
            return False
        if sources.dtype != torch.float32:
            print(f"  sources dtype sai: {sources.dtype} (expected float32)")
            return False

        return True

    except ImportError as e:
        print(f"  Import error: {e}")
        return False
    except Exception as e:
        print(f"  Lỗi khi tạo DataLoader hoặc lấy batch: {e}")
        return False


def verify_training(checkpoints_dir: str = "checkpoints") -> bool:
    """Kiểm tra loss giảm dần qua các epoch từ checkpoints.

    Đọc checkpoint files, trích xuất val_loss, kiểm tra loss giảm dần.

    Args:
        checkpoints_dir: Đường dẫn tới thư mục checkpoints.

    Returns:
        True nếu loss giảm dần, False nếu không.
    """
    ckpt_path = Path(checkpoints_dir)
    if not ckpt_path.exists():
        return None  # SKIP

    ckpt_files = sorted(ckpt_path.glob("epoch_*.pth"))
    if not ckpt_files:
        print("  Không tìm thấy checkpoint files (epoch_*.pth)")
        return False

    if len(ckpt_files) < 2:
        print(f"  Chỉ có {len(ckpt_files)} checkpoint — cần ít nhất 2 để so sánh loss")
        return False

    try:
        import torch
    except ImportError:
        print("  Không thể import torch")
        return False

    # Trích xuất epoch và val_loss từ mỗi checkpoint
    epoch_losses = []
    for ckpt_file in ckpt_files:
        try:
            checkpoint = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
            epoch = checkpoint.get("epoch")
            val_loss = checkpoint.get("val_loss")

            if epoch is None or val_loss is None:
                print(f"  {ckpt_file.name}: thiếu 'epoch' hoặc 'val_loss'")
                return False

            epoch_losses.append((epoch, val_loss, ckpt_file.name))
        except Exception as e:
            print(f"  Lỗi đọc {ckpt_file.name}: {e}")
            return False

    # Sắp xếp theo epoch
    epoch_losses.sort(key=lambda x: x[0])

    print("  Checkpoint losses:")
    for epoch, loss, name in epoch_losses:
        print(f"    Epoch {epoch:2d}: val_loss = {loss:.4f} ({name})")

    # Kiểm tra loss giảm dần (so sánh epoch đầu và cuối)
    first_loss = epoch_losses[0][1]
    last_loss = epoch_losses[-1][1]

    if last_loss < first_loss:
        print(f"  Loss giảm: {first_loss:.4f} → {last_loss:.4f} ✓")
        return True
    else:
        print(f"  Loss KHÔNG giảm: {first_loss:.4f} → {last_loss:.4f}")
        return False


def run_all_checks() -> None:
    """Chạy tất cả kiểm tra, in PASS/FAIL/SKIP cho mỗi bước."""
    checks = [
        ("1. Verify Segments", lambda: verify_segments("segments")),
        ("2. Verify Mixtures", lambda: verify_mixtures("data_2mix")),
        ("3. Verify DataLoader", lambda: verify_dataloader("data_2mix")),
        ("4. Verify Training", lambda: verify_training("checkpoints")),
    ]

    print("=" * 60)
    print("DPRNN-2mix Pipeline Verification")
    print("=" * 60)

    results = []
    for name, check_fn in checks:
        print(f"\n[{name}]")
        result = check_fn()

        if result is None:
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"

        results.append((name, status))
        print(f"  → {status}")

    # Tổng kết
    print("\n" + "=" * 60)
    print("Summary:")
    print("-" * 60)
    for name, status in results:
        print(f"  {name}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    run_all_checks()
