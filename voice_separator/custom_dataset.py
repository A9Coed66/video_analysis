"""
custom_dataset.py — PyTorch Dataset kết nối data_2mix với DPRNN-TasNet.

Đọc dữ liệu từ cấu trúc data_2mix/{split}/{mix,s1,s2}/{index:06d}.wav
và trả về (mixture [T], sources [2, T]) dạng float32 tensor, T=64000.
"""

from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

# Constants
SAMPLE_RATE = 16000
SEGMENT_SAMPLES = 64000  # 4s at 16kHz


class MixDataset(Dataset):
    """Dataset đọc từ data_2mix/{split}/{mix,s1,s2}/"""

    def __init__(self, data_dir: str, split: str = "tr"):
        """Khởi tạo, xác thực số file mix == s1 == s2.

        Args:
            data_dir: Đường dẫn tới thư mục data_2mix.
            split: Tên split ("tr", "cv", "tt").

        Raises:
            FileNotFoundError: Nếu thư mục split không tồn tại.
            ValueError: Nếu số file mix, s1, s2 không khớp.
        """
        self.split_dir = Path(data_dir) / split
        self.mix_dir = self.split_dir / "mix"
        self.s1_dir = self.split_dir / "s1"
        self.s2_dir = self.split_dir / "s2"

        # Xác thực thư mục tồn tại
        for d in (self.mix_dir, self.s1_dir, self.s2_dir):
            if not d.exists():
                raise FileNotFoundError(f"Thư mục không tồn tại: {d}")

        # Liệt kê và sắp xếp file WAV
        self.mix_files = sorted(self.mix_dir.glob("*.wav"))
        self.s1_files = sorted(self.s1_dir.glob("*.wav"))
        self.s2_files = sorted(self.s2_dir.glob("*.wav"))

        n_mix = len(self.mix_files)
        n_s1 = len(self.s1_files)
        n_s2 = len(self.s2_files)

        if not (n_mix == n_s1 == n_s2):
            raise ValueError(
                f"File count mismatch: mix={n_mix}, s1={n_s1}, s2={n_s2}"
            )

    def __len__(self) -> int:
        """Trả về số lượng mixture."""
        return len(self.mix_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Trả về (mixture [T], sources [2, T]) dạng float32.

        Args:
            idx: Chỉ số sample.

        Returns:
            Tuple (mixture, sources) với:
                - mixture: torch.Tensor float32 shape [T], T=64000
                - sources: torch.Tensor float32 shape [2, T]

        Raises:
            RuntimeError: Nếu file WAV không đọc được.
        """
        mix_path = self.mix_files[idx]
        s1_path = self.s1_files[idx]
        s2_path = self.s2_files[idx]

        mixture = self._load_wav(mix_path)
        s1 = self._load_wav(s1_path)
        s2 = self._load_wav(s2_path)

        sources = torch.stack([s1, s2], dim=0)  # [2, T]
        return mixture, sources

    def _load_wav(self, path: Path) -> torch.Tensor:
        """Đọc file WAV, trả về float32 tensor.

        Args:
            path: Đường dẫn tới file WAV.

        Returns:
            torch.Tensor float32 shape [T].

        Raises:
            RuntimeError: Nếu file không đọc được.
        """
        try:
            audio, _sr = sf.read(str(path), dtype="float32")
        except Exception as e:
            raise RuntimeError(f"Không đọc được file WAV: {path}") from e

        return torch.from_numpy(audio).float()
