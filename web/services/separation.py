"""Separation Service — wraps voice_separator/06_predict.py for web usage."""

import importlib.util
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path

import soundfile as sf
import torch

# Load voice_separator/06_predict.py via importlib (filename starts with digit)
_PREDICT_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "voice_separator", "06_predict.py",
)
_spec = importlib.util.spec_from_file_location("voice_separator_predict", _PREDICT_SCRIPT)
_predict_module = importlib.util.module_from_spec(_spec)
sys.modules["voice_separator_predict"] = _predict_module
_spec.loader.exec_module(_predict_module)

find_best_checkpoint = _predict_module.find_best_checkpoint
load_audio = _predict_module.load_audio
load_model = _predict_module.load_model
run_separation = _predict_module.separate
SAMPLE_RATE = _predict_module.SAMPLE_RATE

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "voice_separator", "checkpoints_premium",
)


class SeparationService:
    """DPRNN-TasNet voice separation. Finds best checkpoint on init."""

    def __init__(self, checkpoint_dir: str | None = None) -> None:
        ckpt_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        best_ckpt = find_best_checkpoint(ckpt_dir)
        if best_ckpt is None:
            raise FileNotFoundError(f"No valid checkpoint found in {ckpt_dir}")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._model = load_model(best_ckpt, self._device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from {best_ckpt}: {exc}") from exc

        logger.info("SeparationService ready (device=%s, ckpt=%s)", self._device, best_ckpt)

    def separate(self, file_path: str) -> list[str]:
        """Separate mixed audio into two speaker tracks. Returns [spk1_path, spk2_path]."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        try:
            audio = load_audio(file_path)
            sources = run_separation(self._model, audio, self._device)

            out_dir = tempfile.mkdtemp(prefix="sep_")
            uid = uuid.uuid4().hex[:8]
            stem = Path(file_path).stem
            paths = []
            for i, src in enumerate(sources[:2], start=1):
                p = os.path.join(out_dir, f"{stem}_{uid}_spk{i}.wav")
                sf.write(p, src, SAMPLE_RATE, subtype="FLOAT")
                paths.append(p)

            logger.info("Separation complete: %s", paths)
            return paths
        except Exception as exc:
            logger.exception("Separation failed for %s", file_path)
            raise RuntimeError(f"Separation failed: {exc}") from exc
