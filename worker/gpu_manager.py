"""GPU resource manager for ML workers.

Provides VRAM monitoring, GPU info reporting, and memory cleanup.
Handles environments without CUDA gracefully.
"""

from __future__ import annotations

import gc

import structlog

logger = structlog.get_logger(__name__)


class GPUManager:
    """Manage GPU resources for a single device.

    Parameters
    ----------
    device_index:
        CUDA device ordinal (default ``0``).
    """

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_gpu_info(self) -> dict:
        """Return GPU name, VRAM stats (MB), and CUDA version.

        If CUDA is unavailable the dict contains ``{"available": False}``.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return {"available": False}

            name = torch.cuda.get_device_name(self.device_index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device_index)
            used_bytes = total_bytes - free_bytes

            return {
                "available": True,
                "device_index": self.device_index,
                "name": name,
                "vram_total_mb": round(total_bytes / (1024 * 1024), 2),
                "vram_used_mb": round(used_bytes / (1024 * 1024), 2),
                "vram_free_mb": round(free_bytes / (1024 * 1024), 2),
                "cuda_version": torch.version.cuda,
            }
        except Exception:
            logger.warning(
                "gpu_info_failed",
                device_index=self.device_index,
                exc_info=True,
            )
            return {"available": False}

    def check_vram_available(self, threshold: float = 0.9) -> bool:
        """Return ``True`` if VRAM usage ratio is below *threshold*.

        Returns ``False`` when CUDA is not available.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return False

            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device_index)
            used_bytes = total_bytes - free_bytes
            usage_ratio = used_bytes / total_bytes if total_bytes > 0 else 1.0
            return usage_ratio < threshold
        except Exception:
            logger.warning(
                "vram_check_failed",
                device_index=self.device_index,
                exc_info=True,
            )
            return False

    def cleanup(self) -> None:
        """Free cached GPU memory and run Python garbage collection."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("gpu_cache_cleared", device_index=self.device_index)
        except Exception:
            logger.warning(
                "gpu_cleanup_failed",
                device_index=self.device_index,
                exc_info=True,
            )
        gc.collect()

    def log_gpu_info(self) -> None:
        """Log GPU details at INFO level (called on worker startup)."""
        info = self.get_gpu_info()
        if info.get("available"):
            logger.info(
                "gpu_info",
                device_index=info["device_index"],
                name=info["name"],
                vram_total_mb=info["vram_total_mb"],
                vram_used_mb=info["vram_used_mb"],
                vram_free_mb=info["vram_free_mb"],
                cuda_version=info["cuda_version"],
            )
        else:
            logger.warning("gpu_not_available", device_index=self.device_index)
