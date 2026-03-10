"""Model checkpoint validation for ML workers.

Validates that required model checkpoint files exist at configured paths
before the worker starts processing tasks. Exits with code 1 if any
checkpoints are missing.
"""

from __future__ import annotations

import os
import sys

import structlog

logger = structlog.get_logger(__name__)


def validate_checkpoints(
    dprnn_dir: str,
    pipeline_dir: str | None = None,
) -> list[str]:
    """Check that required model checkpoint directories and files exist.

    Parameters
    ----------
    dprnn_dir:
        Path to the DPRNN checkpoint directory.  Must exist and contain
        at least one ``.pth`` or ``.pt`` file.
    pipeline_dir:
        Optional path to the pipeline model directory.  If provided, the
        directory must exist.

    Returns
    -------
    list[str]
        A list of human-readable error messages.  Empty when all checks pass.
    """
    errors: list[str] = []

    # --- DPRNN checkpoint directory ---
    if not os.path.isdir(dprnn_dir):
        errors.append(
            f"DPRNN checkpoint directory does not exist: {dprnn_dir}"
        )
    else:
        checkpoint_files = [
            f
            for f in os.listdir(dprnn_dir)
            if f.endswith(".pth") or f.endswith(".pt")
        ]
        if not checkpoint_files:
            errors.append(
                f"DPRNN checkpoint directory contains no .pth or .pt files: {dprnn_dir}"
            )

    # --- Pipeline model directory (optional) ---
    if pipeline_dir is not None and not os.path.isdir(pipeline_dir):
        errors.append(
            f"Pipeline model directory does not exist: {pipeline_dir}"
        )

    return errors


def validate_and_exit_on_failure(
    dprnn_dir: str,
    pipeline_dir: str | None = None,
) -> None:
    """Validate checkpoints and exit the process if any are missing.

    Calls :func:`validate_checkpoints` and, when errors are found, logs
    each one at ERROR level via *structlog* then calls ``sys.exit(1)``.
    """
    errors = validate_checkpoints(dprnn_dir, pipeline_dir)

    if errors:
        for error in errors:
            logger.error("checkpoint_validation_failed", detail=error)
        sys.exit(1)

    logger.info(
        "checkpoint_validation_passed",
        dprnn_dir=dprnn_dir,
        pipeline_dir=pipeline_dir,
    )
