"""Unit tests for worker.checkpoint_validator."""

from __future__ import annotations

import pytest

from worker.checkpoint_validator import validate_checkpoints, validate_and_exit_on_failure


# ---------------------------------------------------------------------------
# validate_checkpoints
# ---------------------------------------------------------------------------


class TestValidateCheckpoints:
    """Tests for :func:`validate_checkpoints`."""

    def test_valid_dprnn_dir_with_pth_file(self, tmp_path):
        dprnn = tmp_path / "dprnn"
        dprnn.mkdir()
        (dprnn / "model.pth").write_text("fake")

        errors = validate_checkpoints(str(dprnn))
        assert errors == []

    def test_valid_dprnn_dir_with_pt_file(self, tmp_path):
        dprnn = tmp_path / "dprnn"
        dprnn.mkdir()
        (dprnn / "model.pt").write_text("fake")

        errors = validate_checkpoints(str(dprnn))
        assert errors == []

    def test_dprnn_dir_missing(self, tmp_path):
        missing = str(tmp_path / "nonexistent")
        errors = validate_checkpoints(missing)
        assert len(errors) == 1
        assert "does not exist" in errors[0]
        assert missing in errors[0]

    def test_dprnn_dir_empty_no_checkpoint_files(self, tmp_path):
        dprnn = tmp_path / "dprnn"
        dprnn.mkdir()
        # directory exists but has no .pth/.pt files
        (dprnn / "readme.txt").write_text("not a checkpoint")

        errors = validate_checkpoints(str(dprnn))
        assert len(errors) == 1
        assert "no .pth or .pt files" in errors[0]

    def test_pipeline_dir_missing(self, tmp_path):
        dprnn = tmp_path / "dprnn"
        dprnn.mkdir()
        (dprnn / "model.pth").write_text("fake")

        missing_pipeline = str(tmp_path / "pipeline_missing")
        errors = validate_checkpoints(str(dprnn), pipeline_dir=missing_pipeline)
        assert len(errors) == 1
        assert "Pipeline model directory does not exist" in errors[0]
        assert missing_pipeline in errors[0]

    def test_pipeline_dir_none_is_skipped(self, tmp_path):
        dprnn = tmp_path / "dprnn"
        dprnn.mkdir()
        (dprnn / "model.pth").write_text("fake")

        errors = validate_checkpoints(str(dprnn), pipeline_dir=None)
        assert errors == []

    def test_pipeline_dir_valid(self, tmp_path):
        dprnn = tmp_path / "dprnn"
        dprnn.mkdir()
        (dprnn / "model.pth").write_text("fake")

        pipeline = tmp_path / "pipeline"
        pipeline.mkdir()

        errors = validate_checkpoints(str(dprnn), pipeline_dir=str(pipeline))
        assert errors == []

    def test_both_dirs_missing_returns_multiple_errors(self, tmp_path):
        errors = validate_checkpoints(
            str(tmp_path / "no_dprnn"),
            pipeline_dir=str(tmp_path / "no_pipeline"),
        )
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# validate_and_exit_on_failure
# ---------------------------------------------------------------------------


class TestValidateAndExitOnFailure:
    """Tests for :func:`validate_and_exit_on_failure`."""

    def test_exits_with_code_1_when_checkpoints_missing(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            validate_and_exit_on_failure(str(tmp_path / "missing"))
        assert exc_info.value.code == 1

    def test_does_not_exit_when_checkpoints_valid(self, tmp_path):
        dprnn = tmp_path / "dprnn"
        dprnn.mkdir()
        (dprnn / "model.pth").write_text("fake")

        # Should not raise
        validate_and_exit_on_failure(str(dprnn))

    def test_exits_when_pipeline_dir_missing(self, tmp_path):
        dprnn = tmp_path / "dprnn"
        dprnn.mkdir()
        (dprnn / "model.pth").write_text("fake")

        with pytest.raises(SystemExit) as exc_info:
            validate_and_exit_on_failure(
                str(dprnn), pipeline_dir=str(tmp_path / "missing_pipeline")
            )
        assert exc_info.value.code == 1
