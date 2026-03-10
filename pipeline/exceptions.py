"""Custom exceptions for the Audio Diarization Pipeline."""


class AuthenticationError(Exception):
    """Raised when HF_TOKEN is missing or invalid."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = (
                "HuggingFace token is missing or invalid. "
                "Please set HF_TOKEN in your .env file. "
                "Get your token at https://huggingface.co/settings/tokens"
            )
        super().__init__(message)


class AudioFormatError(Exception):
    """Raised for unsupported audio/video formats."""

    SUPPORTED_FORMATS = (".wav", ".mp4", ".mkv", ".avi")

    def __init__(self, message: str | None = None, file_path: str | None = None):
        if message is None:
            message = (
                f"Unsupported audio/video format"
                f"{f': {file_path}' if file_path else ''}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        super().__init__(message)


class GPUMemoryError(Exception):
    """Raised on out-of-memory GPU errors."""

    def __init__(self, message: str | None = None, chunk_index: int | None = None):
        if message is None:
            chunk_info = f" while processing chunk {chunk_index}" if chunk_index is not None else ""
            message = (
                f"GPU out-of-memory{chunk_info}. "
                "Try reducing max_chunk_duration_sec in PipelineConfig."
            )
        super().__init__(message)
        self.chunk_index = chunk_index
