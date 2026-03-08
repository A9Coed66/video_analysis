"""Compatibility patches for pyannote.audio 4.x with torchaudio 2.6+.

Fixes:
1. torchaudio.list_audio_backends() removed in torchaudio 2.6+
2. torchcodec.AudioDecoder not available on Windows — patches
   pyannote.audio.core.io to use soundfile instead.
3. torchaudio.load/save use torchcodec by default in 2.10+ which
   fails on Windows without FFmpeg DLLs — patches to use soundfile.

This module must be imported BEFORE any pyannote.audio imports.
"""

from __future__ import annotations

import torchaudio

# Patch 1: torchaudio.list_audio_backends was removed in 2.6+
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# Patch 2: torchaudio.load/save default to torchcodec in 2.10+ which
# crashes on Windows without FFmpeg DLLs. Patch to use soundfile.
_original_load = torchaudio.load
_original_save = torchaudio.save


def _patched_torchaudio_load(filepath, *args, **kwargs):
    """Load audio using soundfile backend to avoid torchcodec crash."""
    import soundfile as sf
    import torch

    data, sample_rate = sf.read(str(filepath), dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    return waveform, sample_rate


def _patched_torchaudio_save(filepath, waveform, sample_rate, *args, **kwargs):
    """Save audio using soundfile backend to avoid torchcodec crash."""
    import soundfile as sf

    data = waveform.cpu().numpy()
    if data.ndim == 2:
        data = data.T  # (channels, samples) -> (samples, channels)
    sf.write(str(filepath), data, sample_rate)


torchaudio.load = _patched_torchaudio_load
torchaudio.save = _patched_torchaudio_save


def _to_mono(waveform):
    """Convert waveform to mono by averaging channels."""
    if waveform.shape[0] > 1:
        return waveform.mean(dim=0, keepdim=True)
    return waveform


def apply_patches():
    """Apply all compatibility patches for pyannote.audio 4.x."""
    import types

    import soundfile as sf
    import torch

    from pyannote.audio.core import io as pyannote_io

    # Patch get_audio_metadata to use soundfile instead of AudioDecoder
    def _patched_get_audio_metadata(file):
        if isinstance(file, dict):
            audio_path = str(file.get("audio", file.get("uri", "")))
        else:
            audio_path = str(file)

        info = sf.info(audio_path)
        duration = info.frames / info.samplerate if info.samplerate > 0 else 0.0
        return types.SimpleNamespace(
            num_channels=info.channels,
            sample_rate=info.samplerate,
            num_frames=info.frames,
            duration_seconds_from_header=duration,
            duration_seconds=duration,
        )

    pyannote_io.get_audio_metadata = _patched_get_audio_metadata

    # Patch Audio.__call__ to use soundfile
    def _patched_audio_call(self, file, **kwargs):
        import numpy as np

        if isinstance(file, dict):
            if "waveform" in file and "sample_rate" in file:
                waveform = file["waveform"]
                sample_rate = file["sample_rate"]
                if isinstance(waveform, np.ndarray):
                    waveform = torch.from_numpy(waveform)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                return _to_mono(waveform), sample_rate
            audio_path = str(file.get("audio", file.get("uri", "")))
        else:
            audio_path = str(file)

        data, sample_rate = sf.read(audio_path, dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
        waveform = _to_mono(waveform)

        target_sr = getattr(self, "sample_rate", None)
        if target_sr and target_sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
            sample_rate = target_sr

        return waveform, sample_rate

    pyannote_io.Audio.__call__ = _patched_audio_call

    # Patch Audio.crop to use soundfile
    def _patched_crop(self, file, segment, **kwargs):
        if isinstance(file, dict):
            audio_path = str(file.get("audio", file.get("uri", "")))
        else:
            audio_path = str(file)

        info = sf.info(audio_path)
        sr = info.samplerate
        start_frame = max(0, int(segment.start * sr))
        end_frame = min(info.frames, int(segment.end * sr))

        # Expected segment duration
        expected_duration = segment.end - segment.start

        target_sr = getattr(self, "sample_rate", None) or sr
        expected_frames = int(expected_duration * target_sr)

        if end_frame - start_frame <= 0:
            return torch.zeros(1, max(expected_frames, 0)), target_sr

        data, sample_rate = sf.read(
            audio_path, start=start_frame, stop=end_frame, dtype="float32"
        )
        waveform = torch.from_numpy(data)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
        waveform = _to_mono(waveform)

        if target_sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
            sample_rate = target_sr

        # Pad to expected duration (pyannote batches crops and expects uniform size)
        if waveform.shape[-1] < expected_frames:
            pad_size = expected_frames - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        return waveform, sample_rate

    pyannote_io.Audio.crop = _patched_crop
