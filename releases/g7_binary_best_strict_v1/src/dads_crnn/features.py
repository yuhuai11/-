from __future__ import annotations

import torch
from torch import nn


def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    min_mel = _hz_to_mel(torch.tensor(float(f_min)))
    max_mel = _hz_to_mel(torch.tensor(float(f_max)))
    mel_points = torch.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for mel_idx in range(n_mels):
        left, center, right = int(bins[mel_idx]), int(bins[mel_idx + 1]), int(bins[mel_idx + 2])
        if center > left:
            fb[mel_idx, left:center] = torch.linspace(0, 1, center - left)
        if right > center:
            fb[mel_idx, center:right] = torch.linspace(1, 0, right - center)
    return fb


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        f_min: float,
        f_max: float,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        mel_fb = build_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)
        self.register_buffer("mel_fb", mel_fb, persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        mel = torch.matmul(self.mel_fb, power)
        log_mel = torch.log(mel.clamp_min(1e-10))
        mean = log_mel.mean(dim=(1, 2), keepdim=True)
        std = log_mel.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        return ((log_mel - mean) / std).unsqueeze(1)


def build_dct_matrix(n_mfcc: int, n_mels: int) -> torch.Tensor:
    """Return an orthonormal DCT-II matrix compatible with common MFCC tools."""
    mel_index = torch.arange(n_mels, dtype=torch.float32).unsqueeze(0)
    coeff_index = torch.arange(n_mfcc, dtype=torch.float32).unsqueeze(1)
    matrix = torch.cos(torch.pi / n_mels * (mel_index + 0.5) * coeff_index)
    matrix[0] *= (1.0 / n_mels) ** 0.5
    if n_mfcc > 1:
        matrix[1:] *= (2.0 / n_mels) ** 0.5
    return matrix


class MFCCSpectrogram(nn.Module):
    """Convert waveforms to normalized MFCC time-series images."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_mfcc: int,
        n_mels: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        f_min: float,
        f_max: float,
        preemphasis: float = 0.0,
        window_type: str = "hamming",
    ) -> None:
        super().__init__()
        if n_mfcc > n_mels:
            raise ValueError("n_mfcc must not exceed n_mels")
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.preemphasis = float(preemphasis)
        if window_type == "hamming":
            window = torch.hamming_window(win_length)
        elif window_type == "hann":
            window = torch.hann_window(win_length)
        else:
            raise ValueError("window_type must be 'hamming' or 'hann'")
        self.window_type = window_type
        self.register_buffer("window", window, persistent=False)
        self.register_buffer(
            "mel_fb",
            build_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max),
            persistent=False,
        )
        self.register_buffer("dct", build_dct_matrix(n_mfcc, n_mels), persistent=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if self.preemphasis:
            first = waveform[:, :1]
            rest = waveform[:, 1:] - self.preemphasis * waveform[:, :-1]
            waveform = torch.cat((first, rest), dim=1)

        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        mel = torch.matmul(self.mel_fb, power)
        log_mel = torch.log(mel.clamp_min(1e-10))
        mfcc = torch.matmul(self.dct, log_mel)

        # Normalize each clip while retaining the coefficient-by-time layout.
        mean = mfcc.mean(dim=(1, 2), keepdim=True)
        std = mfcc.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        return ((mfcc - mean) / std).unsqueeze(1)
