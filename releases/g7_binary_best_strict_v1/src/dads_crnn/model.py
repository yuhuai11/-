from __future__ import annotations

import torch
from torch import nn


class ZeroInitializedTemporalAttention(nn.Linear):
    """A linear scorer whose construction consumes no random numbers."""

    def __init__(self, features: int) -> None:
        super().__init__(features, 1, bias=False)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)


class CRNN(nn.Module):
    def __init__(
        self,
        *,
        n_mels: int,
        conv_channels: list[int],
        gru_hidden: int,
        gru_layers: int,
        bidirectional: bool,
        dropout: float,
        temporal_pooling: str = "mean",
    ) -> None:
        super().__init__()
        if temporal_pooling not in {"mean", "attention"}:
            raise ValueError(
                "CRNN temporal_pooling must be 'mean' or 'attention', "
                f"got {temporal_pooling!r}"
            )
        self.temporal_pooling = temporal_pooling
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 2)),
                    nn.Dropout2d(dropout),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)

        reduced_mels = n_mels // (2 ** len(conv_channels))
        rnn_input = conv_channels[-1] * reduced_mels
        self.rnn = nn.GRU(
            input_size=rnn_input,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        rnn_out = gru_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_out, rnn_out // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(rnn_out // 2, 1),
        )
        self.temporal_attention: nn.Linear | None = None
        if temporal_pooling == "attention":
            # Construct this after the classifier so all shared G2 parameters keep
            # the same seeded initialization. Zero scores make the initial softmax
            # exactly uniform, nesting the original mean-pooling model. The custom
            # scorer also avoids advancing the CPU RNG used by later operations.
            self.temporal_attention = ZeroInitializedTemporalAttention(rnn_out)

    def temporal_pool(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.temporal_attention is None:
            return sequence.mean(dim=1)
        scores = self.temporal_attention(sequence).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        batch, channels, mel_bins, frames = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, frames, channels * mel_bins)
        x, _ = self.rnn(x)
        x = self.temporal_pool(x)
        return self.classifier(x).squeeze(1)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        maximum = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        return x * torch.sigmoid(avg + maximum)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("CBAM spatial kernel size must be 3 or 7")
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.amax(x, dim=1, keepdim=True)
        weights = torch.sigmoid(self.conv(torch.cat((avg, maximum), dim=1)))
        return x * weights


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int, spatial_kernel: int) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        dropout: float,
        attention: bool,
        attention_reduction: int,
        spatial_kernel: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = (
            CBAM(out_channels, attention_reduction, spatial_kernel) if attention else nn.Identity()
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        out = self.dropout(out)
        return self.relu(out + identity)


class ResNet10CBAM(nn.Module):
    """Four-stage ResNet10 with configurable CBAM placement for acoustic images."""

    def __init__(
        self,
        *,
        channels: list[int],
        dropout: float,
        attention_stages: list[int],
        attention_reduction: int,
        spatial_kernel: int,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("ResNet10CBAM requires four stage channel values")
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        stages: list[nn.Module] = []
        in_channels = channels[0]
        attention_set = {int(stage) for stage in attention_stages}
        for stage_index, out_channels in enumerate(channels):
            stages.append(
                ResidualBlock(
                    in_channels,
                    out_channels,
                    stride=1 if stage_index == 0 else 2,
                    dropout=dropout,
                    attention=stage_index in attention_set,
                    attention_reduction=attention_reduction,
                    spatial_kernel=spatial_kernel,
                )
            )
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x).squeeze(1)
