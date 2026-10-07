"""Inference-only ECAPA-TDNN adapter.

The network definition is adapted from TaoRuijie's ECAPA-TDNN implementation:
https://github.com/TaoRuijie/ECAPA-TDNN (MIT license).  The upstream checkout in
``Diarization-Model`` remains the source/reference; this module deliberately
contains only the speaker encoder needed at runtime.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, 1),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, dilation, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, 1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        padding = math.floor(kernel_size / 2) * dilation
        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size, dilation=dilation, padding=padding)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(self.nums)])
        self.conv3 = nn.Conv1d(width * scale, planes, 1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.bn1(self.relu(self.conv1(x)))
        pieces = torch.split(out, self.width, 1)
        merged = None
        previous = None
        for index in range(self.nums):
            previous = pieces[index] if index == 0 else previous + pieces[index]
            previous = self.bns[index](self.relu(self.convs[index](previous)))
            merged = previous if merged is None else torch.cat((merged, previous), 1)
        out = torch.cat((merged, pieces[self.nums]), 1)
        out = self.bn3(self.relu(self.conv3(out)))
        return self.se(out) + residual


class PreEmphasis(nn.Module):
    def __init__(self, coef=0.97):
        super().__init__()
        self.register_buffer(
            "flipped_filter", torch.tensor([-coef, 1.0]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        return F.conv1d(F.pad(x.unsqueeze(1), (1, 0), "reflect"), self.flipped_filter).squeeze(1)


class ECAPA_TDNN(nn.Module):
    def __init__(self, C=1024):
        super().__init__()
        self.torchfbank = nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80,
            ),
        )
        self.conv1 = nn.Conv1d(80, C, 5, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, 3, 2)
        self.layer2 = Bottle2neck(C, C, 3, 3)
        self.layer3 = Bottle2neck(C, C, 3, 4)
        self.layer4 = nn.Conv1d(3 * C, 1536, 1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, 1), nn.ReLU(), nn.BatchNorm1d(256), nn.Tanh(),
            nn.Conv1d(256, 1536, 1), nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x, aug=False):
        del aug  # augmentation is intentionally unavailable in the runtime adapter
        with torch.inference_mode():
            x = (self.torchfbank(x) + 1e-6).log()
            x = x - x.mean(dim=-1, keepdim=True)
        x = self.bn1(self.relu(self.conv1(x)))
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = self.relu(self.layer4(torch.cat((x1, x2, x3), dim=1)))
        length = x.size(-1)
        global_x = torch.cat((
            x,
            x.mean(dim=2, keepdim=True).repeat(1, 1, length),
            torch.sqrt(x.var(dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, length),
        ), dim=1)
        weights = self.attention(global_x)
        mean = torch.sum(x * weights, dim=2)
        std = torch.sqrt((torch.sum((x ** 2) * weights, dim=2) - mean ** 2).clamp(min=1e-4))
        return self.bn6(self.fc6(self.bn5(torch.cat((mean, std), 1))))


def filter_speaker_encoder_state(checkpoint):
    """Return only speaker_encoder.* weights and reject malformed checkpoints."""
    if not isinstance(checkpoint, dict):
        raise ValueError("ECAPA checkpoint must be a state-dict mapping")
    prefix = "speaker_encoder."
    filtered = {key[len(prefix):]: value for key, value in checkpoint.items() if key.startswith(prefix)}
    if not filtered:
        raise ValueError("ECAPA checkpoint contains no speaker_encoder.* weights")
    return filtered


class ECAPAEncoder:
    """Strict checkpoint loader and normalized 192-D embedding interface."""
    sample_rate = 16000
    embedding_size = 192

    def __init__(self, checkpoint_path, device="cuda"):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("ECAPA diarization requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for ECAPA diarization but is not available")
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"ECAPA checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = filter_speaker_encoder_state(checkpoint)
        self.model = ECAPA_TDNN(C=1024).to(device=self.device, dtype=torch.float32)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        del checkpoint, state
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def embed(self, audio):
        samples = torch.as_tensor(audio, device=self.device, dtype=torch.float32).reshape(1, -1)
        embedding = self.model(samples, aug=False)
        return F.normalize(embedding, p=2, dim=1)[0]

    def warmup(self, sample_count=32000):
        self.embed(torch.zeros(sample_count, dtype=torch.float32))
