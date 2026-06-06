import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class AudioToMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=64):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mel = self.mel_spec(x)
        mel_db = self.amplitude_to_db(mel)
        mean = mel_db.mean(dim=[-2, -1], keepdim=True)
        std = mel_db.std(dim=[-2, -1], keepdim=True)
        return (mel_db - mean) / (std + 1e-8)


class BaselineCNN(nn.Module):
    def __init__(self, num_classes, n_mels=64, channels=(32, 64, 128),
                 kernel_size=3, dropout_conv=0.2, dropout_fc=0.3, fc_dim=64):
        super().__init__()
        layers = []
        in_c = 1
        for out_c in channels:
            layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(dropout_conv),
            ])
            in_c = out_c
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels[-1], fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


class CRNNModel(nn.Module):
    def __init__(self, num_classes, n_mels=64, conv_channels=(32, 64, 128),
                 lstm_hidden=256, lstm_layers=2, lstm_dropout=0.3, fc_dim=256,
                 dropout_conv=0.2, dropout_fc=0.3):
        super().__init__()
        conv_layers = []
        in_c = 1
        for out_c in conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Dropout2d(dropout_conv),
            ])
            in_c = out_c
        self.conv = nn.Sequential(*conv_layers)

        pool_factor = 2 ** len(conv_channels)
        lstm_input_dim = conv_channels[-1] * (n_mels // pool_factor)
        self.lstm = nn.LSTM(
            lstm_input_dim, lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True, dropout=lstm_dropout if lstm_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class RoPESelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rope(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPESelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpectrogramTransformer(nn.Module):
    def __init__(self, num_classes, n_mels=64, dim=256, depth=6, heads=8,
                 mlp_ratio=4, dropout=0.1, dropout_cls=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_mels, dim),
            nn.LayerNorm(dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])
        return self.classifier(x)
