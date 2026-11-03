"""dots.tts patch encoder and flow-matching DiT, ported from the official release (Apache-2.0).

Module and parameter names follow the release so its weights load unchanged and exports load in the
official runtime. Only the training path is kept."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta):
        super().__init__()
        self.dim, self.theta = dim, float(theta)

    @torch.autocast("cuda", enabled=False)
    def forward(self, positions):
        inv_freq = 1.0 / self.theta ** (
            torch.arange(0, self.dim, 2, device=positions.device, dtype=torch.float32) / self.dim
        )
        freqs = positions.float()[..., None] * inv_freq
        return torch.cat((freqs, freqs), dim=-1)


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, *, qk_norm=False, rotary_theta=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.rotary = None if rotary_theta is None else RotaryEmbedding(head_dim, rotary_theta)

    def forward(self, x, mask, positions=None):
        """mask: (B, L, L) bool, True where the query may attend to the key."""
        b, length, _ = x.shape
        q, k, v = (
            proj(x).view(b, length, self.num_heads, -1).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rotary is not None:
            angles = self.rotary(positions).unsqueeze(1)
            cos, sin = angles.cos(), angles.sin()
            q, k = ((t * cos + rotate_half(t) * sin).to(t.dtype) for t in (q, k))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1))
        return self.o_proj(out.transpose(1, 2).reshape(b, length, -1))


class Mlp(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size, activation):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.act = activation
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_hidden_size):
        super().__init__()
        # The release builds these layers without qk-norm or rotary positions.
        self.attn = Attention(hidden_size, num_heads)
        self.attn_norm = nn.RMSNorm(hidden_size)
        self.ffn = Mlp(hidden_size, ffn_hidden_size, nn.SiLU())
        self.ffn_norm = nn.RMSNorm(hidden_size)

    def forward(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.ffn(self.ffn_norm(x))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(config["hidden_size"], config["num_heads"], config["ffn_hidden_size"])
            for _ in range(config["num_layers"])
        )

    def forward(self, x):
        length = x.shape[1]
        mask = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()[None]
        for layer in self.layers:
            x = layer(x, mask)
        return x


class PatchEncoder(nn.Module):
    """Causal semantic encoder: (B, 4N, 128) raw VAE latents -> (B, N, LLM width)."""

    def __init__(self, latent_dim, out_dim, patch_size, config):
        super().__init__()
        self.ds_proj = nn.Conv1d(latent_dim, latent_dim, kernel_size=2, stride=2)
        self.in_proj = nn.Linear(latent_dim, config["hidden_size"])
        self.encoder = Encoder(config)
        self.merge = patch_size // 2
        self.out_proj = nn.Linear(config["hidden_size"] * self.merge, out_dim)

    def forward(self, latents):
        # The release's causal convolution pads one frame on the left before the stride-2 window.
        x = self.ds_proj(F.pad(latents.transpose(1, 2), (1, 0))).transpose(1, 2)
        x = self.encoder(self.in_proj(x))
        b, length, width = x.shape
        return self.out_proj(x.reshape(b, length // self.merge, self.merge * width))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_size=256):
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t):
        half = self.frequency_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["hidden_size"]
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = Attention(
            width,
            config["num_heads"],
            qk_norm=config["qk_norm"],
            rotary_theta=config["rotary_theta"],
        )
        self.ffn = Mlp(width, config["ffn_hidden_size"], nn.GELU(approximate="tanh"))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x, condition, mask, positions):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaLN_modulation(condition).chunk(
            6, dim=1
        )
        x = x + gate_a.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_a, scale_a), mask, positions
        )
        return x + gate_f.unsqueeze(1) * self.ffn(modulate(self.norm2(x), shift_f, scale_f))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    def __init__(self, in_dim, out_dim, config):
        super().__init__()
        width = config["hidden_size"]
        self.input_layer = nn.Linear(in_dim, width)
        self.time_embedder = TimestepEmbedder(width)
        self.blocks = nn.ModuleList(DiTBlock(config) for _ in range(config["num_layers"]))
        self.output_layer = FinalLayer(width, out_dim)

    def forward(self, x, t, condition, mask, positions):
        """condition: (B, width) per-sample adaLN input added to the time embedding."""
        c = self.time_embedder(t) + condition
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x, c, mask, positions)
        return self.output_layer(x, c)
