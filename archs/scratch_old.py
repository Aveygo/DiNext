# Heavily modified from https://github.com/chuanyangjin/fast-DiT

import torch
import torch.nn as nn
import numpy as np
import math
from archs.next import Next, LayerNorm
from timm.models.layers import trunc_normal_, DropPath
import torch.utils.checkpoint as checkpoint

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4, **block_kwargs):
        super().__init__()

        self.norm1 = torch.nn.GroupNorm(1, hidden_size)
        self.norm2 = torch.nn.GroupNorm(1, hidden_size)

        self.next = Next(hidden_size, num_heads)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), out_features=hidden_size)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(),nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(-1).unsqueeze(-1) * self.next(self.modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=4,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        patch_size=2,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        self.head = nn.Sequential(*[
            torch.nn.PixelUnshuffle(patch_size),
            nn.Conv2d(in_channels * (patch_size ** 2), hidden_size, 1, 1, 0),
            nn.GELU(approximate="tanh")
        ])

        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.tail = nn.Sequential(*[
            nn.Conv2d(hidden_size, self.out_channels * (patch_size ** 2), 1, 1, 0),
            torch.nn.PixelShuffle(patch_size),
        ])

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        c = self.t_embedder(t) + self.y_embedder(y, self.training)
        x = self.head(x)
        for block in self.blocks:
            x = checkpoint.checkpoint(self.ckpt_wrapper(block), x, c)

        x = self.tail(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

def DiT_XL_2(**kwargs):
    return DiT(depth=12, hidden_size=1024, num_heads=1)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=1)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,
    'DiT-S/8':  DiT_S_8,
}
