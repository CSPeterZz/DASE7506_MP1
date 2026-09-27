"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory
change. Explain it, measure its cost and perform a mechanism ablation. Merely
renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


MODEL_OPTIONS = {
    'vocab': 2048, 'context': 256, 'width': 256, 'heads': 4, 'depth': 8,
    'use_rope': True, 'rope_theta': 10000.0,
    'use_partial_rope': True, 'rope_fraction': 0.25,
    'use_xsa': True, 'xsa_last_n_layers': 6,
    'use_rmsnorm': False, 'use_swiglu': True, 'use_smear': False,
    'residual_dropout': 0.1,
    'copy_enabled': True, 'copy_dim': 32,
    'copy_initial_gate': 0.1, 'copy_fixed_gate': None,
}


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate, self.up = nn.Linear(width, hidden), nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config, use_xsa=True):
        super().__init__()
        width, self.heads = config['width'], config['heads']
        self.use_xsa = use_xsa
        head_dim = width // self.heads
        rotary_dim = round(head_dim * config['rope_fraction'])
        if (rotary_dim < 2 or rotary_dim % 2
                or not math.isclose(rotary_dim, head_dim * config['rope_fraction'])):
            raise ValueError('rope_fraction must select a positive even number of dimensions.')
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = SwiGLU(width, config['ffn_hidden'])
        self.dropout = nn.Dropout(config['residual_dropout'])
        frequencies = config['rope_theta'] ** (-torch.arange(0, rotary_dim, 2).float() / rotary_dim)
        angles = torch.arange(config['context']).float()[:, None] * frequencies
        cos, sin = torch.ones(config['context'], head_dim // 2), torch.zeros(config['context'], head_dim // 2)
        cos[:, :rotary_dim // 2], sin[:, :rotary_dim // 2] = angles.cos(), angles.sin()
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def rotate(self, x):
        length = x.shape[-2]
        cos = self.rope_cos[:length].to(x.dtype)[None, None]
        sin = self.rope_sin[:length].to(x.dtype)[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.rotate(q), self.rotate(k)
        # Each position attends only to itself and earlier input tokens.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.use_xsa:
            # XSA: https://arxiv.org/abs/2603.09078. Remove the self-value direction.
            dtype = attended.dtype
            with torch.autocast(device_type=x.device.type, enabled=False):
                direction = F.normalize(v.float(), dim=-1, eps=1e-6)
                output = attended.float()
                attended = (output - (output * direction).sum(-1, keepdim=True) * direction).to(dtype)
        x = x + self.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.dropout(self.mlp(self.norm2(x)))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        first_xsa = config['depth'] - config['xsa_last_n_layers']
        self.blocks = nn.ModuleList([
            Block(config, config['use_xsa'] and index >= first_xsa)
            for index in range(config['depth'])
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.copy_enabled = config['copy_enabled']
        if self.copy_enabled:
            self.copy_query = nn.Linear(width, config['copy_dim'], bias=False)
            self.copy_key = nn.Linear(width, config['copy_dim'], bias=False)
            self.copy_scale = config['copy_dim'] ** -0.5
            self.fixed_gate = config['copy_fixed_gate']
            if self.fixed_gate is None:
                self.copy_gate = nn.Linear(width, 1)
        self.apply(self.initialize)
        self.head.weight = self.token.weight
        if self.copy_enabled and self.fixed_gate is None:
            nn.init.zeros_(self.copy_gate.weight)
            initial = config['copy_initial_gate']
            nn.init.constant_(self.copy_gate.bias, math.log(initial / (1 - initial)))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        if not 0 < ids.shape[1] <= self.context:
            raise ValueError('Input length must be between 1 and context.')
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """Return normalized log probabilities, also valid cross-entropy logits."""
        features = self.features(ids)
        vocab_logp = F.log_softmax(self.head(features).float(), dim=-1)
        if not self.copy_enabled:
            return vocab_logp
        q, k = self.copy_query(features).float(), self.copy_key(features).float()
        with torch.autocast(device_type=ids.device.type, enabled=False):
            scores = (q @ k.transpose(-2, -1)) * self.copy_scale
        length = ids.shape[1]
        future = torch.ones(length, length, dtype=torch.bool, device=ids.device).triu(1)
        attention = F.softmax(scores.masked_fill(future, float('-inf')), dim=-1)
        # Only observed tokens are copied; all temporary state is local to this call.
        indices = ids.unsqueeze(1).expand(-1, length, -1)
        probabilities = torch.zeros_like(vocab_logp).scatter_add(-1, indices, attention)
        copy_logp = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        copy_logp = copy_logp - torch.logsumexp(copy_logp, dim=-1, keepdim=True)
        if self.fixed_gate is None:
            gate = self.copy_gate(features).float()
            log_copy, log_vocab = F.logsigmoid(gate), F.logsigmoid(-gate)
        else:
            log_copy, log_vocab = math.log(self.fixed_gate), math.log1p(-self.fixed_gate)
        return torch.logaddexp(vocab_logp + log_vocab, copy_logp + log_copy)

    def predict_log_probs(self, ids):
        """Evaluation interface: finite, normalized natural-log probabilities."""
        return self(ids)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Final experimental checkpoints used a backbone wrapper; weights are identical in layout.
        renamed = {}
        for name, value in state_dict.items():
            key = name.removeprefix('backbone.')
            if key in renamed:
                raise ValueError(f'Duplicate checkpoint key after prefix conversion: {key}')
            renamed[key] = value
        return super().load_state_dict(renamed, strict=strict, assign=assign)


def build_model(config):
    """Build the final predictor; explicit checkpoint/config settings take precedence."""
    for key, value in MODEL_OPTIONS.items():
        if key != 'xsa_last_n_layers':
            config.setdefault(key, value)
    config.setdefault('xsa_last_n_layers', config['depth'])
    config.setdefault('ffn_hidden', 8 * ((config['width'] + 2) // 3))
    for key in ('width', 'heads', 'depth', 'context', 'vocab', 'ffn_hidden', 'copy_dim'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if config['width'] % config['heads'] or (config['width'] // config['heads']) % 2:
        raise ValueError('width must be divisible by heads, with an even head dimension.')
    if not config['use_rope'] or config['use_rmsnorm'] or not config['use_swiglu'] or config['use_smear']:
        raise ValueError('This final implementation requires RoPE, LayerNorm, SwiGLU and no SmearGate.')
    if config.get('use_gqa', False):
        raise ValueError('This final implementation uses MHA, not GQA.')
    if not config['use_partial_rope']:
        config['rope_fraction'] = 1.0
    if not 0 < config['rope_fraction'] <= 1:
        raise ValueError('rope_fraction must be in (0, 1].')
    if not math.isfinite(config['rope_theta']) or config['rope_theta'] <= 0:
        raise ValueError('rope_theta must be finite and positive.')
    if not 0 <= config['residual_dropout'] < 1:
        raise ValueError('residual_dropout must be in [0, 1).')
    count = config['xsa_last_n_layers']
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= config['depth']:
        raise ValueError('xsa_last_n_layers must be an integer between zero and depth.')
    if not 0 < config['copy_initial_gate'] < 1:
        raise ValueError('copy_initial_gate must be in (0, 1).')
    if config['copy_fixed_gate'] is not None and not 0 < config['copy_fixed_gate'] < 1:
        raise ValueError('copy_fixed_gate must be in (0, 1).')
    return GPT(config)
