"""Decoder-only GPT-style Transformer, matching report/phase2_model_report.md §1:
tied input/output embedding, pre-norm residual blocks, RoPE applied to Q/K inside
attention (zero extra parameters), d_ff = 4 * d_model, biases on every Linear/LayerNorm.

Default GPTConfig field values are exactly the chosen configuration (vocab=16,000,
d_model=384, n_layers=11, n_heads=6, d_ff=1536, max_seq_len=512) — instantiating
GPT(GPTConfig()) reproduces the reported 25,663,872-parameter model.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 16_000
    d_model: int = 384
    n_layers: int = 11
    n_heads: int = 6
    d_ff: int = 1536
    max_seq_len: int = 512
    dropout: float = 0.1
    rope_base: float = 10_000.0


def precompute_rope(head_dim: int, max_seq_len: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim / 2)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim / 2)
    cos = torch.cat([cos, cos], dim=-1)[None, None, :, :]
    sin = torch.cat([sin, sin], dim=-1)[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.wq = nn.Linear(cfg.d_model, cfg.d_model)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout_p = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                return_attn: bool = False):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_weights = None
        if return_attn:
            # Explicit softmax(QK^T/sqrt(d_k) + M)V — only for attention analysis
            # (heatmaps/entropy/mean-distance), which needs the weights themselves.
            # Training/inference otherwise always use the fused SDPA path below,
            # which never materializes this matrix.
            scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            y = attn_weights @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_dropout(self.wo(y)), attn_weights


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                return_attn: bool = False):
        attn_out, attn_weights = self.attn(self.ln1(x), cos, sin, return_attn=return_attn)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = precompute_rope(head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                return_attn: bool = False):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, \
            f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)
        attn_weights_by_layer = []
        for block in self.blocks:
            x, attn_weights = block(x, cos, sin, return_attn=return_attn)
            if return_attn:
                attn_weights_by_layer.append(attn_weights)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_attn:
            return logits, loss, attn_weights_by_layer
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, greedy: bool = False) -> torch.Tensor:
        """Autoregressive sampling. `idx`: (B, T) prompt token ids. `greedy=True` ignores
        `temperature`/`top_k` and always takes the argmax (used for the assignment's
        greedy-decoding condition; temperature sampling is used for 0.5/1.0/1.5)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if greedy:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
