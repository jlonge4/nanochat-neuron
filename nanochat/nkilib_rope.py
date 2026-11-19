"""
NKIlib rope_hf as a PyTorch op for nanochat (eager, torch.compile, and autograd).

nanochat layout and convention (nanochat/gpt.py apply_rotary_emb):
  x is (B, T, H, D); cos, sin are (1, T, 1, D/2); halves are rotated by -theta:
    y1 = x1*cos + x2*sin,  y2 = -x1*sin + x2*cos
rope_hf (NKIlib, HF layout) computes x*cos_full + rotate_half(x)*sin_full on (B, H, T, D),
with rotate_half(x) = cat(-x2, x1). Passing sin_full = cat(-sin, -sin) gives nanochat's result.
rope_hf needs T % (128 * LNC) == 0, so callers must fall back for decode (T == 1).
"""
import torch
from torch import Tensor
from torch_neuronx import nki_op

from nkilib.core.embeddings.rope_hf import rope_hf  # pip install git+https://github.com/aws-neuron/nki-library


def _rope_hf_fake(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, backward: bool) -> tuple[Tensor, Tensor]:
    return torch.empty_like(q), torch.empty_like(k)


@nki_op("nanochat::rope_hf", mutates_args={}, fake_fn=_rope_hf_fake)
def rope_hf_op(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, backward: bool) -> tuple[Tensor, Tensor]:
    """q, k: (B, H, T, D) contiguous. cos, sin: (T, D). Returns rotated (q, k)."""
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    rope_hf(q, k, q_out, k_out, cos, sin, backward=backward)
    return q_out, k_out


def _setup_context(ctx, inputs, output):
    _, _, cos, sin, _ = inputs
    ctx.save_for_backward(cos, sin)


def _backward(ctx, grad_q, grad_k):
    cos, sin = ctx.saved_tensors
    dq, dk = rope_hf_op(grad_q.contiguous(), grad_k.contiguous(), cos, sin, True)
    return dq, dk, None, None, None


torch.library.register_autograd("nanochat::rope_hf", _backward, setup_context=_setup_context)


def hf_cos_sin(cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """nanochat (1, T, 1, D/2) cos/sin -> rope_hf (T, D) cos/sin, with sin negated for -theta."""
    c, s = cos[0, :, 0, :], sin[0, :, 0, :]
    return torch.cat([c, c], dim=-1).contiguous(), torch.cat([-s, -s], dim=-1).contiguous()


def nkilib_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Drop-in for (apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)) on nanochat layouts."""
    c, s = hf_cos_sin(cos, sin)
    qo, ko = rope_hf_op(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), c, s, False)
    return qo.transpose(1, 2), ko.transpose(1, 2)
