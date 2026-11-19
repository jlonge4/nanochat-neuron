"""
Profile nanochat's RoPE on Trainium to show ATen lowering (eager vs torch.compile).

  eager    nanochat.gpt.apply_rotary_emb as-is: every ATen op (slice, mul, neg, add, cat)
           dispatches and runs on the device separately
  compile  the same function under torch.compile(backend="neuron", fullgraph=True):
           one compiled region, one NEFF

Shapes match one nanochat d24 attention layer at training length: q and k are
(B=1, T=2048, H=12, D=128) bf16, cos and sin are (1, T, 1, D/2) as gpt.py builds them.

Usage (from the repo root, on a Trainium host):
  python -m dev.profile_rope check
  TORCH_NEURONX_NEFF_CACHE_DIR=$(mktemp -d) python -m dev.profile_rope eager   /tmp/prof/eager
  TORCH_NEURONX_NEFF_CACHE_DIR=$(mktemp -d) python -m dev.profile_rope compile /tmp/prof/compile

Use an empty NEFF cache for each run. The device profile copies NEFFs from that cache into
<outdir>/neffs, and Neuron Explorer needs each .neff next to its .ntff. Open <outdir> in
Neuron Explorer for the system trace (framework, runtime, and hardware lanes) and the device view.
"""
import sys

import torch
import torch_neuronx  # noqa: F401
from torch.profiler import ProfilerActivity
from torch_neuronx.profiling import NeuronConfig, NeuronProfiler, ProfileMode

from nanochat.gpt import apply_rotary_emb  # nanochat's real implementation

B, T, H, D = 1, 2048, 12, 128
torch.manual_seed(0)
dev = "neuron"

q = torch.randn(B, T, H, D, dtype=torch.bfloat16).to(dev)
k = torch.randn(B, T, H, D, dtype=torch.bfloat16).to(dev)
inv_freq = 1.0 / (100000 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))  # base=100000, as in gpt.py
freqs = torch.outer(torch.arange(T, dtype=torch.float32), inv_freq)
cos = freqs.cos().bfloat16()[None, :, None, :].to(dev)
sin = freqs.sin().bfloat16()[None, :, None, :].to(dev)


def rope_eager(q, k):
    return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)


rope_compiled = torch.compile(rope_eager, backend="neuron", fullgraph=True)

VARIANTS = {"eager": rope_eager, "compile": rope_compiled}


def check():
    ref_q, ref_k = (t.float().cpu() for t in rope_eager(q, k))
    for name, fn in VARIANTS.items():
        oq, ok = (t.float().cpu() for t in fn(q, k))
        print(f"{name:14s} max|dq|={(oq - ref_q).abs().max().item():.4g} max|dk|={(ok - ref_k).abs().max().item():.4g}")


def profile(variant, outdir):
    fn = VARIANTS[variant]
    nc = NeuronConfig(
        modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
        profile_output_dir=outdir,
        capture_enabled_for_nc="0",
        correlate_device_time=True,
    )
    exporter = NeuronProfiler(nc)
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        experimental_config=nc,
        record_shapes=True,
        on_trace_ready=exporter.export_trace,
        schedule=torch.profiler.schedule(wait=2, warmup=1, active=1),
    ) as prof:
        for _ in range(4):
            with torch.no_grad():
                fn(q, k)
            torch.neuron.synchronize()
            prof.step()
    print(prof.key_averages().table(sort_by="self_privateuse1_time_total", row_limit=15))


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        profile(sys.argv[1], sys.argv[2])
