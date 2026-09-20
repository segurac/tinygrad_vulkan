# Adreno (vk5143) 256 MiB per-buffer limit

## Symptom
On Snapdragon 870 / Adreno 710 (`VK_ARCH=vk5143`), a conv over a large batch yields
**uniformly wrong** values, while the *identical* SPIR-V is correct on NVIDIA. Size
dependent: correct up to a threshold, wrong above it.

## Root cause (confirmed)
The Adreno Vulkan driver enforces a **2²⁸-byte (256 MiB) per-storage-buffer limit**. When a
single storage buffer exceeds 256 MiB, byte offsets ≥ 2²⁸ wrap around and scramble the
buffer. The conv1 output (`N × 32 × 24 × 24` float32 = `N × 72 KiB`) crosses the limit at
`N = 3641` (`3641 × 73728 = 268,443,648 > 2²⁸ = 268,435,456`).

## Evidence
Boundary test: `conv1` on `ones(N,1,28,28)`, identical pre-compiled SPIR-V run on NVIDIA
(correct) and Adreno, fixed weights.

| N    | conv1 output   | vs 2²⁸ (268,435,456 B) | NVIDIA                  | Adreno                  |
|------|----------------|------------------------|-------------------------|-------------------------|
| 2048 | 150,994,944 B  | under                  | works                   | works ✅                |
| 3640 | 268,369,920 B  | under                  | sum=9539913, −0.2904    | sum=9539913, −0.2904 ✅ |
| 3641 | 268,443,648 B  | **over**               | sum=9542632, −0.2904    | sum=9542286, **0.6759** ❌ |
| 3642 | 268,517,376 B  | over                   | sum=9545188, −0.2904    | sum=9542270, **0.6759** ❌ |
| 4096 | 301,989,888 B  | over                   | correct                 | wrong ❌                |

The boundary is exactly 2²⁸: `N × 73728 = 268435456 → N = 3640.89`, so N=3640 is the last
correct batch and N=3641 the first wrong one.

Why "uniformly wrong" and not a partial write: offsets in `[2²⁸, size)` wrap to
`[0, size−2²⁸)`, so the tail is written on top of the head — the whole buffer is scrambled
(hence head = 0.6759, not the correct −0.2904), not truncated.

## Scope
- Device: Adreno reported as `vk5143`. Likely shared by other Adreno GPUs with 28-bit
  offset handling. NVIDIA has no such limit.
- Trigger: **any** single storage buffer > 256 MiB (not just conv). conv1's output is the
  largest buffer in the MNIST model, so it trips the limit first as batch grows.

## Workaround (mitigate on the client; do not "fix" in codegen)
Keep every buffer ≤ 256 MiB: `N × C × H × W × 4 ≤ 268435456`. For the conv1 output
(`C=32, H=W=24`): **N ≤ 3640** per chunk.

`examples/beautiful_mnist.py`:
- **Training** (batch 512): conv1 output = 36 MiB — fine as-is.
- **Eval** (`get_test_acc`, full 10k): conv1 output = 703 MiB — **must chunk** into ≤ 3640
  rows (we use 2048). Per-chunk accuracy is exact (10k chunked eval = 8.21% = CPU).

It is a driver bug (valid SPIR-V, correct on NVIDIA); the real fix is upstream in the
driver, chunking is the correct client-side mitigation.

## Repro
Dump the exact kernel on NVIDIA and print the correct reference, then run the identical
pre-compiled SPIR-V on Adreno.

```bash
# NVIDIA (dev box): dump + reference
N=3641 VK_ARCH=vk5143 VKDUMP=1 TINYGRAD_SPV_DIR=spv_c1_3641 FWDJSON=mnist_fwd.txt \
  DEV=VULKAN VK_VENDOR=10de python repro.py
# Adreno (tablet): run the identical pre-compiled kernel
N=3641 REPO=$REPO FWDJSON=$H/mnist_fwd.txt TINYGRAD_SPV_DIR=$H/spv_c1_3641 \
  TINYGRAD_SPV_STRICT=1 VK_ARCH=vk5143 DEV=VULKAN VK_VENDOR=5143 \
  VULKAN_LOADER=/system/lib64/libvulkan.so $PY repro.py
```

`repro.py`:
```python
import os, json
from examples.beautiful_mnist import Model
from tinygrad import Tensor
from tinygrad.nn import state
N = int(os.environ["N"])
d = json.load(open(os.environ["FWDJSON"]))
m = Model(); sd = state.get_state_dict(m)
for k, (s, v) in zip(sd, d["params"]): sd[k] = Tensor(v).reshape(s)
state.load_state_dict(m, sd)
y = m.layers[0](Tensor.ones(N, 1, 28, 28)).realize()
print("sum=%.4f head=%s" % (y.sum().item(),
      [round(t, 4) for t in y.reshape(-1)[:4].tolist()]))
```
