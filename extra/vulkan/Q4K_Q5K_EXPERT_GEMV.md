# Fused Q4_K/Q5_K MoE expert GEMV kernels (Vulkan, APU decode)

Fused dequant+GEMV custom kernels for the Vulkan backend, targeting LLM **decode**
(batch*tokens == 1) of GGUF MoE models (e.g. Qwen3.6-35B-A3B) on AMD APUs
(RADV gfx9, shared DDR). The kernels read the **quantized weight bytes
directly** from the GGUF staging buffers — zero copies, zero extra memory —
instead of the baseline scheduler's fused gather+dequant+GEMV codegen.

- `tinygrad/llm/kernels/vulkan.py` — the kernels + weight detection + dispatch wrappers.
- `tinygrad/llm/kernels/amd.py` — `Linear.__call__` gains a small additive Vulkan gate
  (dense Q4_K path).
- `tinygrad/llm/model.py` — `ExpertWeights.__call__` gains the decode gate for the
  routed experts (Q4_K gate/up + Q5_K down).

Everything is gated: non-Vulkan devices, prefill (B*T > 1), unsupported quant types
and failed detection all fall through to the pre-existing code path unchanged.
On the APU the Q5_K down-expert kernel is **4x faster per call** (2019 us -> 531 us
median) and gives **+11-12% net decode** (1.75 -> 1.95 tok/s on Qwen3.6-35B-A3B,
JIT=1). Generation A/B (temp 0) produces bit-identical token sequences.

## Toggling / running

```
# APU
PATH=<llvm-bin>:$PATH CCACHE=0 DEV=VULKAN VK_VENDOR=1002 VK_DEVICE_INDEX=0 python ...
# 3060 (Vulkan via NVIDIA ICD): VK_VENDOR=10de

JIT=1                        # required for usable decode speed on this branch
VULKAN_Q4K=0                 # disable dense + gate/up expert kernels (default ON)
VULKAN_Q5K=0                 # disable down-expert kernel (default ON)
VULKAN_QUANT=0               # legacy master switch for the Q4K side only
```

Verify: `python -m ruff check tinygrad/llm/ && python -m mypy tinygrad/llm/`.

## How it works

### Weight detection (zero-copy)
`_find_quant_bytes(weight, ggml_type, block_size, block_bytes)` finds the flat uint8
buffer a dequantized weight reads from: locate the uint8 SHRINK/BUFFER node in
`weight.uop.toposort()` whose byte count is `numel // 256 * block_bytes`, then verify
that the weight's dequant expression is *exactly* `ggml_data_to_tensor(raw, n,
ggml_type)` (UOp-key equality after stripping RESHAPE/STAGE/float-cast views).
Returns a flat alias of the on-device staging buffer — so the quantized bytes are
already resident from load time (see `RADV_BIGCOPY.md` for the per-tensor staging
that made this possible on gfx9). Detection is numel-based, so the same finder works
for dense `(out, in)` and 3-D `(n_experts, out, in)` weights. Q6_K/Q8_0/f32 weights
fail the size+key check and are left on the baseline path.

### Kernel shape (proven structure on this renderer)
One work item per output row (dense) or per (expert-slot, row) pair (MoE gather).
Inside: LOCAL axis over the 32-element groups, REDUCE axis over the 32 elements of a
group (one weight byte each):

```
o = GLOBAL(out)                     # MoE: GLOBAL(k*out); e, row = o//out, o%out
                                    # MoE: expert = sel[0,0,e]  (per-thread base offset)
g = LOCAL(in//32);  b = REDUCE(32)
block, sg = g//8, g%8
bbase = expert*out_row_bytes + row*row_bytes + block*144      # (Q4_K: 144 B/block)
d, dmin = half LE from B(0..3)
sc, mn  = scale/min per ggml layout, via where(sg<4, ...)     # LOCAL-axis where: OK
q       = group-parity nibble of B(16 + (sg//2)*32 + b)       # Q4_K
        = nibble(B(48+(sg//2)*32+b)) + 16*bit(sg) of B(16+b)  # Q5_K (qh top bit)
term = (d*sc*q - dmin*mn) * x[...]
total = term.reduce(b, ADD).reduce(g, ADD)
out[...].store(total).end(o).sink(arg=KernelInfo(name=..., opts_to_apply=()))
```

Key points:
- `expert = sel[0, 0, e]` reads the top-k index once per work item; it is constant
  inside the reduce loops, so the variable byte-base is fine (no where on the REDUCE
  axis).
- The down-expert path receives x as `(B, T, k, in)` — **per-expert rows** (expert
  slot e uses x row e) — while gate/up get one shared `(B, T, in)` vector. The decode
  gates in `model.py` differ accordingly: Q4_K `prod(x.shape[:-1]) == 1`, Q5_K
  `prod(x.shape[:-2]) == 1`.
- `opts_to_apply=()` is mandatory (see renderer constraints below).

### ggml byte layouts used here
- **Q4_K** (ggml_type 12, 144 B / 256 elems): `d` half LE @0-1, `dmin` half LE @2-3,
  12 scale bytes @4-15, 128 weight bytes @16-143. Group `sg` (0..7), element `i`
  (0..31): weight byte = `16 + (sg//2)*32 + i`; **low** nibble if sg even, **high** if
  sg odd (nibble picked by *group* parity, not element parity — this was wrong in our
  first attempt and only accidentally correct for sg=0).
  sc = sg<4 ? s[sg]&63 : (s[8+sg-4]&0xF)|((s[sg-4]>>6)<<4);
  mn = sg<4 ? s[4+sg]&63 : (s[8+sg-4]>>4)|((s[4+sg-4]>>6)<<4);
  value = d*sc*q - dmin*mn.
- **Q5_K** (ggml_type 13, 176 B / 256 elems): d/dmin @0-3, scales @4-15, qh 32 B
  @16-47 (top bit of each weight), qs 128 B @48-175 (5th nibble, same group-parity
  layout as Q4_K). q = qs_nibble + 16 * bit(sg) of qh[i//... per gguf.py]. The
  `ggml_data_to_tensor` branch in `tinygrad/llm/gguf.py` is the ground truth — the
  kernels were verified **bit-exact** (max abs err 0.0) against it on synthetic data.

## Vulkan/NIR renderer constraints (empirical, do not re-litigate)

Hard-won limits of the UOp custom-kernel path on this backend (RADV gfx9):

1. **`opts_to_apply=()` in `KernelInfo` is required.** The opt heuristic's
   `SPLIT(16, LOCAL)` on a reduce axis produces a grouped reduce that **computes
   wrong** on this backend (measured err ~7000 vs 0.0 with the kernel otherwise
   identical).
2. **Only `.end()` the GLOBAL axis.** `.reduce(axis)` already closes its range
   (`reduce_ranges_to_acc`, `tinygrad/codegen/__init__.py`); an explicit `.end()` on a
   reduced axis creates a double END and trips
   `assert y.src[1] not in x.backward_slice_with_self` in
   `tinygrad/codegen/late/linearizer.py`.
3. **No `.where()` whose condition varies over a REDUCE axis** (asserts in the NIR
   rewrite). Conditions on GLOBAL/LOCAL axes are fine (used for `sg<4`, group parity).
4. **Only constant shift amounts inside REDUCE loops.** A shift by a REDUCE-axis value
   (e.g. `>> ((i%2)*4)`) is rejected. Restructure so the shift depends on a LOCAL axis
   value that is loop-invariant (the Q5_K `qh` top-bit shift by `sg` works this way).
5. `has_local=True`, `has_shared=False` (nir.py) — workgroup-local scalars/arrays OK,
   no shared-memory staging buffer.
6. The `custom_gemm` accumulator pattern (`.set(0.0)` / `.after(k)` on a REDUCE-tagged
   range, `test/null/test_custom_kernel.py`) raises a `KeyError` in the NIR rewriter.
   Use the `.reduce(axis, arg=Ops.ADD)` chain (or a REG placeholder) instead.
7. **In-cb visibility bug (RADV)**: a kernel reading data written by an *earlier
   dependent* kernel in the same command buffer sees stale data even with a barrier
   (retested broken on mesa 26.1.6). That is why `VK_PER_KERNEL=1` (per-kernel cb +
   submit + fence) is the AMD default in `vulkan_rt.py`. **Independent** kernels in one
   cb are safe (verified 32/32), so level-batched submits are possible — expected
   +3-10% (see "Remaining work").
8. Byte loads, masks, `//`, `%`, `.bitcast` to half/f32, `.eq()`, and in-store casts
   all work. `uop == const` returns a Python bool (use `.eq()`).

## What we measured (Qwen3.6-35B-A3B, APU, JIT=1, 40 tokens)

Model facts that shaped this work: the 80 **Q4_K** tensors are the routed **gate/up**
experts (256 experts, 512x2048); 37 **Q5_K** = the routed **down** experts
(256x2048x512); 3 Q6_K down experts (blk 34/38/39) + Q6_K LM head; dense attn/SSM/
shared Linears are Q8_0; router/norms f32. The tinygrad scheduler **already**
auto-fuses gather+dequant+GEMV for these (it streams the quant bytes — there is no
f16 weight materialization to eliminate), so the win here is per-kernel efficiency,
not traffic.

| VULKAN_Q4K, VULKAN_Q5K | tok/s | kernels/step |
|---|---|---|
| 0, 0 (baseline) | 1.75 | 1210 |
| 1, 0 | 1.72 | 1290 |
| 0, 1 | 1.96 | 1290 |
| 1, 1 (defaults) | 1.95 | 1290 |

- Isolated per call: baseline down `r_32_8_16_4_2_2_2_2_32` 2019 us ->
  `q5k_moe_gemv_2048_512_256` 531 us; baseline gate/up 344 us -> 224 us.
- The Q4_K gate/up path is net ~0 in the real flow: its per-call win is cancelled by
  scheduler side effects (the custom-kernel boundary makes the scheduler split two
  elementwise chains out and re-codegen the down GEMV to a slower variant, +80
  kernels/step). Kept on by default (harmless, [1,1] ~= [0,1], preserves A/B ability);
  set `VULKAN_Q4K=0` for the marginally fastest config.
- Reference: llama.cpp/Ollama on the same APU: 20.5 tok/s gen, 32 tok/s prefill
  (~54 GB/s effective = the DDR roofline).

## Remaining work (ranked)

1. **Critical-path fusion**: decode is a 658-deep serial chain (1290 kernels, only
   1-3 independent per level; the 8 experts are already fused by the scheduler).
   Shortening the chain (llama.cpp-style: residual-add fused into GEMVs, add_rms,
   gate/up+activation fusion) is the biggest remaining lever.
2. **Q6_K kernels** (3 down experts ~6 ms/step + the 418 MB/step LM head).
3. **Q5_K kernel efficiency**: 531 us for 5.7 MB = ~11 GB/s, latency-bound at the
   400 MHz DPM state; at peak bandwidth this should be ~150 us (vectorized loads /
   occupancy).
4. **Independent-kernel cb batching** (safe per constraint 7): 1210 -> 658 submits,
   expected +3-10%.
5. **Upstream**: file the RADV in-cb visibility bug (blocks full submit batching;
   with it fixed the roofline ~20 tok/s becomes reachable). `RADV_BIGCOPY.md` has
   the related gfx9 big-buffer bug write-up; `RADV_CONST128.md` another.
