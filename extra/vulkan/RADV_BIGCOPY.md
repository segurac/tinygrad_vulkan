# RADV gfx9 APU: large device-buffer bugs (2^32 compute reads + big H2D copies)

Two distinct silent-corruption bugs hit multi-GiB device buffers on Mesa **RADV** on the
Renoir-class APU (`0x1002:0x1638`, GC IP **gfx90c**, unified memory). Both are invisible
(no `VkResult`, no page fault, no log) and both broke GGUF model loading: the embedding
and LM-head read back **zero** past a size-dependent offset, so chat produced immediate
EOS / garbage. Reproduced on Mesa **26.1.6** (`26.1.6-1~bpo13+1`, driverVersion
`0x6801006`) and **26.2.3** (sid, driverVersion `0x6802003`). NVIDIA (RTX 3060) is
unaffected.

The full source-traced upstream report (with a self-contained raw-Vulkan repro) is at
`/tmp/opencode/RADV_GFX9_BIGBUF_REPORT.md` — file it at gitlab.freedesktop.org/mesa/mesa.

## Bug 1 (the big one): compute reads of a buffer whose GPU VA span crosses 2^32 are a silent no-op

### Symptom
A compute kernel that reads a device buffer returns **zero / leaves the output unwritten**
for the part of the buffer past the 2^32 (4 GiB) boundary of the buffer's GPU VA. Sizes that
fit in one 4 GiB VA block are correct; the moment the span crosses, the tail is wrong:

```
single device buffer, fresh process (deviceAddress base 0xffff800100200000):
  2.0 GiB  OK
  3.5 GiB  OK   (span ends 0xffff8001E0200000, upper-32 stays 0xffff8001)
  3.9 GiB  OK   (span ends below 0xffff800200000000)
  4.0 GiB  BAD  (span ends 0xffff800200200000 -> crosses the 2^32 boundary; whole dispatch a no-op)
```

The whole-file GGUF case is the same boundary at a different base: the 20.6 GiB file buffer's
low-32 base is `0x58C0_0000` and reads are correct only while `base + offset < 0x8000_0000`,
i.e. for a correct window of `0x8000_0000 - 0x58C0_0000 = 0x2740_0000` = **0.6144 GiB** —
exactly where the Qwen embedding zeroed out.

The asymmetry that pinpoints it: the **copy engine (SDMA / MM hub) reads and writes the same
VA range correctly at all sizes** (a 20.6 GiB D2H readback is byte-perfect), while only the
**GFX (compute) path** breaks. So the page tables are valid and fully populated — the failure
is in 32-bit compute addressing, not in VMM.

### Root cause (traced in mesa 26.2.3 source)
RADV addresses compute buffer memory as a **32-bit offset within a single 4 GiB VA window**:
the buffer's upper 32 VA bits are stored once in the buffer descriptor and all in-buffer
offsets are 32-bit. A buffer whose VA span crosses 2^32 therefore has an inconsistent
descriptor for the part past the boundary, and those accesses are a silent no-op.

- `src/amd/vulkan/radv_nir_lower_descriptors.c:69` — `nir_pack_64_2x32_split(ptr, address32_hi)`
- `src/amd/vulkan/radv_nir_lower_descriptors.c:177` —
  `ac_build_raw_buffer_descriptor(..., address32_hi << 32, 0xffffffff, ...)`
- `src/aco/ac_cmdbuf.h:377` — asserts `((va) >> 32) == address32_hi` (the invariant a
  crossing span violates)
- `address32_hi` is KMD-supplied (`src/amd/ac/ac_gpu_info.c:1522`).

This matches the descriptor model in Maister's "DXIL to SPIR-V, part 4" §4: *"Descriptor
memory lives in a 4 GB virtual memory range, so RADV only needs to push a 32-bit pointer to
store a descriptor set and the upper half is synthesized from a constant."*

The VMM allocates the crossing span **silently** (verified via `RADV_DEBUG=bo_history`; the
4 GiB BO is mapped `0x...8001_00200000 .. 0x...8002_00200000`), and `RADV_DEBUG=validatevas`
does **not** catch it (the whole bound range is marked valid, so the shader instrumentation
passes). `RADV_MAX_MEMORY_ALLOCATION_SIZE = 0xFFFFFFFC` (`radv_constants.h:80`) is the
advertised per-allocation cap, but `radv_create_buffer` only enforces it under
`#if DETECT_OS_ANDROID` — on Linux a 4.0 GiB (= 0x100000000, over the cap) allocation is
**accepted** and then breaks in compute.

There is **no** RADV knob to control VA placement (checked the 26.2.3 `RADV_DEBUG` option
table; `32bitva` / `aco` do not exist). So the only app-level mitigation is to never let a
single device buffer's VA span cross 2^32.

### Workaround (landed, core)
`tinygrad/llm/gguf.py::_gguf_parse` no longer materializes the whole GGUF file into one
device buffer. It keeps the file on `DISK` (mmap) and **stages each tensor to the device
individually** (`tensor[off:off+nb].to(None).realize()`), so every dequant reads from its own
small buffer. Per-tensor sizes are far below 2 GiB (Qwen3.6: 733 tensors, max 0.503 GiB;
Moonlight: 430 tensors, max 0.256 GiB), so no buffer's span can cross 2^32. Verified: Qwen3.6
(22.1 GiB) loads in ~26 s on the APU with the embedding fully nonzero; Moonlight (9.82 GiB)
loads clean; both correct on the 3060. `test/runtime/test_gguf.py` (67) and
`test/null/test_gguf.py` (17) pass.

General rule for any big-buffer workload on RADV gfx9 APU: **keep every device buffer under
~2 GiB** (comfortably inside one 4 GiB VA block regardless of the base RADV hands out).

## Bug 2: big single H2D `vkCmdCopyBuffer` left the destination tail zero

### Symptom
A single multi-GiB host→device `vkCmdCopyBuffer` left the destination's first N bytes
correct and the rest **zero**, no error reported (D2H unaffected). Observed boundary was
non-deterministic (~0.4–2 GiB; a 0.389 GiB tensor landed, a 1.02 GiB tensor stopped at
0.406 GiB).

### Workaround (landed, branch-owned)
`tinygrad/runtime/support/vulkan_rt.py::VkRt.cmd_copy` splits every copy into
`COPY_CHUNK = 256 MiB` `vkCmdCopyBuffer` calls, staying below the smallest observed
boundary. Chunked H2D of the 20.6 GiB file verifies byte-perfect at 16/16 offsets.

Note: with the Bug-1 fix, GGUF loads no longer issue a whole-file H2D (each tensor is copied
individually, ≤ 0.503 GiB, still chunked to 256 MiB here). The chunking is kept as a
defensive guard for any other large H2D copy; its exact relationship to the 2^32 issue is not
fully separated out, but it is cheap and was required for correctness before the staging fix.

## Scope
- Device/driver: RADV on a gfx90c APU (one Renoir-class APU confirmed). Both Mesa 26.1.6 and
  26.2.3. NVIDIA unaffected. Other RADV versions / AMD dGPUs unknown.
- Workloads: anything that puts a >~2 GiB buffer on the APU and reads it in a compute shader
  (large GGUF files, big activations). Small-buffer workloads are unaffected.

## Repro
Self-contained raw-Vulkan (no tinygrad) repro of Bug 1 — allocate an N GiB buffer, H2D a
pattern, dispatch a `xor 1` compute kernel, D2H, compare:

```bash
# APU, Mesa 26.1.6 RADV:
PATH=$HOME/.bin:$PATH CCACHE=0 DEV=VULKAN VK_VENDOR=1002 VK_DEVICE_INDEX=0 python vk_bigbuf_repro_final.py
#   2.0/3.5/3.9 GiB -> OK ;  4.0 GiB -> BAD (dispatch no-op, deviceAddress base 0xffff800100200000)
```

Python (tinygrad) repro of the model-load symptom: load a >~2 GiB GGUF on the APU and check
the embedding past the old truncation row — before the fix it is all zero, after it is
nonzero (see `verify_core_fix.py`).
