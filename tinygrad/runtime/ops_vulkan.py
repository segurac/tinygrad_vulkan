from __future__ import annotations
import struct, os, time, ctypes
from typing import Any, cast
from tinygrad.device import Compiled, Allocator, BufferStorage, BufferSpec, MMIOInterface, Program, TinyELF
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import getenv, mv_address
from tinygrad.renderer.nir import SPIRVRenderer
from tinygrad.runtime.support import vulkan_rt as vkrt

# per-arch single storage-buffer size limit, bytes (absent/None = unlimited). The Adreno
# vk5143 Vulkan driver mis-addresses buffers > 2^28 bytes and returns silently-wrong values
# instead of erroring (see extra/vulkan/ADRENO_256MB.md). Extend this table as more affected
# GPUs are found; VK_MAX_BUFFER overrides it (0 = unlimited).
_VULKAN_MAX_BUFFER = {"vk5143": 2**28}

class VulkanBuffer:
  # a view into a VkBuffer (device-local data, host-visible params/staging);
  # the descriptor addresses (handle, offset)
  __slots__ = ("vbuf", "offset")
  def __init__(self, vbuf, offset:int=0): self.vbuf, self.offset = vbuf, offset
  @property
  def handle(self): return self.vbuf.handle

def _desc(buf:VulkanBuffer):
  # (vbuf, offset, range): range extends to the end of the physical allocation so gated-load
  # over-reads stay in-bounds; RADV's robustBufferAccess turns any true OOB read into a 0
  return (buf.vbuf, buf.offset, buf.vbuf.size - buf.offset)

def _pack_params(vals:tuple[int|float, ...], var_dts:tuple[DType, ...]) -> bytes:
  # one 8-byte slot per scalar param, in ascending-slot order (== vals order).
  # SPIRVRenderer loads each slot as a u64 then f2f<bitsize>/i2i<u2u: floats must therefore
  # be stored as their float64 bit-pattern, ints as their native little-endian value.
  u8 = bytearray(8 * len(vals))
  for i, (v, dt) in enumerate(zip(vals, var_dts)):
    off = i * 8
    if dt in dtypes.floats: struct.pack_into('<d', u8, off, float(v))
    elif dt in dtypes.ints: struct.pack_into(f'<{dt.fmt}', u8, off, v)
    else: raise RuntimeError(f"unsupported VULKAN param dtype {dt}")
  return bytes(u8)

def _copy_to_arr(dst_arr, src_mv:memoryview, n:int):
  # C-level memcpy from a host memoryview (read-only OK, e.g. an mmap'd weights file) into a
  # mapped host-visible buffer. A ctypes slice assignment (dst[0:n]=src) is ~1000x slower.
  try:
    ctypes.memmove(dst_arr, (ctypes.c_uint8 * n).from_buffer(src_mv), n)
  except (BufferError, TypeError):
    ctypes.memmove(dst_arr, bytes(src_mv), n)

def _copy_to_mv(dst_mv:memoryview, src_arr, n:int):
  # C-level memcpy from a mapped host-visible buffer into a host memoryview (writable).
  ctypes.memmove((ctypes.c_uint8 * n).from_buffer(dst_mv), src_arr, n)

def _warmup_vendors(rt) -> bool:
  # NV 550 drops ~44% of a cold pipeline's first-dispatch stores; dispatch it once (result
  # discarded) so the real first launch is actually the second. VK_WARMUP: off / all /
  # comma-sep vendor ids; default is NV only.
  mode = os.environ.get("VK_WARMUP", "")
  if mode == "off": return False
  if mode == "all": return True
  if mode: return rt.vendor in {int(v, 0) for v in mode.split(",") if v.strip()}
  return rt.vendor == 0x10de

def _arm_unit_diff(i, ra, rb) -> str:
  # one line per differing unit of two consecutive steps: which program/bufs/grid/vals changed
  if ra[0] != rb[0]:
    return f"[{i}] kind A={ra[0]} B={rb[0]}: A={ra!r} B={rb!r}"
  if ra[0] == "K":
    parts = [f"[{i}] K prog#{ra[1]} {ra[2]}"]
    if (ra[1], ra[2], ra[3]) != (rb[1], rb[2], rb[3]):
      parts.append(f"program A=#{ra[1]}/{ra[2]} B=#{rb[1]}/{rb[2]}")
    if ra[4] != rb[4]:
      ch = [f"buf{j}: A={ra[4][j] if j < len(ra[4]) else '-'} B={rb[4][j] if j < len(rb[4]) else '-'}"
            for j in range(max(len(ra[4]), len(rb[4])))
            if j >= len(ra[4]) or j >= len(rb[4]) or ra[4][j] != rb[4][j]]
      parts.append("bufs " + "; ".join(ch))
    if ra[5] != rb[5]:
      parts.append(f"grid A={ra[5]} B={rb[5]}")
    if ra[7] != rb[7]:
      ch = [f"{ra[6][k] if k < len(ra[6]) else k}: {ra[7][k]!r}->{rb[7][k]!r}"
            for k in range(max(len(ra[7]), len(rb[7])))
            if k >= len(ra[7]) or k >= len(rb[7]) or ra[7][k] != rb[7][k]]
      parts.append("vals " + "; ".join(ch))
    return " | ".join(parts)
  return f"[{i}] {ra[0]}: A={ra[1:]} B={rb[1:]}"

def _arm_report(dev) -> str:
  # VK_ARMDIFF report: diff the last two recorded kernel steps (steady decode) and list every
  # per-step-changing input; also flag any config launched >1x in one step with different vals
  # (a UBO race hazard for the resubmit path) and the H2D/D2H buffer identities.
  a, b = dev._arm_steps[-2], dev._arm_steps[-1]
  out = [f"VK_ARMDIFF: {len(dev._arm_steps)} kernel steps recorded; diffing last two (steady decode)",
         f"step A: {len(a)} units | step B: {len(b)} units"]
  for tag, s in (("A", a), ("B", b)):
    nk = sum(1 for r in s if r[0] == "K")
    nh = sum(1 for r in s if r[0] == "H2D")
    nd = sum(1 for r in s if r[0] == "D2H")
    out.append(f"  {tag}: {nk} kernels, {nh} H2D, {nd} D2H")
  if len(a) == len(b):
    diffs = [(i, ra, rb) for i, (ra, rb) in enumerate(zip(a, b)) if ra != rb]
    out.append(f"{len(diffs)}/{len(a)} units differ between consecutive steps:")
    for i, ra, rb in diffs: out.append("  " + _arm_unit_diff(i, ra, rb))
  else:
    out.append(f"UNIT COUNT MISMATCH ({len(a)} vs {len(b)}):")
    for i in range(max(len(a), len(b))):
      ra = a[i] if i < len(a) else None
      rb = b[i] if i < len(b) else None
      if ra != rb: out.append(f"  [{i}] A={ra!r} B={rb!r}")
  for tag, s in (("A", a), ("B", b)):
    seen:dict[tuple, set] = {}
    for r in s:
      if r[0] == "K": seen.setdefault((r[1], r[3], r[4], len(r[7])), set()).add(r[7])
    multi = {k: v for k, v in seen.items() if len(v) > 1}
    if multi:
      out.append(f"step {tag}: {len(multi)} config(s) launched >1x in-step with DIFFERENT vals (UBO race hazard):")
      for (pid, nbufs, bufs, nvals), v in list(multi.items())[:5]:
        b0 = bufs[0] if bufs else "-"
        out.append(f"  prog#{pid} ({dev._arm_names.get(pid)}) nbufs={nbufs} nvals={nvals} bufs0={b0}: "
                   f"{len(v)} distinct vals sets, e.g. {list(v)[:2]}")
  for tag, s in (("A", a), ("B", b)):
    for i, r in enumerate(s):
      if r[0] in ("H2D", "D2H"):
        out.append(f"step {tag} [{i}] {r[0]}: buf=0x{r[1]:x} off={r[2]} size={r[3]} | staging=0x{r[4]:x} size={r[5]} | bytes={r[6]}")
  return "\n".join(out) + "\n"

class VulkanAllocator(Allocator):
  # a few host-visible staging buffers, kept mapped, reused for every host<->device copy.
  # safe to reuse without per-buffer tracking: kernels accumulate in the pending command
  # buffer (submitted at synchronize/copy boundaries), so a LRU-reused buffer may still be
  # touched by pending or in-flight work; _copyin drains the device first, _copyout appends
  # its D2H after the pending kernels and waits on its fence before the CPU reads.
  STAGING_MAX = 4
  def __init__(self, dev:Compiled):
    super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)
    self._staging_bufs:list = []
  def alloc(self, size:int, options:BufferSpec|None=None) -> BufferStorage:
    # check before the base alloc, whose error wrapper would re-raise this as a bare MemoryError
    if (maxb := getattr(self.dev, "max_buffer", None)) and size > maxb:
      raise RuntimeError(f"VULKAN: {size/2**20:.0f} MiB buffer exceeds this device's {maxb/2**20:.0f} MiB per-buffer limit. "
                         f"This Adreno driver mis-addresses larger buffers and would return silently-wrong values, so the "
                         f"allocation is refused. Run in smaller sub-batches so no single tensor exceeds {maxb} bytes "
                         f"(see extra/vulkan/ADRENO_256MB.md).")
    return super().alloc(size, options)
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    if options.external_ptr is not None: raise RuntimeError("VULKAN does not support external_ptr")
    if options.host:
      vbuf = self.dev.rt.buffer(size, host_visible=True)
      return BufferStorage(VulkanBuffer(vbuf, 0), None, MMIOInterface(mv_address(vbuf.map()), size, fmt='B'))
    # device-local: on a dGPU this is VRAM (the host-visible heap is a small ReBAR window),
    # on an APU it is the same unified memory, on llvmpipe rt.buffer falls back to host memory
    return BufferStorage(VulkanBuffer(self.dev.rt.buffer(size, host_visible=False), 0), None, None)
  def _free(self, storage:BufferStorage, options:BufferSpec): pass  # VkBuffers live until rt.close(); the LRU cache reuses them
  def _offset(self, buf:VulkanBuffer, size:int, offset:int) -> VulkanBuffer: return VulkanBuffer(buf.vbuf, buf.offset + offset)
  def _staging(self, nbytes:int):
    if (s := min((b for b in self._staging_bufs if b.size >= nbytes), key=lambda b: b.size, default=None)) is not None: return s
    if len(self._staging_bufs) >= self.STAGING_MAX:  # drop the smallest; it may still be in flight
      self.dev.rt.synchronize()
      smallest = min(self._staging_bufs, key=lambda b: b.size)
      self._staging_bufs.remove(smallest)
      self.dev.rt.free_buffer(smallest)
    s = self.dev.rt.buffer(nbytes, host_visible=True)
    s.map()
    self._staging_bufs.append(s)
    return s
  def _copyin(self, dest:VulkanBuffer, src:memoryview):
    # the LRU-reused dest may still be read/written by pending (unsubmitted) or in-flight
    # kernels; drain everything before reusing it
    rt = self.dev.rt
    t0 = rt._pt0()
    if (fd:=self.dev._fd) is not None:
      fd._rec(("H2D", dest.vbuf, dest.offset, src.nbytes, bytes(src) if src.nbytes <= 64 else None))
    if self.dev._arm:
      self.dev._arm_step_done()  # a host->device copyin starts a new step (VK_ARMDIFF)
    self.dev.synchronize()
    st = self._staging(src.nbytes)
    if self.dev._arm:
      self.dev._arm_cur.append(("H2D", dest.vbuf.handle.value, dest.offset, dest.vbuf.size,
                                st.handle.value, st.size, src.nbytes))
    _copy_to_arr(st.map(), src.cast('B'), src.nbytes)
    self.dev.rt.cmd_copy(dest.vbuf, st, src.nbytes, dst_off=dest.offset)
    self.dev.rt.submit()  # async H2D in its own cb; same-queue order places it before later kernels
    rt._pt("copyin", t0)
  def _copyout(self, dest:memoryview, src:VulkanBuffer):
    rt = self.dev.rt
    t0 = rt._pt0()
    if (fd:=self.dev._fd) is not None:
      fd._rec(("D2H", src.vbuf, src.offset, dest.nbytes))
    st = self._staging(dest.nbytes)
    # the D2H is recorded after any pending kernels in the current cb, so it runs after them
    if self.dev._arm:
      self.dev._arm_cur.append(("D2H", src.vbuf.handle.value, src.offset, src.vbuf.size,
                                st.handle.value, st.size, dest.nbytes))
    self.dev.rt.cmd_copy(st, src.vbuf, dest.nbytes, src_off=src.offset)
    self.dev.rt.submit(wait=True)  # wait before the CPU reads staging
    _copy_to_mv(dest, st.map(), dest.nbytes)
    rt._pt("copyout", t0)
  def _map(self, buf): raise RuntimeError("VULKAN cross-device map not supported")

class VulkanProgram(Program['VulkanDevice']):
  def __init__(self, dev:VulkanDevice, obj:TinyELF):
    self.dev, self.name, self.signature, self.lib = dev, obj.name, obj.signature, obj.lib
    self._cache:dict[tuple, tuple] = {}
    self._n = 0
    # VK_REPLAY: per-signature template CBTs (rkey = (nbufs, nvals, bufs..., grid)), the
    # rotation cursor for the VK_REPLAY_POOL templates, and the sighting count per rkey
    self._tpl:dict[tuple, list] = {}
    self._tpl_rr:dict[tuple, int] = {}
    self._rkey_n:dict[tuple, int] = {}
    self._arm_id = dev._arm_alloc(self) if dev._arm else -1

  def _launch_cfg(self, bufs:tuple[VulkanBuffer, ...], nvals:int, key:tuple|None=None) -> tuple:
    if key is None:
      key = (len(bufs), nvals) + tuple((b.vbuf.handle.value, b.offset) for b in bufs)
    if (cfg:=self._cache.get(key)) is not None: return cfg, False
    rt = self.dev.rt
    ubo = rt.buffer(8 * nvals, host_visible=True) if nvals else None
    ubo_map = ubo.map() if ubo is not None else None
    bindings:tuple[tuple[int, int, int, int], ...] = ()
    dbufs:list = []
    if ubo is not None:
      bindings = ((0, 0, vkrt.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, vkrt.VK_SHADER_STAGE_COMPUTE_BIT),)
      dbufs.append(ubo)
    for i, b in enumerate(bufs):
      bindings += ((0, i + 1, vkrt.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, vkrt.VK_SHADER_STAGE_COMPUTE_BIT),)
      dbufs.append(_desc(b))
    dsl = rt.create_descriptor_set_layout(bindings, dbufs)
    module = rt.create_shader_module(self.lib)
    pipeline = rt.create_compute_pipeline(module, "main", [dsl])
    cfg = (pipeline, dsl, ubo, ubo_map)
    self._cache[key] = cfg
    return cfg, True

  def __del__(self):
    # release this program's pipelines/modules/descriptors as soon as it is unreferenced
    # (BEAM search creates and drops thousands of candidate programs per run)
    rt = getattr(getattr(self, "dev", None), "rt", None)
    if rt is None: return
    for pipeline, dsl, ubo, _ in list(getattr(self, "_cache", {}).values()):
      try: rt.destroy_cfg(pipeline, dsl, ubo)
      except Exception: pass

  def __call__(self, *bufs:int|Any, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int|float, ...]=(), wait:bool=False, timeout:int|None=None) -> float|None:
    bufs = cast(tuple[VulkanBuffer, ...], bufs)
    nvals = len(vals)
    rt = self.dev.rt
    t0 = rt._pt0()
    if (fd:=self.dev._fd) is not None:
      fd._rec(("K", self, bufs, global_size, vals,
               tuple(self.signature[len(bufs) + i][2] for i in range(nvals))))
    cfg_key = (len(bufs), nvals) + tuple((b.vbuf.handle.value, b.offset) for b in bufs)
    (pipeline, dsl, ubo, ubo_map), is_new = self._launch_cfg(bufs, nvals, cfg_key)
    if nvals:
      var_dts = tuple(self.signature[len(bufs) + i][2] for i in range(nvals))
      u0 = rt._pt0()
      ubo_map[:8 * nvals] = _pack_params(vals, var_dts)
      rt._pt("ubo", u0)
    if self.dev._arm:
      self.dev._arm_cur.append(("K", self._arm_id, self.name, len(bufs), cfg_key[2:], global_size,
                                tuple(self.signature[len(bufs) + i][0] for i in range(nvals)), vals))
    if is_new and _warmup_vendors(rt):
      # throwaway full-grid dispatch: primes the cold pipeline so the real launch below is
      # its second (the first loses stores on NV 550); must be the full grid -- a smaller
      # warmup grid does not prime it. Drained before the real dispatch.
      rt.cmd_bind_pipeline(pipeline)
      rt.cmd_bind_descriptor_sets(pipeline, dsl.set)
      rt.cmd_dispatch(*global_size)
      rt.submit(wait=True)
    # global_size is the grid (workgroup count) and local_size is the block size (baked into the
    # pipeline from the shader's workgroup_size); they are independent in Vulkan (no multiple rule).
    if os.environ.get("VKDEBUG"):
      print(f"[VULKAN] {self.name} global={global_size} local={local_size} nbufs={len(bufs)} nvals={nvals}", flush=True)
      if os.environ.get("VKDUMP"):
        import re as _re
        safe = _re.sub(r"\W+", "_", self.name)
        with open(f"/tmp/opencode/kernels/{self._n}_{safe}.spv", "wb") as f: f.write(self.lib)
        self._n += 1
    if wait:
      # timeout 0 (beam's int(early_stop*1e3) truncates to it for microsecond-scale times)
      # would be a 0 ns non-blocking fence wait: use the default timeout instead
      timeout = self.dev.wait_timeout_ms if not timeout else timeout
      # timing contract (time_call/BEAM uses the return value as the kernel time): drain
      # first so the measurement isolates this dispatch from the pending batch
      self.dev.rt.synchronize(timeout)
    st = time.perf_counter() if wait else 0
    if not wait and rt._replay:
      # pre-recorded resubmit: a repeat of this exact (program, bufs, nvals, grid) signature
      # re-uses a recorded TEMPLATE CBT instead of re-recording bind/bdesc/dispatch. The UBO
      # rewrite above already refreshed the per-step vals it reads at execute time, and the
      # CBT's descriptor set still addresses the same buffer handles (VulkanAllocator never
      # destroys VkBuffers -- _free is a no-op and the LRU reuses the same handle), so the
      # recorded dispatch is still valid. The template flows through the same VK_STREAM
      # chunked chained submits; replay_cb ends any active fresh unit first (pending order
      # must stay == program order) and refuses a template whose previous resubmit is still
      # pending/in-flight (a primary CBT may not be pending twice).
      rkey = cfg_key + (global_size,)
      n = self._rkey_n.get(rkey, 0)
      self._rkey_n[rkey] = n + 1
      if n:
        tpls = self._tpl.setdefault(rkey, [])
        # fill the rotating pool first (VK_REPLAY_POOL templates), then rotate through them.
        # A template whose previous resubmit is still pending/in-flight refuses the resubmit
        # (a primary CBT may not be pending twice) -- try the next pool slot, else fall
        # through to a fresh record below (correct, slightly slower for that call).
        if len(tpls) < rt._replay_pool:
          cbt = rt.record_template(pipeline, dsl.set, global_size)
          if cbt is not None: tpls.append(cbt)
        for _ in range(len(tpls)):
          i = self._tpl_rr.get(rkey, 0) % len(tpls)
          self._tpl_rr[rkey] = i + 1
          if rt.replay_cb(tpls[i]):
            rt._repl[0] += 1
            rt._pt("call", t0)
            return None
    self.dev.rt.cmd_bind_pipeline(pipeline)
    self.dev.rt.cmd_bind_descriptor_sets(pipeline, dsl.set)
    self.dev.rt.cmd_dispatch(*global_size)
    # record-only by default: the dispatch stays in the pending command buffer and is
    # submitted at the next boundary (synchronize / _copyin / _copyout / wait=True launch).
    # vals are per-program compile-time constants, so the UBO write above stays valid while
    # earlier launches of this program are still pending.
    if wait:
      # timeout (BEAM passes a per-candidate device timeout) is honored via the fence wait
      self.dev.rt.submit(wait=True, timeout_ms=timeout)
      rt._pt("call", t0)
      return time.perf_counter() - st
    rt._pt("call", t0)
    return None

class VulkanDevice(Compiled):
  wait_timeout_ms = 30000
  def __init__(self, device:str=""):
    self.rt = vkrt.VkRt()
    # VK_ARMDIFF=1 (default 0): record every VulkanProgram.__call__ (program id, buffer
    # handles, grid, vals) and every H2D/D2H copy, one record-list per step (a step starts
    # at an H2D copyin), and after 8 kernel steps diff the last two steady ones into
    # /work/opencode/radv_batch/armdiff.txt to list exactly what changes per decode step.
    self._arm = os.environ.get("VK_ARMDIFF", "0") == "1"
    self._arm_dump_f = open("/tmp/opencode/radv_batch/armdump.jsonl", "w") if os.environ.get("VK_ARMDIFF_DUMP", "0") == "1" else None
    self._arm_next = 0
    self._arm_names:dict[int, str] = {}
    self._arm_cur: list = []
    self._arm_steps: list = []
    self._fd:VulkanFastDecode|None = None
    # VK_ARCH overrides the target arch (e.g. dump Adreno-keyed spv from a desktop box); the
    # arch selects the renderer/limits, independent of the physical device running the spv.
    arch = getenv("VK_ARCH", "") or ("radv" if self.rt.vendor == 0x1002 else f"vk{self.rt.vendor:04x}")
    self.max_buffer = getenv("VK_MAX_BUFFER", _VULKAN_MAX_BUFFER.get(arch) or 0) or None
    super().__init__(device, VulkanAllocator(self), [SPIRVRenderer], VulkanProgram, arch=arch)
  def _arm_alloc(self, prg) -> int:
    self._arm_next += 1
    self._arm_names[self._arm_next] = prg.name
    return self._arm_next
  def _arm_step_done(self):
    # an H2D copyin starts a new step: close the previous step's record. Keep only steps that
    # ran kernels (weight-load copy bursts during model init don't count); after 8 steady
    # kernel steps write the diff report and stop collecting.
    if not self._arm_cur: return
    cur, self._arm_cur = self._arm_cur, []
    if not any(r[0] == "K" for r in cur): return
    self._arm_steps.append(cur)
    if self._arm_dump_f is not None:
      import json as _json
      self._arm_dump_f.write(_json.dumps(cur) + "\n")
    if len(self._arm_steps) >= 8:
      with open("/work/opencode/radv_batch/armdiff.txt", "w") as f: f.write(_arm_report(self))
      self._arm = False
  def synchronize(self, timeout:int|None=None):
    # no timeline on this backend (work is not signaled into dev.timeline): flush the
    # pending command buffer, then wait on the submit ring's fences
    self.rt.synchronize(timeout)
  def finalize(self):
    try: super().finalize()
    except RuntimeError as e: print(f"VULKAN synchronization failed before finalizing: {e}")
    self.rt.close()

class VulkanFastDecode:
  # VULKAN_FASTDECODE=1: re-drive a captured steady decode step (one generated token) without
  # the framework (JIT replay / linear rewrite / schedule / program dispatch). The recipe is the
  # captured step's kernel sequence (per-kernel pipeline + descriptor set + UBO) and its H2D/D2H
  # staging, re-recorded fresh every step with the per-step (token, start_pos) update. The
  # Step-1 stability audit (radv_batch/FASTDECODE.md) shows the steady step is deterministic
  # except for the 4B token H2D payload, the single start_pos val in every val-kernel, and the
  # start_pos+1 grid-x of the attention kernels; finish() validates exactly that and refuses to
  # build a recipe for anything else (the caller then stays on the normal path).
  def __init__(self, dev:VulkanDevice):
    self.dev, self.rt = dev, dev.rt
    self._cap: list = []
    self._armed = False
    self.ready = False
    self.ops: list = []
    self._d2h: tuple|None = None
    self._progs: list = []   # keep the captured programs (and their cfg caches) alive
    self.op_names: list[str] = []  # per-op label, parallel to self.ops (for TS profiling)
    # private mapped staging: never LRU-reused, so a fast step needs no device drain. Each
    # H2D gets its own 64B slice: the copies execute only after the whole step is recorded,
    # so a shared slot would carry the last writer's bytes to every copy reading it (the
    # normal path hides this with its per-copyin device drain; a fast step has none)
    self._st_in = dev.rt.buffer(64 * 16, host_visible=True)
    self._st_out = dev.rt.buffer(64, host_visible=True)
    self._st_in_arr = self._st_in.map()
    self._st_out_arr = self._st_out.map()
  def arm(self):
    if self._armed or self.ready: raise RuntimeError("VULKAN_FASTDECODE: double arm")
    self._cap, self._armed = [], True
  def _rec(self, rec):
    if self._armed: self._cap.append(rec)
  def finish(self, sp:int):
    if not self._armed: raise RuntimeError("VULKAN_FASTDECODE: finish without arm")
    self._armed = False
    ks = [r for r in self._cap if r[0] == "K"]
    h2ds = [r for r in self._cap if r[0] == "H2D"]
    if not ks or not h2ds: raise RuntimeError(f"VULKAN_FASTDECODE: bad capture ({len(ks)}K {len(h2ds)}H2D)")
    toks = [r for r in h2ds if r[3] == 4]
    if len(toks) != 1: raise RuntimeError(f"VULKAN_FASTDECODE: expected one 4B token H2D, got {len(toks)}")
    for r in h2ds:
      if not (r[1].handle.value == toks[0][1].handle.value and r[2] == toks[0][2]) and r[4] is None:
        raise RuntimeError(f"VULKAN_FASTDECODE: large H2D payload in the captured step ({r[3]}B)")
    # the step may start with the previous step's input materialization (a D2H read of the
    # argmax buffer before the token H2D): the fast path writes the token directly, so only
    # the D2H after the token H2D (the out.item() read) is part of the recipe
    i_tok = next(i for i, r in enumerate(self._cap) if r[0] == "H2D" and r[1].handle.value == toks[0][1].handle.value and r[2] == toks[0][2])
    d2hs = [r for i, r in enumerate(self._cap) if r[0] == "D2H" and i > i_tok]
    if len(d2hs) != 1: raise RuntimeError(f"VULKAN_FASTDECODE: expected one D2H after the token H2D, got {len(d2hs)}")
    self._d2h = (d2hs[0][1], d2hs[0][2])
    ops: list = []
    self.op_names = []
    h2d_slot = 0
    for r in self._cap:
      if r[0] == "K":
        prog, bufs, grid, vals, var_dts = r[1], r[2], r[3], r[4], r[5]
        if vals and any(v != sp for v in vals): raise RuntimeError(f"VULKAN_FASTDECODE: non-start_pos vals {vals} in {prog.name}")
        if grid[0] == sp + 1 and "start_pos" not in prog.name:
          raise RuntimeError(f"VULKAN_FASTDECODE: grid-x {sp + 1} on non-start_pos kernel {prog.name}")
        (pipeline, dsl, ubo, ubo_map), _ = prog._launch_cfg(bufs, len(vals))
        self._progs.append(prog)
        ops.append(("K", pipeline, dsl.set, ubo_map, len(vals), var_dts, grid, grid[0] == sp + 1))
        self.op_names.append(prog.name)
      elif r[0] == "H2D":
        if h2d_slot >= 16 or r[3] > 64: raise RuntimeError("VULKAN_FASTDECODE: too many/large H2D ops in the step")
        is_tok = r[1].handle.value == toks[0][1].handle.value and r[2] == toks[0][2]
        ops.append(("H2D", r[1], r[2], r[3], None if is_tok else r[4], is_tok, 64 * h2d_slot))
        self.op_names.append("H2D_token" if is_tok else "H2D_seed")
        h2d_slot += 1
      # the single D2H is stored in self._d2h, not in the op list
    self.ops = ops
    self.ready = True
  def step(self, token:int, sp:int) -> int:
    # re-record the captured sequence with this step's (token, start_pos); the final
    # submit(wait=True) fence covers the whole step (the VK_STREAM cross-submit semaphore
    # chain orders the chunks, and the D2H is the last recorded command). Every H2D payload
    # is written to its own staging slice up front: the copies execute only after recording
    # continues, so a later host write to a shared slot would corrupt an earlier pending copy
    rt = self.rt
    assert self._d2h is not None
    self.last_ts: list|None = None
    ts = rt.ts_pool(len(self.ops) + 2) if getenv("VULKAN_FASTDECODE_TS") else None
    for op in self.ops:
      if op[0] != "H2D": continue
      _, vbuf, off, nbytes, payload, is_tok, base = op
      if is_tok: struct.pack_into("<i", self._st_in_arr, base, token)
      else: self._st_in_arr[base:base + nbytes] = payload
    if ts is not None: rt.ts_write(ts, 0)  # standalone first CBT: step start
    for i, op in enumerate(self.ops):
      if op[0] == "K":
        _, pipeline, dset, ubo_map, nvals, var_dts, grid, sp_x = op
        if nvals: ubo_map[:8 * nvals] = _pack_params((sp,) * nvals, var_dts)
        rt.cmd_bind_pipeline(pipeline)
        rt.cmd_bind_descriptor_sets(pipeline, dset)
        rt.cmd_dispatch(sp + 1 if sp_x else grid[0], grid[1], grid[2])
      else:
        _, vbuf, off, nbytes, payload, is_tok, base = op
        rt.cmd_copy(vbuf, self._st_in, nbytes, dst_off=off, src_off=base)
      if ts is not None: rt.ts_write(ts, i + 1)
    d2h_vbuf, d2h_off = self._d2h
    rt.cmd_copy(self._st_out, d2h_vbuf, 4, src_off=d2h_off)
    if ts is not None: rt.ts_write(ts, len(self.ops) + 1)
    rt.submit(wait=True)
    if ts is not None: self.last_ts = rt.ts_read(ts, len(self.ops) + 2)
    return struct.unpack_from("<i", self._st_out_arr, 0)[0]
