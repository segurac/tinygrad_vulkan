from __future__ import annotations
import struct, os, time
from typing import Any, cast
from tinygrad.device import Compiled, Allocator, BufferStorage, BufferSpec, MMIOInterface, Program, TinyELF
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import mv_address
from tinygrad.renderer.nir import SPIRVRenderer
from tinygrad.runtime.support import vulkan_rt as vkrt

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
    self.dev.synchronize()
    st = self._staging(src.nbytes)
    st.map()[0:src.nbytes] = src.cast('B')
    self.dev.rt.cmd_copy(dest.vbuf, st, src.nbytes, dst_off=dest.offset)
    self.dev.rt.submit()  # async H2D in its own cb; same-queue order places it before later kernels
  def _copyout(self, dest:memoryview, src:VulkanBuffer):
    st = self._staging(dest.nbytes)
    # the D2H is recorded after any pending kernels in the current cb, so it runs after them
    self.dev.rt.cmd_copy(st, src.vbuf, dest.nbytes, src_off=src.offset)
    self.dev.rt.submit(wait=True)  # wait before the CPU reads staging
    dest[:] = bytes(st.map()[0:dest.nbytes])
  def _map(self, buf): raise RuntimeError("VULKAN cross-device map not supported")

class VulkanProgram(Program['VulkanDevice']):
  def __init__(self, dev:VulkanDevice, obj:TinyELF):
    self.dev, self.name, self.signature, self.lib = dev, obj.name, obj.signature, obj.lib
    self._cache:dict[tuple, tuple] = {}
    self._n = 0

  def _launch_cfg(self, bufs:tuple[VulkanBuffer, ...], nvals:int) -> tuple:
    key = (len(bufs), nvals) + tuple((b.vbuf.handle.value, b.offset) for b in bufs)
    if (cfg:=self._cache.get(key)) is not None: return cfg
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
    return cfg

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
    pipeline, dsl, ubo, ubo_map = self._launch_cfg(bufs, nvals)
    if nvals:
      var_dts = tuple(self.signature[len(bufs) + i][2] for i in range(nvals))
      ubo_map[:8 * nvals] = _pack_params(vals, var_dts)
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
      return time.perf_counter() - st
    return None

class VulkanDevice(Compiled):
  wait_timeout_ms = 30000
  def __init__(self, device:str=""):
    self.rt = vkrt.VkRt()
    super().__init__(device, VulkanAllocator(self), [SPIRVRenderer], VulkanProgram,
                     arch="radv" if self.rt.vendor == 0x1002 else f"vk{self.rt.vendor:04x}")
  def synchronize(self, timeout:int|None=None):
    # no timeline on this backend (work is not signaled into dev.timeline): flush the
    # pending command buffer, then wait on the submit ring's fences
    self.rt.synchronize(timeout)
  def finalize(self):
    try: super().finalize()
    except RuntimeError as e: print(f"VULKAN synchronization failed before finalizing: {e}")
    self.rt.close()
