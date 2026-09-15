from __future__ import annotations
import struct, os
from typing import Any, cast
from tinygrad.device import Compiled, Allocator, BufferStorage, BufferSpec, MMIOInterface, Program, TinyELF
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import mv_address
from tinygrad.renderer.nir import SPIRVRenderer
from tinygrad.runtime.support import vulkan_rt as vkrt

class VulkanBuffer:
  # a view into a host-visible VkBuffer; the descriptor addresses (handle, offset)
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
  def __init__(self, dev:Compiled): super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage:
    if options.external_ptr is not None: raise RuntimeError("VULKAN does not support external_ptr")
    vbuf = self.dev.rt.buffer(size, host_visible=True)
    mapped = vbuf.map()
    return BufferStorage(VulkanBuffer(vbuf, 0), None, MMIOInterface(mv_address(mapped), size, fmt='B'))
  def _free(self, storage:BufferStorage, options:BufferSpec): pass  # VkBuffers live until rt.close(); the LRU cache reuses them
  def _offset(self, buf:VulkanBuffer, size:int, offset:int) -> VulkanBuffer: return VulkanBuffer(buf.vbuf, buf.offset + offset)
  def _copyin(self, dest:VulkanBuffer, src:memoryview):
    self.dev.synchronize()
    dest.vbuf.map()[dest.offset:dest.offset+src.nbytes] = src.cast('B')
  def _copyout(self, dest:memoryview, src:VulkanBuffer):
    self.dev.synchronize()
    dest[:] = bytes(src.vbuf.map()[src.offset:src.offset+dest.nbytes])
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
    cfg = (pipeline, dsl.set, ubo_map)
    self._cache[key] = cfg
    return cfg

  def __call__(self, *bufs:int|Any, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int|float, ...]=(), wait:bool=False, timeout:int|None=None) -> float|None:
    bufs = cast(tuple[VulkanBuffer, ...], bufs)
    nvals = len(vals)
    pipeline, desc_set, ubo_map = self._launch_cfg(bufs, nvals)
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
    self.dev.rt.cmd_bind_pipeline(pipeline)
    self.dev.rt.cmd_bind_descriptor_sets(pipeline, desc_set)
    self.dev.rt.cmd_dispatch(*global_size)
    self.dev.rt.submit()
    return None

class VulkanDevice(Compiled):
  wait_timeout_ms = 30000
  def __init__(self, device:str=""):
    self.rt = vkrt.VkRt()
    super().__init__(device, VulkanAllocator(self), [SPIRVRenderer], VulkanProgram,
                     arch="radv" if self.rt.vendor == 0x1002 else f"vk{self.rt.vendor:04x}")
  def finalize(self):
    try: super().finalize()
    except RuntimeError as e: print(f"VULKAN synchronization failed before finalizing: {e}")
    self.rt.close()
