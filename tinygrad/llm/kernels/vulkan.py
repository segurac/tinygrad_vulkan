import functools
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import getenv, prod
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

GGML_Q4_K = 12
GGML_Q5_K = 13
_Q4K_BLOCK_SIZE, _Q4K_BLOCK_BYTES = 256, 144
_Q5K_BLOCK_SIZE, _Q5K_BLOCK_BYTES = 256, 176

@functools.cache
def vulkan_q4k_enabled(device) -> bool:
  # VULKAN_Q4K overrides the legacy VULKAN_QUANT; both default ON
  return str(device).startswith("VULKAN") and getenv("VULKAN_Q4K", getenv("VULKAN_QUANT", 1)) != 0

@functools.cache
def vulkan_q5k_enabled(device) -> bool:
  # default ON: 4-way A/B on the APU showed a net +11-12% decode speedup with Q5_K down experts fused
  return str(device).startswith("VULKAN") and getenv("VULKAN_Q5K", 1) != 0

@functools.cache
def vulkan_quant_supported(device) -> bool:
  return vulkan_q4k_enabled(device) or vulkan_q5k_enabled(device)

def _find_quant_bytes(weight: Tensor, ggml_type: int, block_size: int, block_bytes: int) -> Tensor|None:
  # find the flat quantized byte buffer a dequantized weight reads from (zero-copy alias of the GGUF staging buffer)
  if not isinstance(n:=weight.numel(), int) or n % block_size: return None
  # note: detection is numel-based and shape-agnostic, so it also works for the 3-D (n_experts, out, in) routed-expert weights
  target = n // block_size * block_bytes
  raw = next((u for u in weight.uop.toposort() if u.dtype == dtypes.uint8 and prod(u.shape) == target and u.op in (Ops.SHRINK, Ops.BUFFER)), None)
  if raw is None: return None
  # only storage/order-preserving views may sit between the bytes and the dequantization expression
  def unwrapped(u:UOp) -> UOp:
    while u.op in (Ops.RESHAPE, Ops.STAGE) or (u.op is Ops.CAST and dtypes.is_float(u.dtype) and dtypes.is_float(u.src[0].dtype)):
      u = u.src[0]
    return u
  if unwrapped(weight.uop).key != unwrapped(ggml_data_to_tensor(Tensor(raw), n, ggml_type).uop).key: return None
  if raw.contiguous_view_offset() is None or raw.buf_uop.dtype != dtypes.uint8: return None
  return Tensor(raw).reshape(-1).contiguous()

def find_q4k_bytes(weight: Tensor) -> Tensor|None:
  return _find_quant_bytes(weight, GGML_Q4_K, _Q4K_BLOCK_SIZE, _Q4K_BLOCK_BYTES)

def find_q5k_bytes(weight: Tensor) -> Tensor|None:
  return _find_quant_bytes(weight, GGML_Q5_K, _Q5K_BLOCK_SIZE, _Q5K_BLOCK_BYTES)

def find_q4k_expert_bytes(weight: Tensor) -> Tensor|None:
  # routed-expert (n_experts, out, in) Q4_K weight; the size check + dequant key match are numel-based, so reuse the dense detector
  return find_q4k_bytes(weight)

@functools.cache
def q4k_gemv_kernel(out:UOp, W8:UOp, x:UOp, out_features:int, in_features:int) -> UOp:
  groups = in_features // 32
  row_bytes = (in_features // _Q4K_BLOCK_SIZE) * _Q4K_BLOCK_BYTES
  o = UOp.range(out_features, 0, AxisType.GLOBAL)
  g = UOp.range(groups, 1, AxisType.LOCAL)     # one thread per 32-elem group
  b = UOp.range(32, 2, AxisType.REDUCE)        # 32 elems/group, one per byte
  block, sg = g // 8, g % 8
  bbase = o * row_bytes + block * _Q4K_BLOCK_BYTES
  def B(k): return W8[bbase + k]
  d    = ((B(0).cast(dtypes.uint16) | B(1).cast(dtypes.uint16).lshift(8))).bitcast(dtypes.float16).float()
  dmin = ((B(2).cast(dtypes.uint16) | B(3).cast(dtypes.uint16).lshift(8))).bitcast(dtypes.float16).float()
  sglt4 = (sg < 4)
  sc = sglt4.where(B(4 + sg) & 63, (B(8 + sg) & 15) | ((B(sg) >> 6).lshift(4)))
  mn = sglt4.where(B(8 + sg) & 63, (B(8 + sg) >> 4) | ((B(4 + sg) >> 6).lshift(4)))
  ds, dm = d * sc.float(), dmin * mn.float()
  sg_even = (sg & 1).eq(0)
  wbyte = W8[bbase + 16 + (sg // 2) * 32 + b]
  q = sg_even.where(wbyte & 15, (wbyte >> 4) & 15)
  term = (ds * q.float() - dm) * x[g * 32 + b]
  total = term.reduce(b, arg=Ops.ADD).reduce(g, arg=Ops.ADD)
  return out[o].store(total).end(o).sink(
    arg=KernelInfo(name=f"q4k_gemv_{out_features}_{in_features}", opts_to_apply=()))

def vulkan_q4k_linear(qweight: Tensor, x: Tensor, in_features: int, out_features: int) -> Tensor:
  shape = x.shape
  out = Tensor.empty(out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(q4k_gemv_kernel, out_features=out_features, in_features=in_features)
  res = Tensor.custom_kernel(out, qweight, x.reshape(in_features), fxn=fxn)[0]
  return res.cast(x.dtype).reshape(*shape[:-1], out_features)

@functools.cache
def q4k_moe_gemv_kernel(out:UOp, W8:UOp, x:UOp, sel:UOp, out_features:int, in_features:int, n_experts:int) -> UOp:
  # batched gather GEMV over the k active experts: one GLOBAL work item per (expert slot, output row)
  # sel is the (1, 1, k) decode view (possibly non-contiguous); index it directly so no copy kernel is needed
  k = sel.shape[-1]
  groups = in_features // 32
  row_bytes = (in_features // _Q4K_BLOCK_SIZE) * _Q4K_BLOCK_BYTES
  expert_bytes = out_features * row_bytes
  o = UOp.range(k * out_features, 0, AxisType.GLOBAL)
  e, row = o // out_features, o % out_features
  expert = sel[0, 0, e]                # one int32 read per work item, constant inside the reduce loops
  g = UOp.range(groups, 1, AxisType.LOCAL)     # one thread per 32-elem group
  b = UOp.range(32, 2, AxisType.REDUCE)        # 32 elems/group, one per byte
  block, sg = g // 8, g % 8
  bbase = expert * expert_bytes + row * row_bytes + block * _Q4K_BLOCK_BYTES
  def B(k): return W8[bbase + k]
  d    = ((B(0).cast(dtypes.uint16) | B(1).cast(dtypes.uint16).lshift(8))).bitcast(dtypes.float16).float()
  dmin = ((B(2).cast(dtypes.uint16) | B(3).cast(dtypes.uint16).lshift(8))).bitcast(dtypes.float16).float()
  sglt4 = (sg < 4)
  sc = sglt4.where(B(4 + sg) & 63, (B(8 + sg) & 15) | ((B(sg) >> 6).lshift(4)))
  mn = sglt4.where(B(8 + sg) & 63, (B(8 + sg) >> 4) | ((B(4 + sg) >> 6).lshift(4)))
  ds, dm = d * sc.float(), dmin * mn.float()
  sg_even = (sg & 1).eq(0)
  wbyte = W8[bbase + 16 + (sg // 2) * 32 + b]
  q = sg_even.where(wbyte & 15, (wbyte >> 4) & 15)
  term = (ds * q.float() - dm) * x[g * 32 + b]
  total = term.reduce(b, arg=Ops.ADD).reduce(g, arg=Ops.ADD)
  return out[e, row].store(total.cast(out.dtype)).end(o).sink(
    arg=KernelInfo(name=f"q4k_moe_gemv_{out_features}_{in_features}_{n_experts}", opts_to_apply=()))

def vulkan_q4k_expert_linear(qweight: Tensor, sel: Tensor, x: Tensor, in_features: int, out_features: int, n_experts: int) -> Tensor:
  # decode-only (B*T==1) routed-expert GEMV: x (B, T, 1, in), sel (B, T, k) int32 -> out (B, T, k, out_features)
  k = sel.shape[-1]
  out = Tensor.empty(k, out_features, dtype=x.dtype, device=x.device)
  fxn = functools.partial(q4k_moe_gemv_kernel, out_features=out_features, in_features=in_features, n_experts=n_experts)
  res = Tensor.custom_kernel(out, qweight, x.reshape(in_features), sel, fxn=fxn)[0]
  return res.reshape(*x.shape[:-2], k, out_features)

@functools.cache
def q5k_moe_gemv_kernel(out:UOp, W8:UOp, x:UOp, sel:UOp, out_features:int, in_features:int, n_experts:int) -> UOp:
  # batched gather GEMV over the k active down experts, Q5_K dequant (ggml_type 13):
  # 176-byte block = d:2 dmin:2 scales:12 qh:32 qs:128; per group sg, element i:
  # q = nibble(qs[(sg//2)*32+i], high iff sg odd) + 16 * bit(sg) of qh[i]
  # x is the flat (k, in) per-expert activation rows: expert slot e uses row e
  k = sel.shape[-1]
  groups = in_features // 32
  row_bytes = (in_features // _Q5K_BLOCK_SIZE) * _Q5K_BLOCK_BYTES
  expert_bytes = out_features * row_bytes
  o = UOp.range(k * out_features, 0, AxisType.GLOBAL)
  e, row = o // out_features, o % out_features
  expert = sel[0, 0, e]                # one int32 read per work item, constant inside the reduce loops
  g = UOp.range(groups, 1, AxisType.LOCAL)     # one thread per 32-elem group
  b = UOp.range(32, 2, AxisType.REDUCE)        # 32 elems/group, one per byte
  block, sg = g // 8, g % 8
  bbase = expert * expert_bytes + row * row_bytes + block * _Q5K_BLOCK_BYTES
  def B(k): return W8[bbase + k]
  d    = ((B(0).cast(dtypes.uint16) | B(1).cast(dtypes.uint16).lshift(8))).bitcast(dtypes.float16).float()
  dmin = ((B(2).cast(dtypes.uint16) | B(3).cast(dtypes.uint16).lshift(8))).bitcast(dtypes.float16).float()
  sglt4 = (sg < 4)
  sc = sglt4.where(B(4 + sg) & 63, (B(8 + sg) & 15) | ((B(sg) >> 6).lshift(4)))
  mn = sglt4.where(B(8 + sg) & 63, (B(8 + sg) >> 4) | ((B(4 + sg) >> 6).lshift(4)))
  ds, dm = d * sc.float(), dmin * mn.float()
  sg_even = (sg & 1).eq(0)
  wbyte = W8[bbase + 48 + (sg // 2) * 32 + b]
  q = sg_even.where(wbyte & 15, (wbyte >> 4) & 15)
  q = q + (16 * ((W8[bbase + 16 + b] >> sg) & 1))   # qh top bit; sg is LOCAL, so the shift is constant in the REDUCE loop
  term = (ds * q.float() - dm) * x[e * in_features + g * 32 + b]
  total = term.reduce(b, arg=Ops.ADD).reduce(g, arg=Ops.ADD)
  return out[e, row].store(total.cast(out.dtype)).end(o).sink(
    arg=KernelInfo(name=f"q5k_moe_gemv_{out_features}_{in_features}_{n_experts}", opts_to_apply=()))

def vulkan_q5k_expert_linear(qweight: Tensor, sel: Tensor, x: Tensor, in_features: int, out_features: int, n_experts: int) -> Tensor:
  # decode-only (B*T==1) down-expert GEMV over Q5_K bytes: x (B, T, k, in) per-expert rows, sel (B, T, k) int32
  # -> out (B, T, k, out_features); x row e feeds expert slot e (the model always passes the k-row activation)
  k = sel.shape[-1]
  out = Tensor.empty(k, out_features, dtype=x.dtype, device=x.device)
  fxn = functools.partial(q5k_moe_gemv_kernel, out_features=out_features, in_features=in_features, n_experts=n_experts)
  res = Tensor.custom_kernel(out, qweight, x.reshape(k * in_features), sel, fxn=fxn)[0]
  return res.reshape(*x.shape[:-2], k, out_features)
