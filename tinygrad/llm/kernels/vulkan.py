import functools
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import getenv, prod
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
_Q4K_BLOCK_SIZE, _Q4K_BLOCK_BYTES = 256, 144
_Q5K_BLOCK_SIZE, _Q5K_BLOCK_BYTES = 256, 176
_Q6K_BLOCK_SIZE, _Q6K_BLOCK_BYTES = 256, 210

@functools.cache
def vulkan_q4k_enabled(device) -> bool:
  # VULKAN_Q4K overrides the legacy VULKAN_QUANT; both default ON
  return str(device).startswith("VULKAN") and getenv("VULKAN_Q4K", getenv("VULKAN_QUANT", 1)) != 0

@functools.cache
def vulkan_q5k_enabled(device) -> bool:
  # default ON: 4-way A/B on the APU showed a net +11-12% decode speedup with Q5_K down experts fused
  return str(device).startswith("VULKAN") and getenv("VULKAN_Q5K", 1) != 0

@functools.cache
def vulkan_q6k_enabled(device) -> bool:
  return str(device).startswith("VULKAN") and getenv("VULKAN_Q6K", 1) != 0

@functools.cache
def vulkan_ssmab_enabled(device) -> bool:
  # default ON: decode A/B on the APU shows ~+2-4% (30 SSM blocks: 2 byte-wise f32 gemvs + epilogues -> 1 f16 kernel)
  return str(device).startswith("VULKAN") and getenv("VULKAN_SSMAB", 1) != 0

@functools.cache
def vulkan_quant_supported(device) -> bool:
  return vulkan_q4k_enabled(device) or vulkan_q5k_enabled(device) or vulkan_q6k_enabled(device)

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

def find_q6k_bytes(weight: Tensor) -> Tensor|None:
  return _find_quant_bytes(weight, GGML_Q6_K, _Q6K_BLOCK_SIZE, _Q6K_BLOCK_BYTES)

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

@functools.cache
def q6k_moe_gemv_kernel(out:UOp, W8:UOp, x:UOp, sel:UOp, out_features:int, in_features:int, n_experts:int) -> UOp:
  # batched gather GEMV over the k active down experts, Q6_K dequant (ggml_type 14), 210-byte block of 256 elems.
  # Layout (matches ggml_data_to_tensor): element e, h=e//128, f=e%128:
  #   xl = low nibble of byte h*64+f if f<64 else high nibble of byte h*64+f-64
  #   xh = 2-bit field f//32 of byte 128+h*32+f%32;  value = d * (xl+16*xh-32) * scales[e//16] (f16 d at 208:210)
  # thread g owns elems {k : k%32 in {g, g+16}}; REDUCE b over (256-elem block, half, quartet):
  # cb = b//4, h = (b//2)%2, q = b%2, i = g+16*q; 4 elems per step e_j = cb*256+h*128+i+32*j
  # read 2 xl bytes + 1 xh byte + 4 scales (all bit shifts are constant)
  k = sel.shape[-1]
  row_bytes = (in_features // _Q6K_BLOCK_SIZE) * _Q6K_BLOCK_BYTES
  expert_bytes = out_features * row_bytes
  o = UOp.range(k * out_features, 0, AxisType.GLOBAL)
  e, row = o // out_features, o % out_features
  expert = sel[0, 0, e]                # one int32 read per work item, constant inside the reduce loop
  bbase = expert * expert_bytes + row * row_bytes
  g = UOp.range(16, 1, AxisType.LOCAL)          # 16 threads x 32 elems/thread = 512
  b = UOp.range(8, 2, AxisType.REDUCE)          # (block, half, quartet)
  cb, h, q = b // 4, (b // 2) % 2, b % 2
  i = g + 16 * q
  wbase = bbase + cb * _Q6K_BLOCK_BYTES
  d = ((W8[wbase + 208].cast(dtypes.uint16) | W8[wbase + 209].cast(dtypes.uint16).lshift(8))).bitcast(dtypes.float16).float()
  xl1 = W8[wbase + h * 64 + i]            # elems i (low) and i+64 (high)
  xl2 = W8[wbase + h * 64 + 32 + i]       # elems i+32 (low) and i+96 (high)
  xh1 = W8[wbase + 128 + h * 32 + i]      # fields 0..3 of elems i, i+32, i+64, i+96
  base = e * in_features + cb * _Q6K_BLOCK_SIZE + h * 128 + i
  def term(j, qx, xh_shifted):
    # match the dequant expression: d * (q - 32) * scales, all in f32
    w = d * ((qx + 16 * xh_shifted).cast(dtypes.int32) - 32).cast(dtypes.float32)
    return (w * W8[wbase + 192 + 8 * h + q + 2 * j].bitcast(dtypes.int8).float()) * x[base + 32 * j]
  total = (term(0, xl1 & 15, xh1 & 3) + term(1, xl2 & 15, (xh1 >> 2) & 3)
          + term(2, (xl1 >> 4) & 15, (xh1 >> 4) & 3) + term(3, (xl2 >> 4) & 15, (xh1 >> 6) & 3))
  total = total.reduce(b, arg=Ops.ADD).reduce(g, arg=Ops.ADD)
  return out[e, row].store(total.cast(out.dtype)).end(o).sink(
    arg=KernelInfo(name=f"q6k_moe_gemv_{out_features}_{in_features}_{n_experts}", opts_to_apply=()))

def vulkan_q6k_expert_linear(qweight: Tensor, sel: Tensor, x: Tensor, in_features: int, out_features: int, n_experts: int) -> Tensor:
  # decode-only (B*T==1) down-expert GEMV over Q6_K bytes: x (B, T, k, in) per-expert rows, sel (B, T, k) int32
  # -> out (B, T, k, out_features); x row e feeds expert slot e (the model always passes the k-row activation)
  k = sel.shape[-1]
  out = Tensor.empty(k, out_features, dtype=x.dtype, device=x.device)
  fxn = functools.partial(q6k_moe_gemv_kernel, out_features=out_features, in_features=in_features, n_experts=n_experts)
  res = Tensor.custom_kernel(out, qweight, x.reshape(k * in_features), sel, fxn=fxn)[0]
  return res.reshape(*x.shape[:-2], k, out_features)

@functools.cache
def ssmab_kernel(out_a:UOp, out_b:UOp, x:UOp, wa:UOp, wb:UOp, bias:UOp, a:UOp, n_heads:int, dim:int) -> UOp:
  # fused SSM alpha/beta GEMV pair, decode (T=1): 2 kernels -> 1. All inputs are f16 buffers (post-attn_norm
  # x, realized HALF weights, ssm_a, ssm_dt.bias); product f16, accumulation f32, matmul output rounded to
  # f16 -- bit-matching the generic f16 matmul. Epilogues: alpha: +bias -> softplus -> *a in f32; beta:
  # sigmoid in f16. The two dot products share one local reduction over a 2-element stack: two separate
  # reduce(t) reductions would bufferize to the same shared array and the second scan would read the first's data
  t = UOp.range(16, 1, AxisType.LOCAL)
  r = UOp.range(128, 2, AxisType.REDUCE)
  o = UOp.range(n_heads, 0, AxisType.GLOBAL)
  k = t * 128 + r
  xh = x[k]
  pair = UOp.stack(xh * wa[o * dim + k], xh * wb[o * dim + k])
  ab = pair.cast(dtypes.float32).reduce(r, arg=Ops.ADD).reduce(t, arg=Ops.ADD)
  pa, pb = ab.index(0), ab.index(1)
  log_alpha = (pa.cast(dtypes.float16).float() + bias[o].float()).softplus() * a[o].float()
  beta = pb.cast(dtypes.float16).sigmoid()
  return UOp.group(out_a[o].store(log_alpha), out_b[o].store(beta)).end(o).sink(
    arg=KernelInfo(name=f"ssmab_{n_heads}_{dim}", opts_to_apply=()))

def vulkan_ssmab_alpha_beta(x:Tensor, alpha_w:Tensor, beta_w:Tensor, bias:Tensor, a:Tensor, n_heads:int) -> tuple[Tensor, Tensor]|None:
  # decode-only (T=1) fused SSM alpha/beta: x (B, 1, dim) f16 (post attn_norm); alpha_w/beta_w (n_heads, dim)
  # and bias/a (n_heads,) f16 (pre-realized so the kernel reads them directly, no per-step cast) ->
  # (log_alpha (B, 1, n_heads, 1) f32, beta (B, 1, n_heads) f16 with sigmoid applied)
  dim = x.shape[-1]
  if not isinstance(dim, int) or dim // 128 * 128 != dim or not isinstance(n_heads, int): return None
  if x.dtype is not dtypes.half or alpha_w.dtype is not dtypes.half or beta_w.dtype is not dtypes.half: return None
  out_a = Tensor.empty(n_heads, dtype=dtypes.float32, device=x.device)
  out_b = Tensor.empty(n_heads, dtype=dtypes.float16, device=x.device)
  fxn = functools.partial(ssmab_kernel, n_heads=n_heads, dim=dim)
  res = Tensor.custom_kernel(out_a, out_b, x.reshape(dim), alpha_w.reshape(-1), beta_w.reshape(-1),
                             bias.reshape(-1), a.reshape(-1), fxn=fxn)
  B, T, _ = x.shape
  return res[0].reshape(B, T, n_heads, 1), res[1].reshape(B, T, n_heads)

@functools.cache
def moe_selprob_kernel(out_sel:UOp, out_probs:UOp, rank:UOp, logits:UOp, E:int, K:int) -> UOp:
  # fused MoE router tail, decode: topk-sel + gather + softmax in one kernel (4 kernels -> 1). It replaces
  # the model's sel-scatter, gather, softmax-max and softmax-expsum; the gemv and rank stay the model's own
  # kernels (their exact reduction schedules are what bit-exactness is measured against, so they are not
  # reproduced here). rank (E,) int32 is the pairwise_topk rank (rank[e] = #{j: logits[e]>logits[j]} +
  # ties where e<j); logits (E,) f32 are the raw router gemv outputs. One work item re-derives all K sel/s
  # (256-iter scans, L2-hot global reads only, no local memory or barriers) and stores them all. sel[m] is
  # the unique index of rank E-K+m (MAX with a -1 sentinel, so the sentinel never wins). s[m] =
  # logits[sel[m]] via a zero-masked ADD (one exact value + E-1 zeros). The softmax is
  # exp2((s + max*-1)*1.4426950408889634) with a sequential 0..K-1 sum and e*(1/ssum), matching the
  # generic softmax kernels bit-for-bit.
  m = UOp.range(1, 0, AxisType.GLOBAL)
  sels, ss, ax = [], [], 1
  for mp in range(K):
    u1, u2 = UOp.range(E, ax, AxisType.REDUCE), UOp.range(E, ax+1, AxisType.REDUCE)
    ax += 2
    sels.append(rank[u1].eq(E - K + mp).where(u1.cast(dtypes.int32), -1).reduce(u1, arg=Ops.MAX))
    ss.append(rank[u2].eq(E - K + mp).where(logits[u2], 0.0).reduce(u2, arg=Ops.ADD))
  mx = ss[0]
  for s in ss[1:]: mx = mx.maximum(s)
  es = [((s + mx * -1.0) * 1.4426950408889634).exp2() for s in ss]
  ssum = es[0]
  for e in es[1:]: ssum = ssum + e
  recip = ssum.reciprocal()
  stores = [out_sel[mp].store(sels[mp]) for mp in range(K)] + \
           [out_probs[mp].store(es[mp] * recip) for mp in range(K)]
  return UOp.group(*stores).end(m).sink(
    arg=KernelInfo(name=f"moe_selprob_{E}_{K}", opts_to_apply=()))

def vulkan_moe_selprob(rank:Tensor, logits:Tensor, n_experts:int, k:int) -> tuple[Tensor, Tensor]|None:
  # decode-only (B*T==1) MoE router tail: rank (B, 1, n_experts) int32 + logits (B, 1, n_experts) f32
  # -> (sel (B, 1, k) int32, probs (B, 1, k) f32 = softmax of the top-k gathered logits). The caller must
  # have gating==SOFTMAX_WEIGHT, no selection bias, and normalize_topk False (the raw-logits fast path).
  if not isinstance(n_experts, int) or n_experts <= 0 or not isinstance(k, int) or k <= 0: return None
  if rank.shape != logits.shape or rank.shape[-1] != n_experts or prod(rank.shape[:-1]) != 1: return None
  if rank.dtype is not dtypes.int32 or logits.dtype is not dtypes.float: return None
  B, T = rank.shape[:-1]
  sel_out = Tensor.empty(k, dtype=dtypes.int32, device=rank.device)
  probs_out = Tensor.empty(k, dtype=dtypes.float32, device=rank.device)
  fxn = functools.partial(moe_selprob_kernel, E=n_experts, K=k)
  res = Tensor.custom_kernel(sel_out, probs_out, rank.reshape(n_experts), logits.reshape(n_experts), fxn=fxn)
  return res[0].reshape(B, T, k), res[1].reshape(B, T, k)

@functools.cache
def vulkan_router_enabled(device) -> bool:
  return str(device).startswith("VULKAN") and getenv("VULKAN_ROUTER", 1) != 0

@functools.cache
def vulkan_ssmconv_enabled(device) -> bool:
  return str(device).startswith("VULKAN") and getenv("VULKAN_SSMCONV", 1) != 0

@functools.cache
def ssmconv_co_kernel(out_co:UOp, state:UOp, qkv:UOp, w:UOp, sp:UOp, n_htot:int, head:int, ch:int) -> UOp:
  # fused SSM conv1d part 1, decode (T=1): the generic path runs ~6 kernels per block (window zero-fill,
  # conv_state->window, qkv gemv->window, window->conv_state, q rstd, k rstd) plus the v conv dot inside the
  # scan mega-kernel. This part replaces the conv dot: per-channel dot + silu. Work items (h, r):
  # j = h*head + r is the channel. sp is the bound start_pos variable (ALU PARAM form in the body, the
  # bound value is a call arg rebound per replay): sp==0 resets the conv state (the model's initial flag).
  # The dot is raw f32: conv_state values are always f16-representable (zero-init, exact f32 row copies,
  # and f16 qkv outputs), so the baseline kernel's f16 window-load rounding is a no-op on them, and the
  # f16 weights are exact in f32. (The round-trip cast itself is driver-dependent here: Mesa folds
  # f32->f16->f32 in some opt paths, so it must not carry semantics.)
  # No reduce axis here: a per-head reduce inside this 8192-work-item grid makes every work item run a
  # 128-iteration latency-bound global-load loop, ~2ms/block on this GPU. The denominator is a separate
  # small kernel (ssmconv_den_kernel) shaped exactly like the generic rstd (one work item per head).
  h = UOp.range(n_htot, 0, AxisType.GLOBAL)
  r = UOp.range(head, 1, AxisType.GLOBAL)
  ne0 = sp.ne(0)  # (start_pos != 0): keep state; 0 when start_pos==0 (reset)
  j = h * head + r
  s0, s1, s2 = ne0.where(state[j], 0.0), ne0.where(state[ch + j], 0.0), ne0.where(state[2 * ch + j], 0.0)
  q = qkv[j]
  dot = (s0 * w[4*j].float() + s1 * w[4*j+1].float() + s2 * w[4*j+2].float() + q.float() * w[4*j+3].float())
  return UOp.group(out_co[j].store(dot.silu())).end(h, r).sink(
    arg=KernelInfo(name=f"ssmconv_co_{ch}_{head}", opts_to_apply=()))

@functools.cache
def ssmconv_den_kernel(out_den:UOp, co:UOp, n_dh:int, head:int) -> UOp:
  # fused SSM conv1d part 1b: the q/k normalize denominators (normalize(p=2, dim=-1, eps=1e-6) of the
  # post-silu conv output), one work item per q/k head with the sequential head-dim reduce. Same shape as
  # the generic rstd kernel (small60 dump: one work item per head, sequential reduce, no opts), so the
  # result is bit-exact with the generic path; the v heads are never normalized, so only 2*n_k_heads
  # denominators exist.
  h = UOp.range(n_dh, 0, AxisType.GLOBAL)
  rr = UOp.range(head, 1, AxisType.REDUCE)
  v = co[h * head + rr]
  return UOp.group(out_den[h].store((v * v).reduce(rr, arg=Ops.ADD).sqrt().maximum(1e-6))).end(h).sink(
    arg=KernelInfo(name=f"ssmconv_den_{n_dh}_{head}", opts_to_apply=()))

@functools.cache
def ssmconv_shift_kernel(state:UOp, qkv:UOp, d1:UOp, d2:UOp, sp:UOp, n_htot:int, head:int, ch:int) -> UOp:
  # fused SSM conv1d part 2, decode (T=1): the in-place conv_state shift (row i <- row i+1, last row <-
  # qkv, rows 0/1 zeroed when start_pos==0), replacing the window zero-fill + conv_state->window +
  # window->conv_state kernels. Work item j owns channel j across all three rows (two reads, three
  # writes, no cross-item aliasing); the shift stores raw (unrounded) values, like the generic window
  # copy. The caller passes the part-1 outputs as d1/d2 (unused here) so the graph orders this after the
  # part-1 state reads: the in-place write is invisible to the scheduler, so without the edge it could
  # land before part 1's reads.
  h = UOp.range(n_htot, 0, AxisType.GLOBAL)
  r = UOp.range(head, 1, AxisType.GLOBAL)
  ne0 = sp.ne(0)
  j = h * head + r
  s1, s2 = ne0.where(state[ch + j], 0.0), ne0.where(state[2 * ch + j], 0.0)
  q = qkv[j]
  return UOp.group(
    state[j].store(s1),
    state[ch + j].store(s2),
    state[2*ch + j].store(q.float()),
  ).end(h, r).sink(arg=KernelInfo(name=f"ssmconv_shift_{ch}_{head}", opts_to_apply=()))

def vulkan_ssmconv1d(qkv:Tensor, conv_state:Tensor, w16:Tensor, sp:UOp,
                     n_k_heads:int, n_v_heads:int, head_k_dim:int, conv_channels:int) -> tuple[Tensor, Tensor, UOp]|None:
  # decode-only (T=1) fused SSM conv1d: qkv (B, 1, conv_channels) f16 (the raw attn_qkv output), conv_state
  # (1, 3, conv_channels) f32 (persistent, updated in place by the shift kernel), w16 (conv_channels, 4)
  # f16 (pre-realized ssm_conv1d.weight), sp the bound start_pos variable UOp -> (conv_out
  # (conv_channels,) f32 after silu, den (2*n_k_heads,) f32 q/k normalize denominators).
  # sp is a call arg (the framework's bound-Variable mechanism: its value feeds the ALU PARAM in the body
  # through var_vals, rebound per replay); a bound var inlined in the body would leak its STORE node into
  # the kernel sink and fail codegen.
  if not (isinstance(sp, UOp) and sp.is_bound_var): return None
  if not isinstance(n_k_heads, int) or n_k_heads <= 0 or not isinstance(n_v_heads, int) or n_v_heads <= 0: return None
  if not isinstance(head_k_dim, int) or head_k_dim <= 0 or not isinstance(conv_channels, int) or conv_channels <= 0: return None
  if qkv.dtype is not dtypes.half or w16.dtype is not dtypes.half or conv_state.dtype is not dtypes.float: return None
  if conv_state.numel() != 3 * conv_channels or w16.numel() != 4 * conv_channels: return None
  n_htot = 2 * n_k_heads + n_v_heads
  n_dh = 2 * n_k_heads  # only the q/k heads are normalized
  out_co = Tensor.empty(conv_channels, dtype=dtypes.float32, device=qkv.device)
  out_den = Tensor.empty(n_dh, dtype=dtypes.float32, device=qkv.device)
  st, q, w = conv_state.reshape(3 * conv_channels), qkv.reshape(conv_channels), w16.reshape(conv_channels * 4)
  sp_ph = sp.src[0].replace(op=Ops.PARAM)  # the kernel-side form of the Variable (slot -1, name kept)
  co_ph = tuple(UOp.placeholder_like(s, slot=i) for i, s in enumerate((out_co.uop, st.uop, q.uop, w.uop)))
  co_call = ssmconv_co_kernel(*co_ph, sp_ph, n_htot=n_htot, head=head_k_dim, ch=conv_channels).call(
    out_co.uop, st.uop, q.uop, w.uop, sp)
  co_out = out_co.uop.after(co_call)
  # the co output is passed as its after-call edge (not the bare buffer): a bare-buffer call src carries no
  # cross-call ordering into the captured graph, so the den kernel could replay before the co kernel wrote
  # out_co (eager schedules hide this, the JIT replay order does not)
  den_call = ssmconv_den_kernel(UOp.placeholder_like(out_den.uop, slot=0), UOp.placeholder_like(out_co.uop, slot=1),
                                n_dh=n_dh, head=head_k_dim).call(out_den.uop, co_out)
  den_out = out_den.uop.after(den_call)
  # the in-place shift write is invisible to the scheduler (conv_state is both an in and an out of this
  # call), so the call only runs if something in the graph depends on it: the caller must carry the
  # returned call uop into the step (the model orders the recurrent-state read after it, like the
  # generic path's conv_state_store). The co reads and this call's conv_state read/write also order the
  # shift after the part-1 state reads.
  sh_uops = (st.uop, q.uop, co_out, den_out)
  shift_call = ssmconv_shift_kernel(*[UOp.placeholder_like(s, slot=i) for i, s in enumerate(sh_uops)], sp_ph,
                                    n_htot=n_htot, head=head_k_dim, ch=conv_channels).call(*(sh_uops + (sp,)))
  return Tensor(co_out), Tensor(den_out), shift_call

@functools.cache
def ssmscan_a_kernel(vmin:UOp, state:UOp, qkv:UOp, den:UOp, log_alpha:UOp, sp:UOp,
                     H:int, HK:int, K:int, V:int) -> UOp:
  # SSM delta-rule scan, decode (T=1), part 1 of 3: the per-row "inner" reduction. One work item per
  # (h, v) row (H*V total), a sequential K-fold; op tree mirrors generic kernel 11 exactly:
  # inner = REDUCE_k (state0*alpha)*(kraw*1/den_k); vmin = v + inner*-1.0, all f32. state0 zeroes the
  # row when sp==0 (reset).
  h = UOp.range(H, 0, AxisType.GLOBAL)
  v = UOp.range(V, 1, AxisType.GLOBAL)
  k = UOp.range(K, 2, AxisType.REDUCE)
  st = sp.ne(0).where(state[(h * V + v) * K + k], 0.0)
  alpha = (log_alpha[h] * 1.4426950408889634).exp2()
  kn = qkv[(h % HK) * K + k + HK * K] * den[(h % HK) + HK].reciprocal()
  inner = (st * alpha * kn).reduce(k, arg=Ops.ADD)
  return UOp.group(vmin[h * V + v].store(qkv[h * V + v + 2 * HK * K] + inner * -1.0)).end(h, v).sink(
    arg=KernelInfo(name=f"ssmscan_a_{H}_{V}_{K}", opts_to_apply=()))

@functools.cache
def ssmscan_b_kernel(state:UOp, vmin:UOp, qkv:UOp, den:UOp, log_alpha:UOp, beta:UOp, sp:UOp,
                     H:int, HK:int, K:int, V:int) -> UOp:
  # part 2 of 3: the recurrent-state update, IN PLACE (this is what removes the generic path's two
  # full-state copy kernels). Element-wise over (h, v, k); op tree mirrors generic kernel 12 exactly:
  # state' = state0*alpha + (vmin*beta_f16->f32)*(kraw*1/den_k).
  h = UOp.range(H, 0, AxisType.GLOBAL)
  v = UOp.range(V, 1, AxisType.GLOBAL)
  k = UOp.range(K, 2, AxisType.GLOBAL)
  idx = h * V * K + v * K + k
  st = sp.ne(0).where(state[idx], 0.0)
  alpha = (log_alpha[h] * 1.4426950408889634).exp2()
  delta = vmin[h * V + v] * beta[h].cast(dtypes.float32)
  kn = qkv[(h % HK) * K + k + HK * K] * den[(h % HK) + HK].reciprocal()
  return UOp.group(state[idx].store(st * alpha + delta * kn)).end(h, v, k).sink(
    arg=KernelInfo(name=f"ssmscan_b_{H}_{V}_{K}", opts_to_apply=()))

@functools.cache
def ssmscan_c_kernel(core:UOp, state:UOp, qkv:UOp, den:UOp, H:int, HK:int, K:int, V:int) -> UOp:
  # part 3 of 3: the per-row output dot. One work item per (h, v) row, a sequential K-fold over the
  # (already updated) state; op tree mirrors generic kernel 13 exactly:
  # core = REDUCE_k ((qraw*1/den_q)*state')*K**-0.5.
  h = UOp.range(H, 0, AxisType.GLOBAL)
  v = UOp.range(V, 1, AxisType.GLOBAL)
  k = UOp.range(K, 2, AxisType.REDUCE)
  qn = qkv[(h % HK) * K + k] * den[h % HK].reciprocal()
  return UOp.group(core[h * V + v].store(((qn * state[h * V * K + v * K + k]) * (K ** -0.5)).reduce(k, arg=Ops.ADD))).end(
    h, v).sink(arg=KernelInfo(name=f"ssmscan_c_{H}_{V}_{K}", opts_to_apply=()))

def vulkan_ssm_scan(conv_out:Tensor, den:Tensor, log_alpha:Tensor, beta:Tensor, state:Tensor, sp:UOp,
                    n_k_heads:int, n_v_heads:int, head_k_dim:int, head_v_dim:int) -> tuple[Tensor, UOp]|None:
  # decode-only (T=1) fused SSM delta-rule scan: conv_out (1, 1, 2*HK*K + H*V) f32 (the post-silu conv
  # output [q | k | v] from the ssmconv fusion), den (2*HK,) f32 (q den 0..HK-1, k den HK..2HK-1),
  # log_alpha (1, 1, H, 1) f32 and beta (1, 1, H) f16 (the ssmab outputs), state (1, H, V, K) f32
  # (recurrent_state, updated IN PLACE by part 2), sp the bound start_pos variable (0 resets the state)
  # -> core (H*V,) f32 (the per-head output dot). Three kernels replace the generic 5 (the inner reduce,
  # the state update -- now in-place, so the two full-state copies are gone -- and the out dot). The
  # state arg should carry the .after(conv_state_shift_call) edge so the invisible in-place conv_state
  # write stays in the graph; the returned call uop must be carried into the step so the caller reads the
  # state after the in-place update (like ssmconv's shift_call).
  if not (isinstance(sp, UOp) and sp.is_bound_var): return None
  if not all(isinstance(z, int) and z > 0 for z in (n_k_heads, n_v_heads, head_k_dim, head_v_dim)): return None
  H, HK, K, V = n_v_heads, n_k_heads, head_k_dim, head_v_dim
  if H % HK: return None
  if state.dtype is not dtypes.float or conv_out.dtype is not dtypes.float: return None
  # beta arrives f32 (the model .float()s it before this call); the kernel casts to f32 either way
  if den.dtype is not dtypes.float or log_alpha.dtype is not dtypes.float or beta.dtype not in (dtypes.half, dtypes.float): return None
  if state.shape != (1, H, V, K) or conv_out.shape != (1, 1, 2 * HK * K + H * V): return None
  # den/log_alpha/beta are consumed per-head (flat H/2HK vectors): accept any layout with the right numel
  # (the model transposes beta to (1, H, T) before this call)
  if den.numel() != 2 * HK or log_alpha.numel() != H or beta.numel() != H: return None
  sp_ph = sp.src[0].replace(op=Ops.PARAM)
  # flat views: the kernels index 1-D placeholders (a scalar index into a multi-dim placeholder only
  # indexes dim 0); the call args are the same flat views (a RESHAPE on the buffer, like ssmconv's). The
  # state call arg carries the conv-shift after-edge; the c kernel's state arg additionally carries the
  # b-call after-edge so it reads the in-place-updated state (a bare buffer arg carries no cross-call
  # ordering into the captured graph -- the JIT replay order would not).
  vmin = Tensor.empty(H * V, dtype=dtypes.float32, device=state.device)
  core = Tensor.empty(H * V, dtype=dtypes.float32, device=state.device)
  st_f, co_f, den_f, la_f, be_f = state.reshape(H * V * K), conv_out.reshape(2 * HK * K + H * V), den.reshape(2 * HK), \
    log_alpha.reshape(H), beta.reshape(H)
  a_call = ssmscan_a_kernel(UOp.placeholder_like(vmin.uop, slot=0), UOp.placeholder_like(st_f.uop, slot=1),
                            UOp.placeholder_like(co_f.uop, slot=2), UOp.placeholder_like(den_f.uop, slot=3),
                            UOp.placeholder_like(la_f.uop, slot=4), sp_ph, H=H, HK=HK, K=K, V=V).call(
    vmin.uop, st_f.uop, co_f.uop, den_f.uop, la_f.uop, sp)
  # cross-call ordering is carried only by after-edges (a bare-buffer call src carries none into the
  # captured graph): the b kernel's vmin read and the c kernel's state read are the a/b calls' after-edges
  vmin_after_a = vmin.uop.after(a_call)
  b_call = ssmscan_b_kernel(UOp.placeholder_like(st_f.uop, slot=0), UOp.placeholder_like(vmin.uop, slot=1),
                            UOp.placeholder_like(co_f.uop, slot=2), UOp.placeholder_like(den_f.uop, slot=3),
                            UOp.placeholder_like(la_f.uop, slot=4), UOp.placeholder_like(be_f.uop, slot=5), sp_ph,
                            H=H, HK=HK, K=K, V=V).call(
    st_f.uop, vmin_after_a, co_f.uop, den_f.uop, la_f.uop, be_f.uop, sp)
  c_call = ssmscan_c_kernel(UOp.placeholder_like(core.uop, slot=0), UOp.placeholder_like(st_f.uop, slot=1),
                            UOp.placeholder_like(co_f.uop, slot=2), UOp.placeholder_like(den_f.uop, slot=3),
                            H=H, HK=HK, K=K, V=V).call(core.uop, st_f.uop.after(b_call), co_f.uop, den_f.uop)
  return Tensor(core.uop.after(c_call)), c_call

@functools.cache
def vulkan_ssm_scan_enabled(device) -> bool:
  return str(device).startswith("VULKAN") and getenv("VULKAN_SCAN", 0) != 0
