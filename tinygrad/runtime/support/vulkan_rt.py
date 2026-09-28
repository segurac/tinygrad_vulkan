"""Minimal pure-ctypes Vulkan 1.2 compute runtime (loader: libvulkan.so.1).

Struct layouts match Vulkan-Headers v1.4.305 (extra/vulkan/include/vulkan/vulkan_core.h).
Only what a compute device needs: one queue family, buffers (mapped or
device-local + staging), a ring of (command buffer, fence) pairs so submits can be
async and pipelined, compute pipelines, storage/uniform buffer descriptors.
Kernels accumulate in the pending command buffer; submit() ends+submits it, so work is
launched in batches (one submit per batch, not per kernel).
"""
import ctypes as C
import glob
import os
import struct
import time
from ctypes import c_uint32, c_uint64, c_uint8, c_int32, c_size_t, c_void_p, c_char_p, c_float

def _find_loader() -> str:
  cands = [os.environ.get("VULKAN_LOADER", "")]
  cands += glob.glob("/usr/lib/*/libvulkan.so.1") + glob.glob("/usr/lib/libvulkan.so.1") + \
           glob.glob("/usr/local/lib/*/libvulkan.so.1") + glob.glob("/usr/local/lib/libvulkan.so.1")
  for c in cands:
    if c and os.path.exists(c): return c
  raise RuntimeError(f"cannot find the Vulkan loader (set VULKAN_LOADER); tried {cands}")

LOADER = _find_loader()

def VK_MAKE_VERSION(major, minor, patch):
    return (major << 22) | (minor << 12) | patch

VK_SUCCESS = 0
VK_TIMEOUT = -4
VK_ERROR_OUT_OF_HOST_MEMORY = -8
VK_ERROR_OUT_OF_DEVICE_MEMORY = -9
VK_ERROR_INITIALIZATION_FAILED = -10
VK_ERROR_DEVICE_LOST = -11
VK_ERROR_MEMORY_MAP_FAILED = -12
VK_ERROR_LAYER_NOT_PRESENT = -13
VK_ERROR_EXTENSION_NOT_PRESENT = -14
VK_ERROR_INCOMPATIBLE_DRIVER = -15
VK_ERROR_FEATURE_NOT_PRESENT = -16
VK_ERROR_FORMAT_NOT_SUPPORTED = -18
VK_ERROR_UNKNOWN = -1000000001

_RESULTS = {
    0: "VK_SUCCESS", -2: "VK_NOT_READY", -4: "VK_TIMEOUT", -8: "VK_ERROR_OUT_OF_HOST_MEMORY",
    -9: "VK_ERROR_OUT_OF_DEVICE_MEMORY", -10: "VK_ERROR_INITIALIZATION_FAILED",
    -11: "VK_ERROR_DEVICE_LOST", -12: "VK_ERROR_MEMORY_MAP_FAILED",
    -13: "VK_ERROR_LAYER_NOT_PRESENT", -14: "VK_ERROR_EXTENSION_NOT_PRESENT",
    -15: "VK_ERROR_INCOMPATIBLE_DRIVER", -16: "VK_ERROR_FEATURE_NOT_PRESENT",
    -18: "VK_ERROR_FORMAT_NOT_SUPPORTED", -1000000001: "VK_ERROR_UNKNOWN",
}

# in-flight submits allowed: the (command buffer, fence) ring depth. submit() only blocks
# (backpressure) when it wraps around to a slot whose fence has not signaled yet.
RING = 8
# VK_BATCH_SEM: (command buffer, binary semaphore) arena size. One CBT per kernel/copy in a
# step (a 1290-kernel decode step needs ~1291 CBTs), chained in one submit; 4096 leaves headroom.
POOL = 4096
# VK_REPLAY: template command buffer arena size, allocated from the same pool after the
# main arena. Template CBTs are recorded once per (program, bufs, nvals, grid) signature and
# re-submitted (never re-begun) on every later execution of it, so the recording cursor never
# touches them. 8192 covers ~1300 signatures with up to 6 rotating templates each.
TPL_POOL = 8192
# flush points for the pending (unsubmitted) command buffer, so the fence ring keeps
# cycling and the CPU can't outrun the GPU by an unbounded count. CB_FLUSH_KERNELS is the
# normal point (a kernel's bind/bdesc/dispatch must never be split across cbs: a dispatch
# recorded in a fresh cb has no pipeline bound and the driver crashes). CB_FLUSH_MAX is a
# command-count backstop for copy-heavy runs; it only fires between kernels.
CB_FLUSH_KERNELS = 128
CB_FLUSH_MAX = 512

VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 1
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 2
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 4
VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 1
VK_BUFFER_USAGE_TRANSFER_DST_BIT = 2
VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 4
VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 8
VK_BUFFER_USAGE_ALL = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | \
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
VK_SHARING_MODE_EXCLUSIVE = 0
VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0
VK_PIPELINE_BIND_POINT_COMPUTE = 1
VK_SHADER_STAGE_COMPUTE_BIT = 0x20
VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x800
VK_ACCESS_SHADER_READ_BIT = 0x2000
VK_ACCESS_SHADER_WRITE_BIT = 0x4000
VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT = 0x10
VK_QUERY_TYPE_TIMESTAMP = 2
VK_QUERY_RESULT_64_BIT_BIT = 2
VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6
VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC = 8
VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC = 9

# sType values (VkStructureType, vulkan_core.h v1.4.305)
ST_APPLICATION_INFO = 0
ST_INSTANCE_CREATE_INFO = 1
ST_DEVICE_QUEUE_CREATE_INFO = 2
ST_DEVICE_CREATE_INFO = 3
ST_SUBMIT_INFO = 4
ST_MEMORY_ALLOCATE_INFO = 5
ST_SEMAPHORE_CREATE_INFO = 9
ST_FENCE_CREATE_INFO = 8
ST_BUFFER_CREATE_INFO = 12
ST_SHADER_MODULE_CREATE_INFO = 16
ST_PIPELINE_SHADER_STAGE_CREATE_INFO = 18
ST_COMPUTE_PIPELINE_CREATE_INFO = 29
ST_PIPELINE_LAYOUT_CREATE_INFO = 30
ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32
ST_DESCRIPTOR_POOL_CREATE_INFO = 33
ST_DESCRIPTOR_SET_ALLOCATE_INFO = 34
ST_WRITE_DESCRIPTOR_SET = 35
ST_COMMAND_POOL_CREATE_INFO = 39
ST_COMMAND_BUFFER_ALLOCATE_INFO = 40
ST_COMMAND_BUFFER_BEGIN_INFO = 42
ST_MEMORY_BARRIER = 10
ST_QUERY_POOL_CREATE_INFO = 24

# ---------------------------------------------------------------- structs --

class VkApplicationInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("pApplicationName", c_char_p),
                ("applicationVersion", c_uint32), ("pEngineName", c_char_p),
                ("engineVersion", c_uint32), ("apiVersion", c_uint32)]

class VkInstanceCreateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("pApplicationInfo", c_void_p), ("enabledLayerCount", c_uint32),
                ("ppEnabledLayerNames", c_void_p), ("enabledExtensionCount", c_uint32),
                ("ppEnabledExtensionNames", c_void_p)]

class VkDeviceQueueCreateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("queueFamilyIndex", c_uint32), ("queueCount", c_uint32),
                ("pQueuePriorities", c_void_p)]

class VkDeviceCreateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("queueCreateInfoCount", c_uint32), ("pQueueCreateInfos", c_void_p),
                ("enabledLayerCount", c_uint32), ("ppEnabledLayerNames", c_void_p),
                ("enabledExtensionCount", c_uint32), ("ppEnabledExtensionNames", c_void_p),
                ("pEnabledFeatures", c_void_p)]

class VkBufferCreateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("size", c_uint64), ("usage", c_uint32), ("sharingMode", c_uint32),
                ("queueFamilyIndexCount", c_uint32), ("pQueueFamilyIndices", c_void_p)]

class VkMemoryRequirements(C.Structure):
    _fields_ = [("size", c_uint64), ("alignment", c_uint64), ("memoryTypeBits", c_uint32)]

class VkMemoryAllocateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("allocationSize", c_uint64),
                ("memoryTypeIndex", c_uint32)]

class VkCommandPoolCreateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("queueFamilyIndex", c_uint32)]

class VkCommandBufferAllocateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("commandPool", c_void_p),
                ("level", c_uint32), ("commandBufferCount", c_uint32)]

class VkCommandBufferBeginInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("pInheritanceInfo", c_void_p)]

class VkBufferCopy(C.Structure):
    _fields_ = [("srcOffset", c_uint64), ("dstOffset", c_uint64), ("size", c_uint64)]

class VkMemoryBarrier(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("srcAccessMask", c_uint32),
                ("dstAccessMask", c_uint32)]

class VkQueryPoolCreateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("queryType", c_uint32), ("queryCount", c_uint32), ("pipelineStatisticsCount", c_uint32)]

class VkShaderModuleCreateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("codeSize", c_size_t), ("pCode", c_void_p)]

class VkDescriptorSetLayoutBinding(C.Structure):
    _fields_ = [("binding", c_uint32), ("descriptorType", c_uint32),
                ("descriptorCount", c_uint32), ("stageFlags", c_uint32),
                ("pImmutableSamplers", c_void_p)]

class VkDescriptorSetLayoutCreateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("bindingCount", c_uint32), ("pBindings", c_void_p)]

class VkPipelineLayoutCreateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("setLayoutCount", c_uint32), ("pSetLayouts", c_void_p),
                ("pushConstantRangeCount", c_uint32), ("pPushConstantRanges", c_void_p)]

class VkPipelineShaderStageCreateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("stage", c_uint32), ("module", c_void_p), ("pName", c_char_p),
                ("pSpecializationInfo", c_void_p)]

class VkComputePipelineCreateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("stage", VkPipelineShaderStageCreateInfo), ("layout", c_void_p),
                ("basePipelineHandle", c_void_p), ("basePipelineIndex", c_int32)]

class VkDescriptorPoolSize(C.Structure):
    _fields_ = [("type", c_uint32), ("descriptorCount", c_uint32)]

class VkDescriptorPoolCreateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32),
                ("maxSets", c_uint32), ("poolSizeCount", c_uint32), ("pPoolSizes", c_void_p)]

class VkDescriptorSetAllocateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("descriptorPool", c_void_p),
                ("descriptorSetCount", c_uint32), ("pSetLayouts", c_void_p)]

class VkDescriptorBufferInfo(C.Structure):
    _fields_ = [("buffer", c_void_p), ("offset", c_uint64), ("range", c_uint64)]

class VkWriteDescriptorSet(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("dstSet", c_void_p),
                ("dstBinding", c_uint32), ("dstArrayElement", c_uint32),
                ("descriptorCount", c_uint32), ("descriptorType", c_uint32),
                ("pImageInfo", c_void_p), ("pBufferInfo", c_void_p),
                ("pTexelBufferView", c_void_p)]

class VkSubmitInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("waitSemaphoreCount", c_uint32),
                ("pWaitSemaphores", c_void_p), ("pWaitDstStageMask", c_void_p),
                ("commandBufferCount", c_uint32), ("pCommandBuffers", c_void_p),
                ("signalSemaphoreCount", c_uint32), ("pSignalSemaphores", c_void_p)]

class VkFenceCreateInfo(C.Structure):
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32)]

class VkSemaphoreCreateInfo(C.Structure):
    # default (no flags) creates a binary semaphore: auto-reset on the wait that consumes it,
    # so a semaphore chained through a submit needs no explicit reset between batches
    _fields_ = [("sType", c_uint32), ("pNext", c_void_p), ("flags", c_uint32)]

class VkMemoryType(C.Structure):
    _fields_ = [("propertyFlags", c_uint32), ("heapIndex", c_uint32)]

class VkMemoryHeap(C.Structure):
    _fields_ = [("size", c_uint64), ("flags", c_uint32)]

class VkPhysicalDeviceMemoryProperties(C.Structure):
    _fields_ = [("memoryTypeCount", c_uint32), ("memoryTypes", VkMemoryType * 32),
                ("memoryHeapCount", c_uint32), ("memoryHeaps", VkMemoryHeap * 16)]

# sizeof(VkPhysicalDeviceProperties) == 824 per v1.4.305; only the leading fields
# are read (apiVersion@0, driverVersion@4, vendorID@8, deviceID@12, deviceType@16,
# deviceName@20..276). A fixed-size buffer avoids transcribing VkPhysicalDeviceLimits.
PROPS_SIZE = 824

# ------------------------------------------------------------- entry points --

lib = C.CDLL(LOADER)

def _vk(name, argtypes, restype=c_int32):
    f = getattr(lib, name)
    f.restype = restype
    f.argtypes = argtypes
    return f

_v = [c_void_p]
_N = None  # restype: void-returning entry points (per vulkan_core.h v1.4.305)

vkCreateInstance = _vk("vkCreateInstance", _v + _v + [C.POINTER(c_void_p)])
vkDestroyInstance = _vk("vkDestroyInstance", _v + _v, _N)
vkEnumeratePhysicalDevices = _vk("vkEnumeratePhysicalDevices", _v + [C.POINTER(c_uint32)] + _v)
vkGetPhysicalDeviceProperties = _vk("vkGetPhysicalDeviceProperties", _v + _v, _N)
vkGetPhysicalDeviceMemoryProperties = _vk("vkGetPhysicalDeviceMemoryProperties", _v + _v, _N)
vkCreateDevice = _vk("vkCreateDevice", _v + _v + _v + [C.POINTER(c_void_p)])
vkDestroyDevice = _vk("vkDestroyDevice", _v + _v, _N)
vkGetDeviceQueue = _vk("vkGetDeviceQueue", _v + [c_uint32, c_uint32, C.POINTER(c_void_p)], _N)
vkDeviceWaitIdle = _vk("vkDeviceWaitIdle", _v)
vkCreateBuffer = _vk("vkCreateBuffer", _v + _v + _v + [C.POINTER(c_void_p)])
vkDestroyBuffer = _vk("vkDestroyBuffer", _v + _v + _v, _N)
vkGetBufferMemoryRequirements = _vk("vkGetBufferMemoryRequirements", _v + _v + _v, _N)
vkAllocateMemory = _vk("vkAllocateMemory", _v + _v + _v + [C.POINTER(c_void_p)])
vkFreeMemory = _vk("vkFreeMemory", _v + _v + _v, _N)
vkBindBufferMemory = _vk("vkBindBufferMemory", _v + _v + _v + [c_uint64])
vkMapMemory = _vk("vkMapMemory", _v + _v + [c_uint64, c_uint64, c_uint32, C.POINTER(c_void_p)])
vkUnmapMemory = _vk("vkUnmapMemory", _v + _v, _N)
vkCreateCommandPool = _vk("vkCreateCommandPool", _v + _v + _v + [C.POINTER(c_void_p)])
vkDestroyCommandPool = _vk("vkDestroyCommandPool", _v + _v + _v, _N)
vkAllocateCommandBuffers = _vk("vkAllocateCommandBuffers", _v + _v + [C.POINTER(c_void_p)])
vkFreeCommandBuffers = _vk("vkFreeCommandBuffers", _v + _v + [c_uint32] + _v, _N)
vkBeginCommandBuffer = _vk("vkBeginCommandBuffer", _v + _v)
vkResetCommandBuffer = _vk("vkResetCommandBuffer", _v + [c_uint32])
vkEndCommandBuffer = _vk("vkEndCommandBuffer", _v)
vkCmdCopyBuffer = _vk("vkCmdCopyBuffer", _v + _v + _v + [c_uint32] + _v, _N)
vkCmdPipelineBarrier = _vk("vkCmdPipelineBarrier",
                           _v + [c_uint32, c_uint32, c_uint32] + [c_uint32] + _v + [c_uint32] + _v + [c_uint32] + _v, _N)
vkCmdBindPipeline = _vk("vkCmdBindPipeline", _v + [c_uint32] + _v, _N)
vkCmdBindDescriptorSets = _vk("vkCmdBindDescriptorSets", _v + [c_uint32] + _v + [c_uint32, c_uint32] + _v + [c_uint32] + _v, _N)
vkCmdDispatch = _vk("vkCmdDispatch", _v + [c_uint32, c_uint32, c_uint32], _N)
vkCreateFence = _vk("vkCreateFence", _v + _v + _v + [C.POINTER(c_void_p)])
vkDestroyFence = _vk("vkDestroyFence", _v + _v + _v, _N)
vkResetFences = _vk("vkResetFences", _v + [c_uint32] + _v)
vkCreateSemaphore = _vk("vkCreateSemaphore", _v + _v + _v + [C.POINTER(c_void_p)])
vkDestroySemaphore = _vk("vkDestroySemaphore", _v + _v + _v, _N)
vkQueueSubmit = _vk("vkQueueSubmit", _v + [c_uint32] + 2 * _v)
vkWaitForFences = _vk("vkWaitForFences", _v + [c_uint32] + _v + [c_uint32, c_uint64])
vkCreateShaderModule = _vk("vkCreateShaderModule", _v + _v + _v + [C.POINTER(c_void_p)])
vkDestroyShaderModule = _vk("vkDestroyShaderModule", _v + _v + _v, _N)
vkCreateDescriptorSetLayout = _vk("vkCreateDescriptorSetLayout", _v + _v + _v + [C.POINTER(c_void_p)])
vkDestroyDescriptorSetLayout = _vk("vkDestroyDescriptorSetLayout", _v + _v + _v, _N)
vkCreatePipelineLayout = _vk("vkCreatePipelineLayout", _v + _v + _v + [C.POINTER(c_void_p)])
vkDestroyPipelineLayout = _vk("vkDestroyPipelineLayout", _v + _v + _v, _N)
vkCreateComputePipelines = _vk("vkCreateComputePipelines", _v + _v + [c_uint32] + 2 * _v + _v)
vkDestroyPipeline = _vk("vkDestroyPipeline", _v + _v + _v, _N)
vkFreeDescriptorSets = _vk("vkFreeDescriptorSets", _v + _v + [c_uint32] + _v, _N)
vkCreateDescriptorPool = _vk("vkCreateDescriptorPool", _v + _v + _v + [C.POINTER(c_void_p)])
vkDestroyDescriptorPool = _vk("vkDestroyDescriptorPool", _v + _v + _v, _N)
vkAllocateDescriptorSets = _vk("vkAllocateDescriptorSets", _v + _v + [C.POINTER(c_void_p)])
vkUpdateDescriptorSets = _vk("vkUpdateDescriptorSets", _v + [c_uint32] + 2 * _v, _N)
vkCreateQueryPool = _vk("vkCreateQueryPool", _v + _v + _v + [C.POINTER(c_void_p)])
vkCmdWriteTimestamp = _vk("vkCmdWriteTimestamp", _v + [c_uint32] + _v + [c_uint32], _N)
vkGetQueryPoolResults = _vk("vkGetQueryPoolResults", _v + _v + [c_uint32, c_uint32] + [c_size_t] + _v + [c_size_t] + [c_uint32])

class VkError(RuntimeError):
    def __init__(self, op, result):
        self.op, self.result = op, result
        super().__init__(f"{op} -> {_RESULTS.get(result, 'VK_RESULT_%d' % result)}")

def _check(op, r):
    if r != 0:
        raise VkError(op, r)

def _handle(obj):
    return obj.handle if hasattr(obj, "handle") else obj

def _p(obj):
    # c_void_p struct fields reject typed pointers; cast to an untyped one
    return C.cast(C.byref(obj), c_void_p)

def _cvt(h) -> int | None:
    # a CBT/buffer handle value for set/dict membership (c_void_p is unhashable). mypy's
    # ctypes plugin types c_void_p array elements as int|None, hence the cast round-trip.
    return C.cast(h or 0, c_void_p).value

# --------------------------------------------------------------- wrappers --

class VBuffer:
    def __init__(self, rt, handle, mem, size, host_visible):
        self.rt, self.handle, self.mem = rt, handle, mem
        self.size, self.host_visible = size, host_visible
        self._mapped, self._arr = None, None

    def map(self):
        """Map a host-visible buffer; returns a (c_uint8 * size) view."""
        if not self.host_visible:
            raise RuntimeError("device-local buffer: use a host_visible staging buffer + cmd_copy")
        if self._mapped is None:
            ptr = c_void_p()
            _check("vkMapMemory", vkMapMemory(self.rt.device, self.mem, 0, self.size, 0, C.byref(ptr)))
            self._mapped = ptr
            self._arr = (c_uint8 * self.size).from_address(ptr.value)
        return self._arr

class VShaderModule:
    def __init__(self, rt, handle, words):
        self.rt, self.handle, self._words = rt, handle, words

class VDescriptorSetLayout:
    def __init__(self, rt, layout, pool, desc_set, buffers):
        self.rt, self.layout, self.pool = rt, layout, pool
        self.set, self.buffers = desc_set, buffers

class VPipeline:
    def __init__(self, rt, handle, layout, module):
        self.rt, self.handle, self.layout, self.module = rt, handle, layout, module

# ---------------------------------------------------------------- runtime --

class VkRt:
    def __init__(self):
        self.init()

    def init(self):
        app = VkApplicationInfo(sType=ST_APPLICATION_INFO, pApplicationName=b"vkrt",
                                apiVersion=VK_MAKE_VERSION(1, 2, 0))
        ici = VkInstanceCreateInfo(sType=ST_INSTANCE_CREATE_INFO, pApplicationInfo=_p(app))
        inst = c_void_p()
        _check("vkCreateInstance", vkCreateInstance(C.byref(ici), None, C.byref(inst)))
        self.instance = inst

        n = c_uint32(0)
        _check("vkEnumeratePhysicalDevices", vkEnumeratePhysicalDevices(inst, C.byref(n), None))
        devs = (c_void_p * n.value)()
        _check("vkEnumeratePhysicalDevices", vkEnumeratePhysicalDevices(inst, C.byref(n), devs))

        # RADV reports deviceType=CPU on this APU and the loader trampoline for
        # vkGetPhysicalDeviceQueueFamilyProperties segfaults, so pick by vendorID
        # and hardcode queue family 0 (GRAPHICS|COMPUTE|TRANSFER on RADV/NV).
        # VK_VENDOR: space-separated hex vendor IDs (default AMD 0x1002 + NVIDIA 0x10DE).
        # VK_DEVICE_INDEX: which matching physical device to use (default 0).
        wanted = {int(v, 16) for v in os.environ.get("VK_VENDOR", "1002 10de").split()}
        want_idx = int(os.environ.get("VK_DEVICE_INDEX", "0"))
        candidates: list[tuple[c_void_p, tuple[int, int, int, int, int, str]]] = []
        for e in devs:
            d = c_void_p(e)
            p = self._props(d)
            if p[2] in wanted: candidates.append((d, p))
        if want_idx >= len(candidates):
            raise RuntimeError(f"VK_DEVICE_INDEX={want_idx} but only {len(candidates)} device(s) with vendor in {sorted(wanted)}")
        amdev, props = candidates[want_idx]
        self.phys_device = amdev
        self.device_name, self.driver_version, self.vendor, self.device_id = props[5], props[1], props[2], props[3]
        self.api_version = props[0]
        # RADV on this APU does not make a submit's stores visible before the next in-order
        # dispatch may start: with 3+ cbs in flight, a reduce kernel reads stale partial sums
        # from the previous dispatch's temp buffer. It is bit-exact only when every submit's
        # fence is waited on before the next submit (see submit). NV550 and llvmpipe are
        # correct at full async. VK_ASYNC=0/1 overrides the per-vendor default.
        # Retested on mesa 26.1.6 (RADV gfx10.3): still broken -- VK_ASYNC=1 gives plausible
        # training loss but 2.5% eval accuracy (stale weight reads); keep it off.
        self._async = os.environ.get("VK_ASYNC", "0" if self.vendor == 0x1002 else "1") == "1"

        mp = VkPhysicalDeviceMemoryProperties()
        vkGetPhysicalDeviceMemoryProperties(amdev, C.byref(mp))
        self._mem_props = mp

        prio = c_float(1.0)
        qci = VkDeviceQueueCreateInfo(sType=ST_DEVICE_QUEUE_CREATE_INFO, queueFamilyIndex=0,
                                      queueCount=1, pQueuePriorities=_p(prio))
        dci = VkDeviceCreateInfo(sType=ST_DEVICE_CREATE_INFO, queueCreateInfoCount=1,
                                 pQueueCreateInfos=_p(qci))
        dev = c_void_p()
        _check("vkCreateDevice", vkCreateDevice(amdev, C.byref(dci), None, C.byref(dev)))
        self.device = dev

        q = c_void_p()
        vkGetDeviceQueue(dev, 0, 0, C.byref(q))
        self.queue = q

        cpci = VkCommandPoolCreateInfo(sType=ST_COMMAND_POOL_CREATE_INFO, queueFamilyIndex=0)
        pool = c_void_p()
        _check("vkCreateCommandPool", vkCreateCommandPool(dev, C.byref(cpci), None, C.byref(pool)))
        self.cmd_pool = pool
        # the NV 550 driver does not make one dispatch's stores visible to the next
        # dispatch's loads within a single command buffer; an explicit compute->compute
        # memory barrier between dispatches in the same cb restores ordering. VK_BARRIER=0
        # disables it (for A/B diagnosis).
        self._barrier = os.environ.get("VK_BARRIER", "1") == "1"
        # RADV on this APU does not honor in-cb visibility even with that barrier (stale
        # reads for both compute->compute and compute->transfer within one cb; a write
        # only becomes visible across the next submit's fence). There we give every kernel
        # and every copy its own command buffer (flush on bind_pipeline / cmd_copy) and let
        # the _async=False per-submit fence wait serialize them: the pre-batching behavior.
        # Adreno (5143) has the same class of bug but worse: a cb mixing copies/dispatches
        # loses stores (wrong results) and eventually hangs (driver returns VK_TIMEOUT from
        # vkQueueSubmit). There, per-cb kernels with async submits. VK_PER_KERNEL=0/1
        # overrides the per-vendor default (A/B against a driver fix).
        # Retested on mesa 26.1.6 (RADV gfx10.3, sync submits): still broken -- batched cbs
        # (VK_PER_KERNEL=0) give nan loss from step 1, the in-cb visibility bug survives.
        self._per_kernel_submit = os.environ.get("VK_PER_KERNEL", "1" if self.vendor in (0x1002, 0x5143) else "0") == "1"
        # VK_BATCH_SEM=1 (default 0): the per-CBT behavior of _per_kernel_submit above, minus
        # the per-unit submit+fence-wait. Every kernel dispatch and every copy still gets its
        # OWN primary command buffer (RADV's in-cb visibility bug, above), but instead of
        # submit()+fence-wait per unit, each finished CBT is ended and queued in
        # _batch_pending, and one vkQueueSubmit of N chained VkSubmitInfo entries is issued at
        # the next submit() boundary: entry 0 signals semaphore 0, entry i (0<i<N-1) waits
        # semaphore i-1 and signals semaphore i, entry N-1 waits semaphore N-2 and signals
        # none; a single fence (fences[0]) covers the whole submit. The binary semaphores make
        # each CBT start only after the previous one has completed (with memory visibility),
        # so the GPU runs the whole step's kernels back-to-back with no ~24-30us CPU gap to
        # drop into -- which is what lets DPM SCLK stay at 2000 MHz instead of bouncing to
        # 400 MHz between per-kernel submits. Micro-evidence: two CBTs in one submit chained
        # by a binary semaphore were bit-exact at 256KB-64MiB / 1000 iters, +4us/op vs an
        # in-cb barrier (/work/opencode/radv_batch/RADV_BATCHING.md); model-scale verification
        # was the open question. VK_BATCH_SEM_CHUNK=N splits the pending batch into submits of
        # N chained CBTs (N=1 reproduces today's per-kernel submit) -- a diagnostic/fallback if
        # the visibility bug turns out to extend to long semaphore chains at model scale.
        # VK_STREAM=1 (default 0): the one-CBT-per-unit recording of VK_BATCH_SEM, but instead
        # of accumulating the whole step into one giant submit, the pending CBTs are flushed to
        # the queue as soon as VK_STREAM_CHUNK (default 128) of them have ended. Each flush is
        # one vkQueueSubmit of N chained VkSubmitInfo entries -- exactly the VK_BATCH_SEM chain
        # -- except the chain continues ACROSS submit calls: the last entry of a chunk signals
        # a cross-chunk binary semaphore that the next chunk's first entry waits on (no CPU
        # fence wait between chunks; a small fence ring tracks the in-flight chunks and the CPU
        # only waits when a CBT range is about to be re-begun, at submit(wait=True), or at
        # synchronize()). This keeps the GPU fed while the CPU is still recording the rest of
        # the step -- VK_BATCH_SEM idles it for the whole ~96ms recording window and DPM drops
        # SCLK 2000->400 MHz, so its batch starts at the low clock (see BATCH_SEM.md).
        # ASSUMPTION (unproven by the micro-tests, verified at model scale): the binary-
        # semaphore visibility RADV honors within one submit (VK_BATCH_SEM is bit-exact there)
        # also holds across vkQueueSubmit calls on the same queue. That is settled by the
        # 40-token temp-0 decode test (first 10 tokens must be
        # [364, 1141, 25438, 57902, 1680, 430, 279, 242476, 300, 21262]) at every chunk size;
        # see /work/opencode/radv_batch/VK_STREAM.md. VK_STREAM_SYNCCHUNK=1 fence-waits between
        # chunks (diagnostic: isolates the cross-submit question; two proven mechanisms, no new
        # assumption).
        self._stream = os.environ.get("VK_STREAM", "0") == "1"
        self._stream_chunk = min(POOL, int(os.environ.get("VK_STREAM_CHUNK", "128")) or 128)
        self._stream_syncchunk = os.environ.get("VK_STREAM_SYNCCHUNK", "0") == "1"
        # VK_REPLAY=1 (default 0): pre-recorded CBT resubmit. A (program, bufs, nvals, grid)
        # combination gets a TEMPLATE primary CBT in its own arena slot (record_template, see
        # TPL_POOL) once it has been seen; every later execution of the same signature only
        # rewrites its shared-per-config UBO and re-submits the template through the same
        # VK_STREAM chunked chained submits instead of re-recording bind/bdesc/dispatch.
        # Templates are never re-begun; replay_cb refuses (returning False) a template whose
        # previous resubmit is still pending/in-flight (a primary CBT may not be pending
        # twice) and the caller rotates to the next VK_REPLAY_POOL template or falls back to
        # a fresh record. Requires VK_STREAM (templates flow through its chunked submits).
        self._replay = os.environ.get("VK_REPLAY", "0") == "1"
        if self._replay and not self._stream:
            raise RuntimeError("VK_REPLAY requires VK_STREAM=1: template CBTs flow through the chunked chained submits")
        self._replay_pool = max(1, int(os.environ.get("VK_REPLAY_POOL", "1")))
        self._batch_sem = os.environ.get("VK_BATCH_SEM", "0") == "1" and not self._stream
        self._batch_chunk = int(os.environ.get("VK_BATCH_SEM_CHUNK", "0")) or POOL
        self._cb_pool = POOL if self._batch_sem or self._stream else RING
        self._cb_total = self._cb_pool + (TPL_POOL if self._replay else 0)
        cba = VkCommandBufferAllocateInfo(sType=ST_COMMAND_BUFFER_ALLOCATE_INFO, commandPool=pool,
                                           level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=self._cb_total)
        cbs = (c_void_p * self._cb_total)()
        _check("vkAllocateCommandBuffers", vkAllocateCommandBuffers(dev, C.byref(cba), cbs))
        self.cbs, self._cb_active = cbs, False
        # self._slot: the CBT the active command buffer belongs to (or the next one to begin);
        # indexes self.cbs and wraps at self._cb_pool (RING normally, POOL in VK_BATCH_SEM).
        # self._inflight[i]/self._poisoned[i]: per-slot fence state, non-batch mode (see
        # _wait_fence); in VK_BATCH_SEM the whole batch shares fence 0 and _batch_inflight
        # tracks the submitted-not-waited cbs instead. self._ncmd: commands in the active cb.
        self._slot, self._ncmd = 0, 0
        self._inflight, self._poisoned = [False] * self._cb_pool, [False] * self._cb_pool
        self._cb_has_dispatch, self._nkernels, self._in_kernel = False, 0, False
        if self._batch_sem or self._stream:
            # semaphore arena: one binary semaphore per CBT slot (a batch of N cbs chains
            # through semaphores 0..N-2). Allocated once; never reset (auto-reset on wait).
            # _batch_pending: ended-but-unsubmitted CBT handles (fresh main-arena CBTs and,
            # with VK_REPLAY, re-submitted template CBTs); _batch_inflight: CBT handle values
            # in submitted-but-unwaited chunks (VK_BATCH_SEM: the one batch on fence 0;
            # VK_STREAM: the in-flight chunk records in _stream_chunks).
            self._batch_pending: list = []
            self._batch_inflight: set[int | None] = set()
            self._batch_sems = (c_void_p * POOL)()
            for i in range(POOL):
                sci = VkSemaphoreCreateInfo(sType=ST_SEMAPHORE_CREATE_INFO)
                out = c_void_p()
                _check("vkCreateSemaphore", vkCreateSemaphore(dev, C.byref(sci), None, C.byref(out)))
                self._batch_sems[i] = out
            if self._stream:
                # VK_STREAM: _stream_chunks: in-flight chunk records (fence ring index, CBT
                # handle values), oldest first; _stream_wait_sem: the semaphore the next chunk's first
                # entry must wait on (None only before the very first chunk ever submitted --
                # it survives step-boundary drains: the drain's fence wait guarantees the
                # dangling signal is complete, and the next chunk's wait consumes it);
                # _stream_signal_i: which of the two ping-ponged cross semaphores the next
                # chunk's last entry signals; _stream_fence_i: next ring fence for a chunk.
                self._stream_chunks: list[tuple[int, list[int | None]]] = []  # (fence idx, CBT handle values)
                self._stream_wait_sem: int | None = None  # handle of the semaphore the next chunk waits on
                self._stream_signal_i = 0
                self._stream_fence_i = 0
                self._stream_sems = (c_void_p * 2)()
                for i in range(2):
                    sci = VkSemaphoreCreateInfo(sType=ST_SEMAPHORE_CREATE_INFO)
                    out = c_void_p()
                    _check("vkCreateSemaphore", vkCreateSemaphore(dev, C.byref(sci), None, C.byref(out)))
                    self._stream_sems[i] = out
            if self._replay:
                # template arena bookkeeping: _tpl_slot is the next free arena slot (index
                # into self.cbs, offset by _cb_pool); _tpl_live is the set of values of all
                # recorded template CBTs (destroy_cfg drains before a pipeline is destroyed);
                # _repl: [resubmit hits, templates recorded, busy refusals (template still
                # pending/in-flight: rotated to the next pool slot or fresh-recorded)].
                self._tpl_slot, self._tpl_live, self._repl = 0, set(), [0, 0, 0]
        # VK_RT_PROFILE=1 (default 0): accumulate per-section CPU times in self._pt_acc (seconds),
        # for the step breakdown (replay record / submit processing / waits / copies).
        self._prof = os.environ.get("VK_RT_PROFILE", "0") == "1"
        self._pt_acc: dict[str, float] = {}

        self.fences = (c_void_p * RING)()
        for i in range(RING):
            fci = VkFenceCreateInfo(sType=ST_FENCE_CREATE_INFO)
            out = c_void_p()
            _check("vkCreateFence", vkCreateFence(dev, C.byref(fci), None, C.byref(out)))
            self.fences[i] = out

        self._buffers, self._modules, self._dsls, self._pipelines = [], [], [], []

    @staticmethod
    def _props(dev):
        raw = (c_uint8 * PROPS_SIZE)()
        vkGetPhysicalDeviceProperties(dev, raw)
        b = bytes(raw)
        api, drv, vendor, did, dtype = struct.unpack_from("<5I", b, 0)
        name = b[20:276].split(b"\0", 1)[0].decode()
        return api, drv, vendor, did, dtype, name

    # -- optional per-section CPU timing (VK_RT_PROFILE) --

    def _pt0(self):
        return time.perf_counter() if self._prof else 0.0

    def _pt(self, k, t0):
        if self._prof and t0: self._pt_acc[k] = self._pt_acc.get(k, 0.0) + time.perf_counter() - t0

    # -- buffers --

    def buffer(self, size, host_visible=False):
        bci = VkBufferCreateInfo(sType=ST_BUFFER_CREATE_INFO, size=size,
                                 usage=VK_BUFFER_USAGE_ALL, sharingMode=VK_SHARING_MODE_EXCLUSIVE)
        buf = c_void_p()
        _check("vkCreateBuffer", vkCreateBuffer(self.device, C.byref(bci), None, C.byref(buf)))
        try:
            req = VkMemoryRequirements()
            vkGetBufferMemoryRequirements(self.device, buf, C.byref(req))
            mp = self._mem_props
            hvhc = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            # (rank, memtype): host-visible VRAM (ReBAR window) preferred, then sys RAM;
            # rank 1 = "anything usable" fallback
            cands: list[tuple[int, int]] = []
            for i in range(mp.memoryTypeCount):
                if not (req.memoryTypeBits >> i) & 1:
                    continue
                pf = mp.memoryTypes[i].propertyFlags
                if host_visible:
                    if pf & hvhc == hvhc:
                        cands.append((0 if pf & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT else 1, i))
                elif pf & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:
                    cands.append((0, i))
            if not cands and not host_visible:  # no device-local heap: take anything usable
                cands = [(1, i) for i in range(mp.memoryTypeCount) if (req.memoryTypeBits >> i) & 1]
            cands.sort()
            # try in rank order; a full ReBAR maps all VRAM, a partial window (256 MB) fills up fast
            mem, err, idx = c_void_p(), VK_ERROR_OUT_OF_DEVICE_MEMORY, None
            for _, i in cands:
                mai = VkMemoryAllocateInfo(sType=ST_MEMORY_ALLOCATE_INFO, allocationSize=req.size, memoryTypeIndex=i)
                if (r := vkAllocateMemory(self.device, C.byref(mai), None, C.byref(mem))) == VK_SUCCESS:
                    idx = i
                    break
                err = r
            if idx is None:
                raise VkError("vkAllocateMemory", err)
            _check("vkBindBufferMemory", vkBindBufferMemory(self.device, buf, mem, 0))
        except BaseException:
            vkDestroyBuffer(self.device, buf, None)
            raise
        b = VBuffer(self, buf, mem, size, host_visible)
        self._buffers.append(b)
        return b

    def free_buffer(self, vbuf:VBuffer):
        # data buffers are never freed (the LRU reuses them); this exists for the
        # allocator's staging pool to drop a buffer it is growing past
        if vbuf._mapped is not None:
            vkUnmapMemory(self.device, vbuf.mem)
            vbuf._mapped, vbuf._arr = None, None
        vkDestroyBuffer(self.device, vbuf.handle, None)
        vkFreeMemory(self.device, vbuf.mem, None)
        self._buffers.remove(vbuf)

    # -- GPU timestamps (diagnostic): one timestamp query per recorded command boundary --

    def ts_pool(self, n:int):
        qi = VkQueryPoolCreateInfo(sType=ST_QUERY_POOL_CREATE_INFO, queryType=VK_QUERY_TYPE_TIMESTAMP,
                                   queryCount=n)
        pool = c_void_p()
        _check("vkCreateQueryPool", vkCreateQueryPool(self.device, C.byref(qi), None, C.byref(pool)))
        return pool
    def ts_write(self, pool, i:int):
        # record timestamp i at BOTTOM_OF_PIPE of the active CBT (after every command recorded
        # in it so far); starts a fresh CBT when none is active -- the caller submits it like any
        # other unit. Read with ts_read only after the covering fence has signaled.
        if not self._cb_active: self._begin_cmd()
        vkCmdWriteTimestamp(self.cbs[self._slot], VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, i)
    def ts_read(self, pool, n:int) -> list:
        out = (c_uint64 * n)()
        _check("vkGetQueryPoolResults", vkGetQueryPoolResults(self.device, pool, 0, n, n * 8, out, 8,
                                                              VK_QUERY_RESULT_64_BIT_BIT))
        return list(out)

    # -- shaders / pipelines / descriptors --

    def create_shader_module(self, spv_bytes):
        if len(spv_bytes) % 4:
            raise ValueError("SPIR-V size not a multiple of 4")
        words = (c_uint32 * (len(spv_bytes) // 4)).from_buffer_copy(spv_bytes)
        sci = VkShaderModuleCreateInfo(sType=ST_SHADER_MODULE_CREATE_INFO,
                                       codeSize=len(spv_bytes), pCode=C.cast(words, c_void_p))
        mod = c_void_p()
        _check("vkCreateShaderModule", vkCreateShaderModule(self.device, C.byref(sci), None, C.byref(mod)))
        m = VShaderModule(self, mod, words)
        self._modules.append(m)
        return m

    def create_descriptor_set_layout(self, bindings, buffers=()):
        """bindings: [(set, binding, VkDescriptorType[, stage])]; buffers: one entry
        per binding. An entry is a VBuffer (updated at offset 0 with the full buffer
        range) or a (VBuffer, offset, range) tuple to address a sub-buffer view."""
        if len(buffers) != len(bindings):
            raise ValueError("expected one buffer per binding")
        blist = (VkDescriptorSetLayoutBinding * len(bindings))()
        counts: dict[int, int] = {}
        for i, e in enumerate(bindings):
            stage = e[3] if len(e) > 3 else VK_SHADER_STAGE_COMPUTE_BIT
            blist[i] = VkDescriptorSetLayoutBinding(binding=e[1], descriptorType=e[2],
                                                    descriptorCount=1, stageFlags=stage)
            counts[e[2]] = counts.get(e[2], 0) + 1
        lci = VkDescriptorSetLayoutCreateInfo(sType=ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                              bindingCount=len(bindings), pBindings=C.cast(blist, c_void_p))
        layout = c_void_p()
        _check("vkCreateDescriptorSetLayout",
               vkCreateDescriptorSetLayout(self.device, C.byref(lci), None, C.byref(layout)))

        sizes = (VkDescriptorPoolSize * len(counts))()
        for i, (t, n) in enumerate(counts.items()):
            sizes[i] = VkDescriptorPoolSize(type=t, descriptorCount=n)
        ppci = VkDescriptorPoolCreateInfo(sType=ST_DESCRIPTOR_POOL_CREATE_INFO, maxSets=1,
                                          poolSizeCount=len(counts), pPoolSizes=C.cast(sizes, c_void_p))
        pool = c_void_p()
        _check("vkCreateDescriptorPool",
               vkCreateDescriptorPool(self.device, C.byref(ppci), None, C.byref(pool)))

        ainfo = VkDescriptorSetAllocateInfo(sType=ST_DESCRIPTOR_SET_ALLOCATE_INFO,
                                            descriptorPool=pool, descriptorSetCount=1,
                                            pSetLayouts=_p(layout))
        ds = c_void_p()
        _check("vkAllocateDescriptorSets", vkAllocateDescriptorSets(self.device, C.byref(ainfo), C.byref(ds)))

        infos = (VkDescriptorBufferInfo * len(buffers))()
        for i, b in enumerate(buffers):
            vb, off, rng = (b[0], b[1], b[2]) if isinstance(b, tuple) else (b, 0, b.size)
            infos[i] = VkDescriptorBufferInfo(buffer=_handle(vb), offset=off, range=rng)
        writes = (VkWriteDescriptorSet * len(bindings))()
        for i, e in enumerate(bindings):
            writes[i] = VkWriteDescriptorSet(sType=ST_WRITE_DESCRIPTOR_SET, dstSet=ds, dstBinding=e[1],
                                             descriptorCount=1, descriptorType=e[2],
                                             pBufferInfo=_p(infos[i]))
        vkUpdateDescriptorSets(self.device, len(bindings), writes, 0, None)
        dsl = VDescriptorSetLayout(self, layout, pool, ds, list(buffers))
        self._dsls.append(dsl)
        return dsl

    def create_compute_pipeline(self, module, entry="main", desc_set_layouts=None):
        dsls = list(desc_set_layouts or [])
        layout = c_void_p()
        if dsls:
            handles = (c_void_p * len(dsls))()
            for i, d in enumerate(dsls):
                handles[i] = d.layout
            plci = VkPipelineLayoutCreateInfo(sType=ST_PIPELINE_LAYOUT_CREATE_INFO,
                                              setLayoutCount=len(dsls), pSetLayouts=C.cast(handles, c_void_p))
            _check("vkCreatePipelineLayout",
                   vkCreatePipelineLayout(self.device, C.byref(plci), None, C.byref(layout)))
        stage = VkPipelineShaderStageCreateInfo(sType=ST_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                stage=VK_SHADER_STAGE_COMPUTE_BIT,
                                                module=module.handle, pName=entry.encode())
        cpci = VkComputePipelineCreateInfo(sType=ST_COMPUTE_PIPELINE_CREATE_INFO, stage=stage,
                                           layout=layout, basePipelineIndex=-1)
        pipe = c_void_p()
        _check("vkCreateComputePipelines",
               vkCreateComputePipelines(self.device, None, 1, C.byref(cpci), None, C.byref(pipe)))
        p = VPipeline(self, pipe, layout, module)
        self._pipelines.append(p)
        return p

    def destroy_cfg(self, pipeline:VPipeline, dsl:VDescriptorSetLayout, ubo:VBuffer|None=None):
        """Release one program's driver objects as soon as the program is gone, not at
        close: thousands of BEAM-search candidates would otherwise leak pipelines/modules/
        descriptor sets and exhaust the driver's per-context resources. A pending command
        buffer that still dispatches this pipeline is flushed first."""
        if not getattr(self, "device", None) or not self.device: return
        if self._cb_active and self._cb_has_dispatch:
            if self._stream: self.synchronize()  # its dispatch must be DONE, not just in flight
            else: self.submit()
        elif self._replay and (self._tpl_live & (self._batch_inflight | set(self._batch_pending))):
            self.synchronize()  # a live template CBT (pending or in flight) still references this pipeline's objects
        self._pipelines.remove(pipeline)
        self._dsls.remove(dsl)
        if pipeline.module in self._modules: self._modules.remove(pipeline.module)
        _check("vkDestroyPipeline", vkDestroyPipeline(self.device, pipeline.handle, None))
        if pipeline.layout: _check("vkDestroyPipelineLayout", vkDestroyPipelineLayout(self.device, pipeline.layout, None))
        _check("vkFreeDescriptorSets", vkFreeDescriptorSets(self.device, dsl.pool, 1, _p(dsl.set)))
        _check("vkDestroyDescriptorPool", vkDestroyDescriptorPool(self.device, dsl.pool, None))
        _check("vkDestroyDescriptorSetLayout", vkDestroyDescriptorSetLayout(self.device, dsl.layout, None))
        if pipeline.module: _check("vkDestroyShaderModule", vkDestroyShaderModule(self.device, pipeline.module.handle, None))
        if ubo is not None: self.free_buffer(ubo)

    # -- command recording / submit --

    def _wait_fence(self, i, timeout_ns=30_000_000_000):
        """Wait for ring slot i's fence. A timeout poisons the slot (the kernel is still on
        the GPU and the fence cannot signal); later waits on it fail fast instead of paying
        the timeout again, and a non-blocking check recovers the slot if the kernel was only
        slow. Vulkan cannot abort a running compute kernel, so a truly stuck one blocks its
        slot until the context dies."""
        f = C.cast((c_void_p * 1)(self.fences[i]), c_void_p)
        t0 = self._pt0()
        if self._poisoned[i]:
            if vkWaitForFences(self.device, 1, f, 1, 0) == VK_SUCCESS: self._poisoned[i] = False
        else:
            res = vkWaitForFences(self.device, 1, f, 1, timeout_ns)
            if res == VK_TIMEOUT:
                self._poisoned[i] = True
                self._pt("fwait", t0)
                raise VkError("vkWaitForFences", VK_TIMEOUT)
            if res != VK_SUCCESS: _check("vkWaitForFences", res)
        self._pt("fwait", t0)
        self._inflight[i] = False

    def _ensure_cb(self):
        if not self._cb_active:
            # CBT handle value of the slot the cursor is about to (re)begin
            sv = _cvt(self.cbs[self._slot])
            if self._batch_sem:
                # backpressure: the CBT pool would be exhausted (every cb still pending and
                # not yet submitted) or the cursor wrapped to a cb whose batch is still in
                # flight; flush/wait so the cb is idle before it can be reset and re-recorded
                if len(self._batch_pending) + 1 > self._cb_pool or sv in self._batch_inflight:
                    self._batch_submit(wait=True)
            elif self._stream:
                # backpressure: the cursor wrapped to a cb inside an in-flight chunk; wait on
                # that chunk's fence so the cb is idle before it is reset and re-recorded.
                # Expected to be rare on this workload (the CPU records slower than the GPU
                # runs); it also caps the in-flight CBT count at the pool size.
                if len(self._batch_pending) + 1 > self._cb_pool or sv in self._batch_inflight:
                    for fi, cbs in self._stream_chunks:
                        if sv in cbs:
                            self._stream_reap(fi, 30_000_000_000)
                            break
            else:
                # backpressure: wrapping back to a slot whose submit is still in flight; its
                # command buffer must be idle before it can be reset and re-recorded
                if self._inflight[self._slot]: self._wait_fence(self._slot)
            t0 = self._pt0()
            _check("vkResetCommandBuffer", vkResetCommandBuffer(self.cbs[self._slot], 0))
            bi = VkCommandBufferBeginInfo(sType=ST_COMMAND_BUFFER_BEGIN_INFO)
            _check("vkBeginCommandBuffer", vkBeginCommandBuffer(self.cbs[self._slot], C.byref(bi)))
            self._cb_active, self._ncmd, self._cb_has_dispatch, self._nkernels, self._in_kernel = True, 0, False, 0, False
            self._pt("cb", t0)

    def _begin_cmd(self):
        # start recording one command in the current cb. The command-count backstop only
        # fires between kernels (a kernel's bind/bdesc/dispatch must stay in one cb); the
        # kernel-count flush happens in cmd_bind_pipeline, before the record.
        self._ensure_cb()
        if self._ncmd >= CB_FLUSH_MAX and not self._in_kernel: self.submit()
        self._ncmd += 1

    # RADV (mesa 26.1.6, RENOIR) silently truncates H2D vkCmdCopyBuffers above a
    # non-deterministic size (~0.4-2 GiB observed): the destination keeps its first N bytes
    # and the rest stays zero, no error is reported. D2H is unaffected. Chunk to stay below
    # the smallest observed boundary (0.389 GiB tensor landed, 1.02 GiB tensor truncated at
    # 0.406 GiB) (see extra/vulkan/RADV_BIGCOPY.md).
    COPY_CHUNK = 256 * 1024 * 1024
    def cmd_copy(self, dst, src, size=None, src_off=0, dst_off=0):
        t0 = self._pt0()
        n = size if size is not None else min(dst.size, src.size)
        # this copy must not share a cb with a dispatch: end the pending unit's cb and either
        # submit it now (per-kernel mode) or queue it for the next chained submit (batch/stream)
        if self._batch_sem or self._stream: self._batch_end_unit()
        elif self._per_kernel_submit: self.submit()
        self._begin_cmd()
        done = 0
        while done < n:
            c = min(self.COPY_CHUNK, n - done)
            cp = VkBufferCopy(srcOffset=src_off + done, dstOffset=dst_off + done, size=c)
            vkCmdCopyBuffer(self.cbs[self._slot], _handle(src), _handle(dst), 1, C.byref(cp))
            done += c
        self._pt("copy", t0)

    def cmd_bind_pipeline(self, pipeline):
        t0 = self._pt0()
        # a kernel starts here; flush the pending cb at a kernel boundary (RADV: every
        # kernel, others: every CB_FLUSH_KERNELS) so this kernel's commands stay together
        if self._batch_sem or self._stream:
            self._batch_end_unit()  # this kernel gets its own cb, queued for the next chained submit
        elif self._cb_active and (self._per_kernel_submit or self._nkernels >= CB_FLUSH_KERNELS):
            self.submit()
        self._ensure_cb()
        vkCmdBindPipeline(self.cbs[self._slot], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle)
        self._ncmd += 1
        self._nkernels += 1
        self._in_kernel = True
        self._pt("bind", t0)

    def cmd_bind_descriptor_sets(self, pipeline_or_layout, desc_set):
        t0 = self._pt0()
        layout = pipeline_or_layout.layout if hasattr(pipeline_or_layout, "layout") else pipeline_or_layout
        self._begin_cmd()
        # pDescriptorSets is a pointer to an array of set handles, not the handle itself
        sets = (c_void_p * 1)(_handle(desc_set))
        vkCmdBindDescriptorSets(self.cbs[self._slot], VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                                0, 1, sets, 0, None)
        self._pt("bdesc", t0)

    def cmd_dispatch(self, x, y, z):
        t0 = self._pt0()
        self._begin_cmd()
        # NV 550 needs an explicit barrier to see the previous in-cb dispatch's stores
        if self._barrier and self._cb_has_dispatch:
            mb = VkMemoryBarrier(sType=ST_MEMORY_BARRIER, srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,
                                 dstAccessMask=VK_ACCESS_SHADER_READ_BIT)
            vkCmdPipelineBarrier(self.cbs[self._slot], VK_SHADER_STAGE_COMPUTE_BIT, VK_SHADER_STAGE_COMPUTE_BIT,
                                  0, 1, C.cast(C.byref(mb), c_void_p), 0, None, 0, None)
        vkCmdDispatch(self.cbs[self._slot], x, y, z)
        self._cb_has_dispatch, self._in_kernel = True, False
        self._pt("disp", t0)

    # -- VK_BATCH_SEM: per-CBT, semaphore-chained submits --

    def _batch_end_unit(self):
        # end the active unit's command buffer and queue it for the next chained submit; the
        # unit will run after the earlier pending units via binary semaphores. No submit here
        # (VK_STREAM: one as soon as a full chunk has accumulated). _batch_pending holds CBT
        # HANDLES, not arena slots: with VK_REPLAY re-submitted template CBTs (arena slots
        # the recording cursor never touches) flow through the same pending list.
        if not self._cb_active: return
        i = self._slot
        t0 = self._pt0()
        _check("vkEndCommandBuffer", vkEndCommandBuffer(self.cbs[i]))
        self._batch_pending.append(_cvt(self.cbs[i]))
        self._slot = (i + 1) % self._cb_pool
        self._cb_active = False
        self._pt("cb", t0)
        if self._stream and len(self._batch_pending) >= self._stream_chunk:
            self._stream_flush(30_000_000_000)  # feed the queue while the CPU keeps recording

    def _wait_batch_fence(self, timeout_ns=30_000_000_000):
        # wait on the batch fence (fences[0]); it signals when the whole submitted chain is
        # done. Clears _batch_inflight (every cb in the chain is now idle and reusable).
        f = C.cast((c_void_p * 1)(self.fences[0]), c_void_p)
        t0 = self._pt0()
        res = vkWaitForFences(self.device, 1, f, 1, timeout_ns)
        self._pt("fwait", t0)
        if res == VK_TIMEOUT:
            raise VkError("vkWaitForFences", VK_TIMEOUT)
        if res != VK_SUCCESS: _check("vkWaitForFences", res)
        self._batch_inflight.clear()

    def _submit_chain(self, cbs, do_wait, timeout_ns):
        # one vkQueueSubmit of len(cbs) chained VkSubmitInfo entries over fence 0: entry 0
        # signals semaphore 0, entry i (0<i<n-1) waits semaphore i-1 and signals semaphore i,
        # entry n-1 waits semaphore n-2 and signals none. The semaphores make each cb start
        # only after the previous has completed (with memory visibility), preserving the
        # per-unit ordering the RADV in-cb bug requires. fence 0 must not be in flight here
        # (true on RADV: every submit waits before returning). pWaitDstStageMask must be
        # non-NULL when waiting: RADV 26.1.6 segfaults on NULL (see RADV_BATCHING.md, f1).
        n = len(cbs)
        sies = (VkSubmitInfo * n)()
        cb_arr = (c_void_p * n)()
        masks = (c_uint32 * n)()
        for i in range(n):
            cb_arr[i] = cbs[i]
            masks[i] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            sies[i].sType = ST_SUBMIT_INFO
            sies[i].pNext = None
            sies[i].commandBufferCount = 1
            # NOTE: C.byref(arr, off) takes a BYTE offset, not an element index
            sies[i].pCommandBuffers = C.cast(C.byref(cb_arr, i * C.sizeof(c_void_p)), c_void_p)
            if i > 0:
                sies[i].waitSemaphoreCount = 1
                sies[i].pWaitSemaphores = C.cast(C.byref(self._batch_sems, (i - 1) * C.sizeof(c_void_p)), c_void_p)
                sies[i].pWaitDstStageMask = C.cast(C.byref(masks, (i - 1) * C.sizeof(c_uint32)), c_void_p)
            if i < n - 1:
                sies[i].signalSemaphoreCount = 1
                sies[i].pSignalSemaphores = C.cast(C.byref(self._batch_sems, i * C.sizeof(c_void_p)), c_void_p)
        # fence 0 must not be in flight when we reset it; if a prior chain is still out
        # there (only possible with VK_ASYNC=1, unsupported on RADV) wait it out first
        if self._batch_inflight: self._wait_batch_fence(timeout_ns)
        _check("vkResetFences", vkResetFences(self.device, 1, C.cast((c_void_p * 1)(self.fences[0]), c_void_p)))
        self._batch_inflight.update(cbs)
        t0 = self._pt0()
        _check("vkQueueSubmit", vkQueueSubmit(self.queue, n, sies, self.fences[0]))
        self._pt("submit", t0)
        if do_wait: self._wait_batch_fence(timeout_ns)

    def _batch_submit(self, wait:bool=False, timeout_ms:int|None=None):
        """VK_BATCH_SEM: end any active unit's cb, then issue one vkQueueSubmit of chained
        VkSubmitInfo entries for every pending cb (split into _batch_chunk-sized submits),
        each covered by fence 0. wait / not self._async blocks until the fence signals
        (RADV: always -- the GPU then runs the whole batch with no CPU gaps, the point)."""
        if self._cb_active: self._batch_end_unit()
        if not self._batch_pending: return VK_SUCCESS
        timeout_ns = (timeout_ms if timeout_ms is not None else 30000) * 1_000_000
        do_wait = wait or not self._async
        i, n = 0, len(self._batch_pending)
        while i < n:
            chunk = self._batch_pending[i:i + self._batch_chunk]
            self._submit_chain(chunk, do_wait, timeout_ns)
            i += len(chunk)
        self._batch_pending = []
        return VK_SUCCESS

    # -- VK_STREAM: per-chunk, cross-submit-chained submits --

    def _stream_reap(self, fi, timeout_ns=30_000_000_000):
        # wait on ring fence fi (every cb of that chunk has completed and the cross-chunk
        # signal it carries is done: a fence signals after its submit's semaphore signals),
        # then drop the chunk record: the cbs are idle and the fence is free for reuse.
        # Binary semaphores need no reap (auto-reset on the consuming wait).
        f = C.cast((c_void_p * 1)(self.fences[fi]), c_void_p)
        t0 = self._pt0()
        res = vkWaitForFences(self.device, 1, f, 1, timeout_ns)
        self._pt("fwait", t0)
        if res == VK_TIMEOUT:
            raise VkError("vkWaitForFences", VK_TIMEOUT)
        if res != VK_SUCCESS: _check("vkWaitForFences", res)
        for i, (f2, cbs) in enumerate(self._stream_chunks):
            if f2 == fi:
                self._batch_inflight.difference_update(cbs)
                del self._stream_chunks[i]
                break

    def _stream_submit_chunk(self, cbs, timeout_ns):
        # one vkQueueSubmit of len(cbs) chained VkSubmitInfo entries on a ring fence, the
        # VK_BATCH_SEM chain extended across submits: entry 0 additionally waits on the
        # semaphore signaled by the PREVIOUS chunk's last entry (None only before the first
        # chunk ever), and the LAST entry signals the semaphore the NEXT chunk's entry 0
        # will wait on. The two cross semaphores ping-pong, so a submit never waits on and
        # signals the same semaphore. No CPU fence wait is implied.
        n = len(cbs)
        if len(self._stream_chunks) >= RING:
            self._stream_reap(self._stream_chunks[0][0], timeout_ns)  # fence ring: free the oldest
        fi = self._stream_fence_i
        self._stream_fence_i = (fi + 1) % RING
        _check("vkResetFences", vkResetFences(self.device, 1, C.cast((c_void_p * 1)(self.fences[fi]), c_void_p)))
        sies = (VkSubmitInfo * n)()
        cb_arr = (c_void_p * n)()
        masks = (c_uint32 * n)()
        waitsem = (c_void_p * 1)()  # stable slot for entry 0's cross-chunk wait semaphore
        cross = self._stream_wait_sem
        for i in range(n):
            cb_arr[i] = cbs[i]
            masks[i] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            sies[i].sType = ST_SUBMIT_INFO
            sies[i].pNext = None
            sies[i].commandBufferCount = 1
            # NOTE: C.byref(arr, off) takes a BYTE offset, not an element index
            sies[i].pCommandBuffers = C.cast(C.byref(cb_arr, i * C.sizeof(c_void_p)), c_void_p)
            if i > 0:
                sies[i].waitSemaphoreCount = 1
                sies[i].pWaitSemaphores = C.cast(C.byref(self._batch_sems, (i - 1) * C.sizeof(c_void_p)), c_void_p)
                sies[i].pWaitDstStageMask = C.cast(C.byref(masks, (i - 1) * C.sizeof(c_uint32)), c_void_p)
            elif cross is not None:
                waitsem[0] = cross
                sies[i].waitSemaphoreCount = 1
                sies[i].pWaitSemaphores = C.cast(C.byref(waitsem), c_void_p)
                sies[i].pWaitDstStageMask = C.cast(C.byref(masks), c_void_p)
            if i < n - 1:
                sies[i].signalSemaphoreCount = 1
                sies[i].pSignalSemaphores = C.cast(C.byref(self._batch_sems, i * C.sizeof(c_void_p)), c_void_p)
            else:
                # cross-chunk link for the next chunk's entry 0
                sies[i].signalSemaphoreCount = 1
                sies[i].pSignalSemaphores = C.cast(C.byref(self._stream_sems, self._stream_signal_i * C.sizeof(c_void_p)), c_void_p)
        t0 = self._pt0()
        _check("vkQueueSubmit", vkQueueSubmit(self.queue, n, sies, self.fences[fi]))
        self._pt("submit", t0)
        self._stream_chunks.append((fi, list(cbs)))
        self._batch_inflight.update(cbs)
        self._stream_wait_sem = self._stream_sems[self._stream_signal_i]
        self._stream_signal_i ^= 1
        if self._stream_syncchunk:
            self._stream_reap(fi, timeout_ns)  # diagnostic: fence-wait between chunks

    def _stream_flush(self, timeout_ns):
        # flush the ended-but-unsubmitted CBTs as _stream_chunk-sized chained submits; no
        # CPU fence wait (the chunks run on the GPU while the CPU records the next ones)
        while self._batch_pending:
            chunk = self._batch_pending[:self._stream_chunk]
            self._batch_pending = self._batch_pending[len(chunk):]
            self._stream_submit_chunk(chunk, timeout_ns)

    def _stream_submit(self, wait:bool=False, timeout_ms:int|None=None):
        """VK_STREAM: end any active unit's cb, flush the pending cbs as chunk submits, and
        if wait=True wait on the final chunk's fence -- the cross-submit semaphore chain plus
        same-queue order make that one wait cover every earlier chunk of the step too."""
        if self._cb_active: self._batch_end_unit()
        timeout_ns = (timeout_ms if timeout_ms is not None else 30000) * 1_000_000
        self._stream_flush(timeout_ns)
        if wait and self._stream_chunks:
            self._stream_reap(self._stream_chunks[-1][0], timeout_ns)
        return VK_SUCCESS

    def _stream_synchronize(self, timeout_ms:int|None=None):
        """VK_STREAM step-boundary drain: flush the rest as a final chunk, then wait on ALL
        in-flight chunk fences (each is already signaled or will be shortly; the step's next
        submit must start from a fully drained queue)."""
        timeout_ns = (timeout_ms if timeout_ms is not None else 30000) * 1_000_000
        if self._cb_active: self._batch_end_unit()
        self._stream_flush(timeout_ns)
        for fi, _ in list(self._stream_chunks):
            self._stream_reap(fi, timeout_ns)

    # -- VK_REPLAY: pre-recorded template CBTs --

    def record_template(self, pipeline:VPipeline, desc_set, grid:tuple[int, int, int]) -> int | None:
        """Record one dispatch as a TEMPLATE: a fresh primary CBT in its own arena slot
        (offset by _cb_pool, never touched by the recording cursor), ended immediately and
        never re-begun, so it may be re-submitted indefinitely. It binds the per-config
        descriptor set (fixed buffer handles) and reads its UBO at execute time, so rewriting
        the UBO before each resubmit updates what the next execution sees. Returns the CBT
        handle value, or None when the arena is exhausted (the caller falls back to a fresh
        recording)."""
        if self._tpl_slot >= TPL_POOL:
            return None
        cbt = C.cast(self.cbs[self._cb_pool + self._tpl_slot] or 0, c_void_p)
        self._tpl_slot += 1
        bi = VkCommandBufferBeginInfo(sType=ST_COMMAND_BUFFER_BEGIN_INFO)
        _check("vkBeginCommandBuffer", vkBeginCommandBuffer(cbt, C.byref(bi)))
        vkCmdBindPipeline(cbt, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle)
        sets = (c_void_p * 1)(_handle(desc_set))
        vkCmdBindDescriptorSets(cbt, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, sets, 0, None)
        vkCmdDispatch(cbt, grid[0], grid[1], grid[2])
        _check("vkEndCommandBuffer", vkEndCommandBuffer(cbt))
        v = cbt.value
        self._tpl_live.add(v)
        self._repl[1] += 1
        return v

    def replay_cb(self, v:int|None) -> bool:
        """Re-submit a recorded template CBT (by handle value) in the current pending chain,
        so it flows through the same VK_STREAM chunked semaphore-chained submits as fresh
        units. A primary CBT may not be pending twice at once: if the template's previous
        resubmit is still in _batch_pending (not yet submitted -- appending it again would
        put the same CBT into one submit, which RADV silently drops) or in an un-reaped
        chunk, return False and let the caller rotate to the next VK_REPLAY_POOL template or
        fall back to a fresh record. Fence-waiting here instead would stall the CPU for the
        whole in-flight backlog (the GPU runs behind the faster CPU once recording is cheap)."""
        if v in self._batch_inflight or v in self._batch_pending:
            self._repl[2] += 1
            return False
        if self._cb_active:
            # a fresh unit still being recorded must stay BEFORE this resubmit in the queue:
            # its cb is only appended when the NEXT unit starts, so without this end the
            # template would run ahead of it and break data dependencies (observed: NaN
            # logits -> argmax == vocab size). Ending it here keeps pending order == program
            # order for the mixed fresh/template sequence.
            self._batch_end_unit()
        self._batch_pending.append(v)
        if len(self._batch_pending) >= self._stream_chunk:
            self._stream_flush(30_000_000_000)
        return True

    def submit(self, wait:bool=False, timeout_ms:int|None=None):
        """End the pending command buffer and submit it with its ring slot's fence; a
        no-op when nothing is pending. wait=False returns immediately (the work is in
        flight); wait=True blocks until the just-submitted fence signals (timeout_ms,
        default 30 s; a timeout raises VkError so e.g. a slow BEAM candidate is skipped
        instead of blocking forever). On drivers that need it (self._async is False,
        e.g. RADV on this APU) every submit waits on its fence: the next in-order
        dispatch must not start before this submit's stores are visible, so the ring
        never runs deeper than one. In VK_BATCH_SEM this instead flushes the accumulated
        per-unit command buffers as one semaphore-chained submit (see _batch_submit); in
        VK_STREAM it flushes them as cross-submit-chained chunk submits (see _stream_submit)."""
        if self._batch_sem:
            return self._batch_submit(wait, timeout_ms)
        if self._stream:
            return self._stream_submit(wait, timeout_ms)
        if not self._cb_active:
            return VK_SUCCESS
        timeout_ns = (timeout_ms if timeout_ms is not None else 30000) * 1_000_000
        i = self._slot
        _check("vkEndCommandBuffer", vkEndCommandBuffer(self.cbs[i]))
        if not self._inflight[i]:  # fence was signaled by its previous use; reset before reuse
            _check("vkResetFences", vkResetFences(self.device, 1, C.cast((c_void_p * 1)(self.fences[i]), c_void_p)))
        cbs1 = (c_void_p * 1)(self.cbs[i])
        si = VkSubmitInfo(sType=ST_SUBMIT_INFO, commandBufferCount=1, pCommandBuffers=C.cast(cbs1, c_void_p))
        # NOTE: these drivers (NV 550, RADV) treat the pFence arg as the fence object itself;
        # a true VkFence* (pointer to a slot holding the handle) makes both of them crash
        _check("vkQueueSubmit", vkQueueSubmit(self.queue, 1, C.byref(si), self.fences[i]))
        # update the ring state BEFORE the (potentially raising) wait: a timed-out or
        # failing wait must leave the ring consistent - the cb is submitted, the slot
        # advances, and the fence is tracked as in-flight until _wait_fence clears it
        # (_ensure_cb's backpressure waits it out before the cb is re-recorded)
        self._inflight[i] = True
        self._slot = (i + 1) % RING
        self._cb_active = False
        if wait or not self._async: self._wait_fence(i, timeout_ns)
        return VK_SUCCESS

    def synchronize(self, timeout_ms:int|None=None):
        """Flush the pending command buffer (if any), then wait for every in-flight
        submit (all pending fences in the ring)."""
        if self._batch_sem:
            # the whole pending batch is one submit on fence 0; flush it and wait on it
            self._batch_submit(wait=True, timeout_ms=timeout_ms)
            return
        if self._stream:
            # flush the rest as a final chunk, then wait on every in-flight chunk fence
            self._stream_synchronize(timeout_ms)
            return
        timeout_ns = (timeout_ms if timeout_ms is not None else 30000) * 1_000_000
        self.submit(wait=False)
        for i in range(RING):
            if self._inflight[i]: self._wait_fence(i, timeout_ns)

    # -- teardown --

    def close(self):
        if not getattr(self, "device", None):
            return
        self.submit(wait=False)  # a pending (unsubmitted) command buffer would be dropped
        _check("vkDeviceWaitIdle", vkDeviceWaitIdle(self.device))
        for b in self._buffers:
            if b._mapped is not None:
                vkUnmapMemory(self.device, b.mem)
            vkDestroyBuffer(self.device, b.handle, None)
            vkFreeMemory(self.device, b.mem, None)
        for m in self._modules:
            vkDestroyShaderModule(self.device, m.handle, None)
        for p in self._pipelines:
            vkDestroyPipeline(self.device, p.handle, None)
            if p.layout:
                vkDestroyPipelineLayout(self.device, p.layout, None)
        for d in self._dsls:
            vkDestroyDescriptorPool(self.device, d.pool, None)
            vkDestroyDescriptorSetLayout(self.device, d.layout, None)
        vkFreeCommandBuffers(self.device, self.cmd_pool, self._cb_total, C.cast(self.cbs, c_void_p))
        vkDestroyCommandPool(self.device, self.cmd_pool, None)
        if self._batch_sem or self._stream:
            for i in range(POOL):
                vkDestroySemaphore(self.device, self._batch_sems[i], None)
            if self._stream:
                for i in range(2):
                    vkDestroySemaphore(self.device, self._stream_sems[i], None)
        for i in range(RING):
            vkDestroyFence(self.device, self.fences[i], None)
        vkDestroyDevice(self.device, None)
        vkDestroyInstance(self.instance, None)
        self.device = c_void_p(None)
