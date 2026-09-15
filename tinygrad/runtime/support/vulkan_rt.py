"""Minimal pure-ctypes Vulkan 1.2 compute runtime (loader: libvulkan.so.1).

Struct layouts match Vulkan-Headers v1.4.305 (extra/vulkan/include/vulkan/vulkan_core.h).
Only what a compute device needs: one queue family, buffers (mapped or
device-local + staging), a ring of (command buffer, fence) pairs so submits can be
async and pipelined, compute pipelines, storage/uniform buffer descriptors.
"""
import ctypes as C
import glob
import os
import struct
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
vkCmdBindPipeline = _vk("vkCmdBindPipeline", _v + [c_uint32] + _v, _N)
vkCmdBindDescriptorSets = _vk("vkCmdBindDescriptorSets", _v + [c_uint32] + _v + [c_uint32, c_uint32] + _v + [c_uint32] + _v, _N)
vkCmdDispatch = _vk("vkCmdDispatch", _v + [c_uint32, c_uint32, c_uint32], _N)
vkCreateFence = _vk("vkCreateFence", _v + _v + _v + [C.POINTER(c_void_p)])
vkDestroyFence = _vk("vkDestroyFence", _v + _v + _v, _N)
vkResetFences = _vk("vkResetFences", _v + [c_uint32] + _v)
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
vkCreateDescriptorPool = _vk("vkCreateDescriptorPool", _v + _v + _v + [C.POINTER(c_void_p)])
vkDestroyDescriptorPool = _vk("vkDestroyDescriptorPool", _v + _v + _v, _N)
vkAllocateDescriptorSets = _vk("vkAllocateDescriptorSets", _v + _v + [C.POINTER(c_void_p)])
vkUpdateDescriptorSets = _vk("vkUpdateDescriptorSets", _v + [c_uint32] + 2 * _v, _N)

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
        cba = VkCommandBufferAllocateInfo(sType=ST_COMMAND_BUFFER_ALLOCATE_INFO, commandPool=pool,
                                           level=VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=RING)
        cbs = (c_void_p * RING)()
        _check("vkAllocateCommandBuffers", vkAllocateCommandBuffers(dev, C.byref(cba), cbs))
        self.cbs, self._cb_active = cbs, False
        # self._slot is the ring slot the active command buffer belongs to (or the next one
        # to begin). self._inflight[i]: fence i was submitted and has not been waited on.
        self._slot, self._inflight = 0, [False] * RING

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

    # -- command recording / submit --

    def _wait_fence(self, i, timeout_ns=30_000_000_000):
        _check("vkWaitForFences", vkWaitForFences(self.device, 1, C.cast((c_void_p * 1)(self.fences[i]), c_void_p),
                                                  1, timeout_ns))
        self._inflight[i] = False

    def _ensure_cb(self):
        if not self._cb_active:
            # backpressure: wrapping back to a slot whose submit is still in flight; its
            # command buffer must be idle before it can be reset and re-recorded
            if self._inflight[self._slot]: self._wait_fence(self._slot)
            _check("vkResetCommandBuffer", vkResetCommandBuffer(self.cbs[self._slot], 0))
            bi = VkCommandBufferBeginInfo(sType=ST_COMMAND_BUFFER_BEGIN_INFO)
            _check("vkBeginCommandBuffer", vkBeginCommandBuffer(self.cbs[self._slot], C.byref(bi)))
            self._cb_active = True

    def cmd_copy(self, dst, src, size=None, src_off=0, dst_off=0):
        n = size if size is not None else min(dst.size, src.size)
        self._ensure_cb()
        cp = VkBufferCopy(srcOffset=src_off, dstOffset=dst_off, size=n)
        vkCmdCopyBuffer(self.cbs[self._slot], _handle(src), _handle(dst), 1, C.byref(cp))

    def cmd_bind_pipeline(self, pipeline):
        self._ensure_cb()
        vkCmdBindPipeline(self.cbs[self._slot], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle)

    def cmd_bind_descriptor_sets(self, pipeline_or_layout, desc_set):
        layout = pipeline_or_layout.layout if hasattr(pipeline_or_layout, "layout") else pipeline_or_layout
        self._ensure_cb()
        # pDescriptorSets is a pointer to an array of set handles, not the handle itself
        sets = (c_void_p * 1)(_handle(desc_set))
        vkCmdBindDescriptorSets(self.cbs[self._slot], VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                                0, 1, sets, 0, None)

    def cmd_dispatch(self, x, y, z):
        self._ensure_cb()
        vkCmdDispatch(self.cbs[self._slot], x, y, z)

    def submit(self, wait:bool=False):
        """End the recorded command buffer and submit it with its ring slot's fence.
        wait=False returns immediately (the work is in flight); wait=True blocks until
        the just-submitted fence signals (a full queue drain of this slot's work).
        On drivers that need it (self._async is False, e.g. RADV on this APU) every
        submit waits on its fence: the next in-order dispatch must not start before
        this submit's stores are visible, so the ring never runs deeper than one."""
        if not self._cb_active:
            return VK_SUCCESS
        i = self._slot
        _check("vkEndCommandBuffer", vkEndCommandBuffer(self.cbs[i]))
        if not self._inflight[i]:  # fence was signaled by its previous use; reset before reuse
            _check("vkResetFences", vkResetFences(self.device, 1, C.cast((c_void_p * 1)(self.fences[i]), c_void_p)))
        cbs1 = (c_void_p * 1)(self.cbs[i])
        si = VkSubmitInfo(sType=ST_SUBMIT_INFO, commandBufferCount=1, pCommandBuffers=C.cast(cbs1, c_void_p))
        # NOTE: these drivers (NV 550, RADV) treat the pFence arg as the fence object itself;
        # a true VkFence* (pointer to a slot holding the handle) makes both of them crash
        _check("vkQueueSubmit", vkQueueSubmit(self.queue, 1, C.byref(si), self.fences[i]))
        self._inflight[i] = True
        if wait or not self._async: self._wait_fence(i)
        self._slot = (i + 1) % RING
        self._cb_active = False
        return VK_SUCCESS

    def synchronize(self, timeout_ms:int|None=None):
        """Wait for every in-flight submit (all pending fences in the ring)."""
        timeout_ns = (timeout_ms if timeout_ms is not None else 30000) * 1_000_000
        for i in range(RING):
            if self._inflight[i]: self._wait_fence(i, timeout_ns)

    # -- teardown --

    def close(self):
        if not getattr(self, "device", None):
            return
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
        vkFreeCommandBuffers(self.device, self.cmd_pool, RING, C.cast(self.cbs, c_void_p))
        vkDestroyCommandPool(self.device, self.cmd_pool, None)
        for i in range(RING):
            vkDestroyFence(self.device, self.fences[i], None)
        vkDestroyDevice(self.device, None)
        vkDestroyInstance(self.instance, None)
        self.device = c_void_p(None)
