# Vulkan backend

Runs tinygrad on any Vulkan compute device: NVIDIA / AMD / Intel GPUs on Linux, and
Qualcomm Adreno on Android (Termux).

Two parts are needed:

1. **Vulkan loader + device driver** — `libvulkan.so.1` plus the vendor ICD. Talks to the GPU.
2. **SPIR-V compiler library** — the VULKAN backend compiles kernels through mesa zink's
   NIR->SPIR-V exporter: `libtinyzink.so`, linked against the NIR runtime `libtinymesa`.
   Built from `extra/vulkan/`.

## Linux (NVIDIA / AMD / Intel)

```bash
git clone https://github.com/tinygrad/tinygrad && cd tinygrad
pip install -e .
pip install tinymesa==25.2.7.2
bash extra/vulkan/build_libtinyzink.sh     # -> extra/vulkan/libtinyzink.so (found automatically)
```

Install the Vulkan loader (`libvulkan1`) and the driver ICD for your GPU
(Ubuntu/Debian package names; use your distro's equivalent):

| GPU  | Driver package               | Run                                                            |
|------|------------------------------|----------------------------------------------------------------|
| NV   | `sudo apt install nvidia-driver-XXX` | `DEV=VULKAN python examples/beautiful_mnist.py`         |
| AMD  | `sudo apt install mesa-vulkan-drivers`  | `DEV=VULKAN python examples/beautiful_mnist.py`         |
| Intel| `sudo apt install mesa-vulkan-drivers`  | `DEV=VULKAN VK_VENDOR=8086 python examples/beautiful_mnist.py` |

Check the loader sees your GPU: `vulkaninfo --summary` (from `vulkan-tools`).

- `VK_VENDOR` — space-separated hex vendor IDs to select; default `1002 10de`
  (AMD + NVIDIA), so both work with no flags. Intel is `8086`, hence the flag.
- `VK_DEVICE_INDEX` — which matching device to use (default 0), for multi-GPU boxes.

## Android / Termux (Adreno)

The tinymesa PyPI wheels are glibc builds and won't load in Termux, so build both
libraries natively on the device:

```bash
# on the device, in Termux
git clone https://github.com/tinygrad/tinygrad && cd tinygrad
pkg install python
bash extra/vulkan/termux/build_termux.sh   # -> $HOME/zink_build/libtinymesa.so + libtinyzink.so
```

Run (this tablet: Adreno, vendor `0x5143`; `$HOME` = `/data/data/com.termux/files/home`):

```bash
MESA_PATH=$HOME/zink_build/libtinymesa.so TINYGRAD_LIBTINYZINK=$HOME/zink_build/libtinyzink.so \
  DEV=VULKAN VULKAN_LOADER=/system/lib64/libvulkan.so VK_ARCH=vk5143 VK_VENDOR=5143 \
  PYTHONPATH=$PWD python examples/beautiful_mnist_smalltensors.py
```

CPU fallback on the same device (no vulkan libraries needed):

```bash
LIBC_PATH=/system/lib64/libc.so DEV=CPU PYTHONPATH=$PWD python examples/beautiful_mnist_smalltensors.py
```

- `VULKAN_LOADER` is required on Android: the loader lives at `/system/lib64/libvulkan.so`,
  outside the default search paths.
- `VK_ARCH` selects the renderer/limits profile (default derives from the vendor,
  `vk5143` for Qualcomm). Set it explicitly when compiling on a desktop box for the device.
- **Use `examples/beautiful_mnist_smalltensors.py` on Adreno, not `beautiful_mnist.py`**:
  the Adreno driver has a 256 MiB per-buffer limit (see `extra/vulkan/ADRENO_256MB.md`)
  that a full-batch test eval exceeds. The smalltensors variant chunks the eval;
  reported accuracy is identical (`EVAL_BS` tunes the chunk size).

## Env var reference

| Var                  | Meaning                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `VK_VENDOR`          | space-separated hex vendor IDs; default `1002 10de` (Intel 8086, Adreno 5143) |
| `VK_DEVICE_INDEX`    | which matching device (default 0)                                        |
| `VK_ARCH`            | renderer/limits override (default `radv` on AMD, else `vk<vendor>`)      |
| `VULKAN_LOADER`      | explicit path to the Vulkan loader (needed on Android)                   |
| `TINYGRAD_LIBTINYZINK` | path to `libtinyzink.so` (default `extra/vulkan/libtinyzink.so`)       |
| `MESA_PATH`          | path to `libtinymesa.so` (Termux build; on desktop it comes from the pip wheel) |
| `VK_MAX_BUFFER`      | per-buffer size cap override (0 = unlimited; Adreno default 256 MiB)     |
| `EVAL_BS`            | smalltensors eval chunk size (default 1000)                              |
