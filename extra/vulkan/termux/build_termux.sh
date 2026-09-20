#!/bin/bash
# Build libtinymesa.so (NIR runtime) + libtinyzink.so (NIR->SPIR-V) NATIVELY in
# Termux so the Android tablet compiles SPIR-V on-device (no pre-dumped .spv).
#
# The tinymesa PyPI wheels are glibc builds and will not load in Termux, so the
# NIR runtime must be built against Termux's own libc. This script does that,
# then builds the 3-C-file zink exporter (nir_to_spirv / spirv_builder + the
# tinygrad export shim) against the freshly built NIR runtime.
#
# ISOLATED from the x86 server build (../build_libtinyzink.sh): it uses its own
# mesa checkout + output dir and does NOT touch any repo .py files, the x86
# build script, or the x86 mesa cache. APU / GTX 3060 are unaffected.
#
# Output:
#   $OUT/libtinymesa.so    (NIR runtime)        -> run with MESA_PATH=...
#   $OUT/libtinyzink.so    (NIR->SPIR-V export) -> run with TINYGRAD_LIBTINYZINK=...
#
# Run on-device (Adreno 710 = vk5143; adjust VK_ARCH/VK_VENDOR for other Adreno):
#   MESA_PATH=$OUT/libtinymesa.so TINYGRAD_LIBTINYZINK=$OUT/libtinyzink.so \
#     DEV=VULKAN VULKAN_LOADER=/system/lib64/libvulkan.so VK_ARCH=vk5143 VK_VENDOR=5143 \
#     python -c "from tinygrad import Tensor; (Tensor.ones(64,64)@Tensor.ones(64,64)).relu().sum().realize(); print('ok')"
#
# Env overrides: MESASRCDIR TINYMESA_DIR OUT PYTHON CC
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"          # .../extra/vulkan/termux
VK="$(cd "$HERE/.." && pwd)"                    # .../extra/vulkan (shared files)
H="${TERMUX_HOME:-$HOME}"
MESASRCDIR="${MESASRCDIR:-$H/mesa_build/mesa-25.2.7}"
TINYMESA_DIR="${TINYMESA_DIR:-$H/tinymesa}"
TINYMESA_COMMIT="${TINYMESA_COMMIT:-42987d68328116a94b947295fd452c1caaba882f}"
OUT="${OUT:-$H/zink_build}"
TMPD="${TMPD:-$H/.termux_build_tmp}"   # /tmp is not writable in Termux; use $HOME
mkdir -p "$TMPD"
MESA_TAG=mesa-25.2.7
MESA_URL="https://gitlab.freedesktop.org/mesa/mesa/-/archive/${MESA_TAG}/${MESA_TAG}.tar.gz"
PY="${PYTHON:-python3}"
CC="${CC:-cc}"
die(){ echo "ERROR: $*" >&2; exit 1; }

echo "==> [1/9] deps (pkg: ninja zlib patch; pip: meson==1.8.0 mako pyyaml packaging distlib)"
pkg install -y ninja zlib patch >/dev/null 2>&1 || true
# meson >= 1.12 changed how `import('nir').nir` module attribute access is parsed and
# breaks the tinymesa target; pin to 1.8.x (the version the build was validated on).
$PY -m pip install --quiet "meson==1.8.0" mako pyyaml packaging distlib >/dev/null 2>&1 || true

echo "==> [2/9] mesa ${MESA_TAG} sources"
if [ ! -f "$MESASRCDIR/meson.build" ]; then
  mkdir -p "$H/mesa_build" && cd "$H/mesa_build"
  [ -f "${MESA_TAG}.tar.gz" ] || curl -fsSL -o "${MESA_TAG}.tar.gz" "$MESA_URL"
  tar xzf "${MESA_TAG}.tar.gz"
  real=$(ls -d mesa-mesa-* 2>/dev/null | head -1); [ -n "$real" ] && mv "$real" "$MESASRCDIR"
fi
[ -f "$MESASRCDIR/meson.build" ] || die "mesa sources not found at $MESASRCDIR"
cd "$MESASRCDIR"

echo "==> [3/9] tinygrad zink patch + generated headers"
if ! grep -q "tinygrad patch" src/gallium/drivers/zink/nir_to_spirv/nir_to_spirv.c; then
  patch -p1 --fuzz=0 < "$VK/mesa_tinygrad.patch" || die "mesa_tinygrad.patch failed"
fi
if [ ! -f gen/nir_intrinsics.h ] || [ ! -f gen/util/format/u_format_gen.h ] || [ ! -f gen/vk_struct_type_cast.h ]; then
  mkdir -p gen/util/format
  $PY src/util/format/u_format_table.py src/util/format/u_format.yaml --enums > gen/util/format/u_format_gen.h
  $PY src/compiler/builtin_types_h.py gen/builtin_types.h
  $PY src/compiler/nir/nir_intrinsics_h.py --outdir gen
  $PY src/compiler/nir/nir_intrinsics_indices_h.py --outdir gen
  $PY src/compiler/nir/nir_opcodes_h.py > gen/nir_opcodes.h
  $PY src/compiler/nir/nir_builder_opcodes_h.py > gen/nir_builder_opcodes.h
  $PY src/vulkan/util/vk_dispatch_table_gen.py --xml src/vulkan/registry/vk.xml --out-c gen/vk_dispatch_table.c --out-h gen/vk_dispatch_table.h --beta false
  $PY src/vulkan/util/gen_enum_to_str.py --xml src/vulkan/registry/vk.xml --outdir gen --beta false
  $PY src/vulkan/util/vk_struct_type_cast_gen.py --xml src/vulkan/registry/vk.xml --outdir gen --beta false
  $PY src/vulkan/util/vk_extensions_gen.py --xml src/vulkan/registry/vk.xml --out-c gen/vk_extensions.c --out-h gen/vk_extensions.h
  $PY src/gallium/drivers/zink/zink_device_info.py gen/zink_device_info.h gen/zink_device_info.c src/vulkan/registry/vk.xml
  $PY src/gallium/drivers/zink/zink_instance.py gen/zink_instance.h gen/zink_instance.c src/vulkan/registry/vk.xml
fi

echo "==> [4/9] tinymesa recipe (sirhcm/tinymesa @ ${TINYMESA_COMMIT:0:7}) + lean NIR-only target"
cur="$(git -C "$TINYMESA_DIR" rev-parse HEAD 2>/dev/null)"
if [ "$cur" != "$TINYMESA_COMMIT" ]; then
  rm -rf "$TINYMESA_DIR"
  git clone --quiet https://github.com/sirhcm/tinymesa "$TINYMESA_DIR" || die "git clone tinymesa failed"
  git -C "$TINYMESA_DIR" checkout --quiet "$TINYMESA_COMMIT"
fi
# apply_patches.py (commit 42987d6) takes MESA_DIR MESA_TAG and is not re-runnable, so guard on it.
[ -d src/tinymesa ] || $PY "$TINYMESA_DIR/apply_patches.py" "$MESASRCDIR" "$MESA_TAG" || die "apply_patches.py failed"
cp "$HERE/meson.build" src/tinymesa/meson.build

echo "==> [5/9] Termux source patches (memfd_create + getrandom -> raw syscall)"
$PY - "$MESASRCDIR" << 'PYEOF' || die "termux source patch failed"
import sys
M = sys.argv[1]
def patch(path, old, new):
    t = open(path).read()
    if new in t:
        print("  already patched", path); return
    assert old in t, f"pattern not found in {path}"
    open(path, "w").write(t.replace(old, new, 1)); print("  patched", path)
patch(M + "/src/util/anon_file.c",
      '#include "detect_os.h"\n\n#ifndef _WIN32',
      '#include "detect_os.h"\n\n#if defined(__ANDROID__) && defined(HAVE_MEMFD_CREATE)\n#undef HAVE_MEMFD_CREATE\n#endif\n\n#ifndef _WIN32')
patch(M + "/src/util/rand_xor.c",
      '#include "detect_os.h"\n\n#if !DETECT_OS_WINDOWS',
      '#include "detect_os.h"\n\n#if defined(__ANDROID__) && defined(HAVE_GETRANDOM)\n#undef HAVE_GETRANDOM\n#endif\n\n#if !DETECT_OS_WINDOWS')
PYEOF

echo "==> [6/9] AOSP stub headers (cutils/trace, cutils/properties, cutils/native_handle, log/log, android/log)"
PREFIX="$(getconf PREFIX 2>/dev/null || echo /data/data/com.termux/files/usr)"
mkdir -p "$PREFIX/include"
cp -rf "$HERE/aosp_stubs/." "$PREFIX/include/"
echo "    installed to $PREFIX/include"

echo "==> [7/9] meson setup + compile NIR runtime (libtinymesa.so)"
rm -rf "$MESASRCDIR/build"
meson setup "$MESASRCDIR/build" "$MESASRCDIR" \
  -Db_staticpic=true -Dplatforms="[]" -Dglx=disabled \
  -Dvulkan-drivers="" -Dgallium-drivers="" -Dopengl=false -Degl=disabled \
  -Dgbm=disabled -Dvideo-codecs="[]" || die "meson setup failed"
ninja -C "$MESASRCDIR/build" src/tinymesa/libtinymesa.so || die "ninja libtinymesa failed"
mkdir -p "$OUT"
cp "$MESASRCDIR/build/src/tinymesa/libtinymesa.so" "$OUT/libtinymesa.so"

echo "==> [8/9] libtinyzink.so (nir_to_spirv + spirv_builder + tinyzink_api -> NIR->SPIR-V)"
cd "$VK"   # so -Iinclude (vulkan headers) + tinyzink_api.c resolve
INCS="-DHAVE_ENDIAN_H -DHAVE_STRUCT_TIMESPEC -DHAVE_PTHREAD -DHAVE_FUNC_ATTRIBUTE_PACKED \
 -I${MESASRCDIR}/src -I${MESASRCDIR}/include -I${MESASRCDIR}/gen -I${MESASRCDIR}/src/compiler -I${MESASRCDIR}/src/compiler/nir \
 -I${MESASRCDIR}/src/gallium/include -I${MESASRCDIR}/src/gallium/auxiliary -I${MESASRCDIR}/src/vulkan/util -I${MESASRCDIR}/src/vulkan \
 -I${MESASRCDIR}/src/gallium/drivers/zink -I${MESASRCDIR}/src/gallium/drivers/zink/nir_to_spirv -Iinclude"
$CC -O2 -fPIC -c $INCS "$MESASRCDIR/src/gallium/drivers/zink/nir_to_spirv/nir_to_spirv.c" -o "$TMPD/nir_to_spirv.o" || die "compile nir_to_spirv"
$CC -O2 -fPIC -c $INCS "$MESASRCDIR/src/gallium/drivers/zink/nir_to_spirv/spirv_builder.c" -o "$TMPD/spirv_builder.o" || die "compile spirv_builder"
$CC -O2 -fPIC -c $INCS tinyzink_api.c -o "$TMPD/tinyzink_api.o" || die "compile tinyzink_api"
$CC -shared -o "$OUT/libtinyzink.so" "$TMPD/nir_to_spirv.o" "$TMPD/spirv_builder.o" "$TMPD/tinyzink_api.o" \
  -L"$OUT" -ltinymesa -Wl,-rpath,"$OUT" || die "link libtinyzink"
rm -f "$TMPD/nir_to_spirv.o" "$TMPD/spirv_builder.o" "$TMPD/tinyzink_api.o"

echo "==> [9/9] ABI check (sizeof(struct nir_shader) must be 520)"
SZ=$($CC -x c - -o "$TMPD/tinyzink_abi_check" $INCS -L"$OUT" -ltinymesa -Wl,-rpath,"$OUT" - \
  <<< '#include "nir.h"
#include <stdio.h>
int main(){printf("%zu",sizeof(struct nir_shader));}' && "$TMPD/tinyzink_abi_check")
rm -f "$TMPD/tinyzink_abi_check"
[ "$SZ" = "520" ] || die "sizeof(struct nir_shader)=$SZ, expected 520 (preprocessor flags drifted from the tinymesa ABI)"

echo
echo "DONE (sizeof(nir_shader)=$SZ)"
echo "  $OUT/libtinymesa.so   (NIR runtime)      MESA_PATH"
echo "  $OUT/libtinyzink.so   (NIR->SPIR-V)      TINYGRAD_LIBTINYZINK"
echo
echo "Run on-device:"
echo "  MESA_PATH=$OUT/libtinymesa.so TINYGRAD_LIBTINYZINK=$OUT/libtinyzink.so \\"
echo "    DEV=VULKAN VULKAN_LOADER=/system/lib64/libvulkan.so VK_ARCH=vk5143 VK_VENDOR=5143 \\"
echo "    python -c \"from tinygrad import Tensor; (Tensor.ones(64,64)@Tensor.ones(64,64)).relu().sum().realize(); print('ok')\""
