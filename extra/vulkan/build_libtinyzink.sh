#!/bin/bash
# Build libtinyzink.so -- mesa zink's NIR->SPIR-V exporter as a tinygrad compiler backend.
#
# Sources: the two zink exporter .c files from mesa 25.2.7 (fetched from gitlab,
# cached in .cache/), the tinygrad patch (mesa_tinygrad.patch), and tinyzink_api.c.
# Runtime: the tinymesa wheel (pip install tinymesa==25.2.7.2) provides all NIR symbols.
#
# Requires: gcc, curl, python3 + mako (pip install mako) for the header generators,
# and the tinymesa wheel (pip install tinymesa==25.2.7.2) for the NIR runtime.
#
# Usage:
#   bash extra/vulkan/build_libtinyzink.sh
#
# Env overrides:
#   MESA_SRC_DIR   use an existing mesa 25.2.7 checkout instead of downloading
#   PYTHON         interpreter that has `tinymesa` installed (default: python3)
#   TINYGRAD_LIBTINYZINK is read at runtime by tinygrad (not here)
#
# CRITICAL: the -D flags below must match tinygrad's mesa autogen set exactly,
# otherwise struct layouts mismatch the tinymesa wheel ABI (the final ABI check
# fails with sizeof(struct nir_shader) != 520).
set -e
cd "$(dirname "$0")"

MESA_TAG=mesa-25.2.7
MESA_URL="https://gitlab.freedesktop.org/mesa/mesa/-/archive/${MESA_TAG}/${MESA_TAG}.tar.gz"
PY="${PYTHON:-python3}"
case "$PY" in /*) ;; *) PY="$(cd ../.. && pwd)/$PY" ;; esac  # relative to the repo root

# 1. mesa sources
if [ -n "$MESA_SRC_DIR" ]; then
  MESADIR="$MESA_SRC_DIR"
else
  MESADIR=".cache/${MESA_TAG}"
  if [ ! -d "${MESADIR}/src" ]; then
    mkdir -p .cache
    echo "fetching ${MESA_URL}"
    curl -fsSL "$MESA_URL" | tar xz -C .cache
    # gitlab archives extract to a commit-suffixed dir name; normalize it
    real=$(ls -d .cache/mesa-mesa-* 2>/dev/null | head -1)
    [ -n "$real" ] && mv "$real" "$MESADIR"
  fi
fi

# 2. tinygrad patch (idempotent)
SPIRV_C="${MESADIR}/src/gallium/drivers/zink/nir_to_spirv/nir_to_spirv.c"
if ! grep -q "tinygrad patch" "$SPIRV_C"; then
  patch -p1 -d "$MESADIR" --fuzz=0 < mesa_tinygrad.patch
fi

# 3. generated headers (idempotent)
if [ ! -f "${MESADIR}/gen/nir_intrinsics.h" ] || [ ! -f "${MESADIR}/gen/util/format/u_format_gen.h" ] \
   || [ ! -f "${MESADIR}/gen/vk_struct_type_cast.h" ]; then
  ( cd "$MESADIR" && mkdir -p gen/util/format \
    && "$PY" src/util/format/u_format_table.py src/util/format/u_format.yaml --enums > gen/util/format/u_format_gen.h \
    && "$PY" src/compiler/builtin_types_h.py gen/builtin_types.h \
    && "$PY" src/compiler/nir/nir_intrinsics_h.py --outdir gen \
    && "$PY" src/compiler/nir/nir_intrinsics_indices_h.py --outdir gen \
    && "$PY" src/compiler/nir/nir_opcodes_h.py > gen/nir_opcodes.h \
    && "$PY" src/compiler/nir/nir_builder_opcodes_h.py > gen/nir_builder_opcodes.h \
    && "$PY" src/vulkan/util/vk_dispatch_table_gen.py --xml src/vulkan/registry/vk.xml --out-c gen/vk_dispatch_table.c --out-h gen/vk_dispatch_table.h --beta false \
    && "$PY" src/vulkan/util/gen_enum_to_str.py --xml src/vulkan/registry/vk.xml --outdir gen --beta false \
    && "$PY" src/vulkan/util/vk_struct_type_cast_gen.py --xml src/vulkan/registry/vk.xml --outdir gen --beta false \
    && "$PY" src/vulkan/util/vk_extensions_gen.py --xml src/vulkan/registry/vk.xml --out-c gen/vk_extensions.c --out-h gen/vk_extensions.h \
    && "$PY" src/gallium/drivers/zink/zink_device_info.py gen/zink_device_info.h gen/zink_device_info.c src/vulkan/registry/vk.xml \
    && "$PY" src/gallium/drivers/zink/zink_instance.py gen/zink_instance.h gen/zink_instance.c src/vulkan/registry/vk.xml )
fi

# 4. tinymesa wheel (NIR runtime)
TMESA_DIR=$("$PY" -c "import tinymesa,os;print(os.path.dirname(tinymesa.__file__))" 2>/dev/null) \
  || { echo "error: tinymesa not installed -- pip install tinymesa==25.2.7.2"; exit 1; }

# 5. compile + link (flags MUST match tinygrad's mesa autogen: see CRITICAL above)
INCS="-DHAVE_ENDIAN_H -DHAVE_STRUCT_TIMESPEC -DHAVE_PTHREAD -DHAVE_FUNC_ATTRIBUTE_PACKED \
 -I${MESADIR}/src -I${MESADIR}/include -I${MESADIR}/gen -I${MESADIR}/src/compiler -I${MESADIR}/src/compiler/nir \
 -I${MESADIR}/src/gallium/include -I${MESADIR}/src/gallium/auxiliary -I${MESADIR}/src/vulkan/util -I${MESADIR}/src/vulkan \
 -I${MESADIR}/src/gallium/drivers/zink -I${MESADIR}/src/gallium/drivers/zink/nir_to_spirv -Iinclude"

gcc -O2 -fPIC -c $INCS "$MESADIR/src/gallium/drivers/zink/nir_to_spirv/nir_to_spirv.c" -o nir_to_spirv.o
gcc -O2 -fPIC -c $INCS "$MESADIR/src/gallium/drivers/zink/nir_to_spirv/spirv_builder.c" -o spirv_builder.o
gcc -O2 -fPIC -c $INCS tinyzink_api.c -o tinyzink_api.o
gcc -shared -o libtinyzink.so nir_to_spirv.o spirv_builder.o tinyzink_api.o \
  -L"$TMESA_DIR" -ltinymesa -Wl,-rpath,"$TMESA_DIR"
rm -f nir_to_spirv.o spirv_builder.o tinyzink_api.o

# 6. ABI sanity check against the tinymesa wheel
SZ=$(printf '#include "nir.h"\n#include <stdio.h>\nint main(){printf("%%zu",sizeof(struct nir_shader));}' \
  | gcc -x c - -o /tmp/tinyzink_abi_check $INCS -L"$TMESA_DIR" -ltinymesa - && /tmp/tinyzink_abi_check)
[ "$SZ" = "520" ] || { echo "error: sizeof(struct nir_shader)=$SZ, expected 520 -- preprocessor flags drifted from the tinymesa wheel ABI"; exit 1; }
rm -f /tmp/tinyzink_abi_check

echo "built $(pwd)/libtinyzink.so (sizeof(nir_shader)=$SZ, mesa=${MESADIR}, tinymesa=${TMESA_DIR})"
