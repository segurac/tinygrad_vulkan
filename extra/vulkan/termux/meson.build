# Lean NIR-only target for the Termux on-device build.
#
# We only build the NIR runtime (libtinymesa.so) -- the freedreno ir3/common
# drivers and NAK from the upstream tinymesa target are not needed, because our
# 3 C files (zink nir_to_spirv / spirv_builder + the tinygrad export shim) only
# consume the NIR runtime.
#
# _libnir, idep_nir, idep_compiler and idep_mesautil are NOT defined here: they
# are provided by the enclosing mesa build scope. The tinymesa recipe appends
# subdir('tinymesa') to src/meson.build, and src/compiler/nir/meson.build defines
# _libnir, so all of these resolve in this (nested) scope. Do NOT add
# import('nir').nir here -- newer meson parsers reject that attribute access.
shared_library(
  'tinymesa',
  link_whole : [ _libnir ],
  dependencies : [ idep_nir, idep_compiler, idep_mesautil ],
)
