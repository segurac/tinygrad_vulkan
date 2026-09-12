#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include "util/blob.h"
#include "util/ralloc.h"
#include "nir.h"
#include "nir_serialize.h"
#include "glsl_types.h"
#include "zink_shader_keys.h"
#include "zink_types.h"
#include "nir_to_spirv.h"

static int types_inited;

int tinyzink_nirblob_to_spirv(const uint8_t *blob, size_t blob_size,
                              uint32_t **out_words, size_t *out_num_words)
{
   if (!blob || blob_size == 0 || !out_words || !out_num_words) {
      fprintf(stderr, "tinyzink_nirblob_to_spirv: invalid arguments\n");
      return -1;
   }
   if (!types_inited) {
      glsl_type_singleton_init_or_ref();
      types_inited = 1;
   }

   struct blob_reader reader;
   blob_reader_init(&reader, blob, blob_size);

   struct nir_shader_compiler_options opts;
   memset(&opts, 0, sizeof(opts));

   struct nir_shader *s = nir_deserialize(NULL, &opts, &reader);
   if (!s) {
      fprintf(stderr, "tinyzink_nirblob_to_spirv: nir_deserialize failed\n");
      return -1;
   }

   struct zink_shader_info si;
   memset(&si, 0, sizeof(si));
   struct zink_screen *screen = calloc(1, sizeof(*screen));
   if (!screen) {
      fprintf(stderr, "tinyzink_nirblob_to_spirv: out of memory\n");
      ralloc_free(s);
      return -1;
   }
   screen->spirv_version = 0x00010300; /* SPIR-V 1.3 */
   /* leave si.have_EXT_shader_demote_to_helper_invocation = 0 (fragment-only) */

   struct spirv_shader *sp = nir_to_spirv(s, &si, screen);
   if (!sp) {
      fprintf(stderr, "tinyzink_nirblob_to_spirv: nir_to_spirv failed\n");
      free(screen);
      ralloc_free(s);
      return -1;
   }

   uint32_t *words = malloc(sp->num_words * sizeof(uint32_t));
   if (!words) {
      fprintf(stderr, "tinyzink_nirblob_to_spirv: out of memory\n");
      spirv_shader_delete(sp);
      free(screen);
      ralloc_free(s);
      return -1;
   }
   memcpy(words, sp->words, sp->num_words * sizeof(uint32_t));

   *out_words = words;
   *out_num_words = sp->num_words;

   spirv_shader_delete(sp);
   free(screen);
   ralloc_free(s);
   return 0;
}

void tinyzink_free_words(uint32_t *words)
{
   free(words);
}
