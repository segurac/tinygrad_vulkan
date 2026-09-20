/* AOSP stub for Termux (mesa NIR build). atrace -> no-op. */
#ifndef CUTILS_TRACE_H
#define CUTILS_TRACE_H
#include <stdint.h>
#define ATRACE_TAG_GRAPHICS 0x01
static inline int atrace_begin(uint32_t tag, const char *fmt, ...) { (void)tag; (void)fmt; return 0; }
static inline int atrace_end(uint32_t tag) { (void)tag; return 0; }
static inline int atrace_init(void) { return 0; }
#endif
