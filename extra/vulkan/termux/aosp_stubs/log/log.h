/* AOSP stub for Termux (mesa NIR build). android log -> no-op. */
#ifndef ANDROID_LOG_LOG_H
#define ANDROID_LOG_LOG_H
#include <stdarg.h>
typedef enum android_LogPriority {
  ANDROID_LOG_UNKNOWN = 0, ANDROID_LOG_DEFAULT = 1, ANDROID_LOG_VERBOSE = 2,
  ANDROID_LOG_DEBUG = 3, ANDROID_LOG_INFO = 4, ANDROID_LOG_WARN = 5,
  ANDROID_LOG_ERROR = 6, ANDROID_LOG_FATAL = 7, ANDROID_LOG_SILENT = 8
} android_LogPriority;
static inline int __android_log_print(int prio, const char *tag, const char *fmt, ...) {
  (void)prio; (void)tag; (void)fmt; return 0;
}
static inline int __android_log_vprint(int prio, const char *tag, const char *fmt, va_list ap) {
  (void)prio; (void)tag; (void)fmt; (void)ap; return 0;
}
static inline int __android_log_write(int prio, const char *tag, const char *msg) {
  (void)prio; (void)tag; (void)msg; return 0;
}
/* LOG_PRI -> __android_log_print */
#define LOG_PRI(prio, tag, fmt, ...) __android_log_print(prio, tag, fmt, ##__VA_ARGS__)
#endif
