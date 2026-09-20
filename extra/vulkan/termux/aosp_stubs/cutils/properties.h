/* AOSP stub for Termux (mesa NIR build). property system -> empty. */
#ifndef CUTILS_PROPERTIES_H
#define CUTILS_PROPERTIES_H
#define PROPERTY_KEY_MAX 32
#define PROPERTY_VALUE_MAX 92
static inline int property_get(const char *key, char *value, const char **success) {
  (void)key; (void)success; if (value) value[0] = "\0"[0]; return 0;
}
#endif
