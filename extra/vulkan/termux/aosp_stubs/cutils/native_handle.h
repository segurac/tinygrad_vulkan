/* AOSP stub for Termux (mesa NIR build). */
#ifndef CUTILS_NATIVE_HANDLE_H
#define CUTILS_NATIVE_HANDLE_H
struct native_handle {
  int version;   /* sizeof(native_handle_t) */
  int numFds;    /* number of file-descriptors at &data[0] */
  int numInts;   /* number of ints at &data[numFds] */
  int data[1];   /* numFds + numInts ints */
};
typedef struct native_handle native_handle_t;
#endif
