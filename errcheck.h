#ifndef ERRCHECK_H
  #define ERRCHECK_H

  // make c++ friendly
  #ifdef __cplusplus
    extern "C" {
  #endif

  /*
   *  Libraries
   */

  #include <stdio.h>
  #include <errno.h>
  #include <string.h>

  #include <cuda_runtime.h>

  /*
   *  Macros
   */

  #define CPU_ERROR 1
  #define GPU_ERROR 2

  #ifdef NO_ERRCHECK
    #define errcheck_cpu() {}
    #define errcheck_gpu() {}
  #else
    // check errno, clean up and exit if non-zero
    #define errcheck_cpu()  if (errno) { \
                              fprintf(stderr, __FILE__ " %d: %s\n", __LINE__, strerror(errno)); \
                              exit(CPU_ERROR); \
                            }

    // check cudaerr, exit if non-zero
    #define errcheck_gpu()  cudaThreadSynchronize(); cuda_errno = cudaGetLastError(); \
                            if (cuda_errno != cudaSuccess) { \
                              fprintf(stderr, __FILE__ " %d: %s\n", __LINE__, cudaGetErrorString(cuda_errno)); \
                              exit(GPU_ERROR); \
                            }
  #endif

  // cuda error holder
  extern cudaError_t cuda_errno;

  #ifdef __cplusplus
    }
  #endif

#endif
