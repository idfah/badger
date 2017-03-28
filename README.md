# Project Badger

## An experimental matrix library for optimized CPU and GPU computations.

### by Elliott Forney, 2010

During a graduate course on high-performance computing (cs675) we spent considerable time exploring and comparing different modern approaches for vectorizing certain problems in computer science.  In particular, we compared the relative performance of implementations that were highly tuned for traditional Intel CPU's versus implementations that were tuned for NVIDIA's GPU's using CUDA (at the time, CUDA was brand new).  For the CPU implementations, we used technologies like pthreads, OpenMP and SSE vectorization (both automatic and manual) and for the GPU implementations we explored various methods for distributing thread blocks and for "virtualizing" computations within thread blocks as well as different approaches for leveraging the GPU memory hierarchy (host, global, shared, texture and register).  Somewhat surprisingly, we discovered that many of the commonly cited benchmarks compare unoptimized CPU code to highly optimized GPU code, overstating the performance benefits of using GPU's.  Nevertheless, we found that GPU's are often 3-5x faster than CPU implementations (for problems that fit the SIMT model) and GPU's are also considerably cheaper per GFlop.

For my final project in cs675 I set out to implement a highly optimized artificial neural network in both C and CUDA.  Since artifical neural nets (ANN) can largely be implemented as matrix operations, I set out to create a matrix library that contained all of the necessary functions, a project that I call badger.

Badger contains an API for performing various matrix operations and manipulations.  This API is intended to be called from C.  Additionally, badger contains two backend libraries that a program built with the badger API can link against.  The first library, matrix_cpu, utilizes only the CPU and contains various OpenMP and SSE optimizations.  The only matrix operation that is not native to matrix_cpu is matrix multiply, for which we use a standard BLAS routine (ATLAS or MKL).  The second library, matrix_gpu, makes use of NVIDIA GPU's whenever possible.  One innovation of this library is that copies from host to device memory are managed automatically and transfers are only performed when necessary.  matrix_gpu also contains a highly-optimized matrix multiply routine based off the work of Vasily Volkov (which was faster than CUDABlas at the time) as well as various seperate kernels for small and large problem instances.

Please note that badger is an experimental library and I have not worked on it since 2010.  This means that it is incomplete and likely contains a number of bugs.  That said, I think it is an interesting project that someone might find useful.  Someday, I hope to return to project badger and turn it into a more widely useful library.  [More information is available on my home page.](http://www.elliottforney.com/projects/badger/)
