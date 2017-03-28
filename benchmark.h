/***********************\
* Benchmarking routines *
*                       *
* by                    *
*   Elliott Forney      *
*   3.15.2010           *
\***********************/

#ifndef BENCHMARK_H
  #define BENCHMARK_H

  // make c++ friendly
  #ifdef __cplusplus
    extern "C" {
  #endif

  /*
   *  Libraries
   */

  #include <stdbool.h>
  #include <time.h>

  /*
   *  Type definitions
   */

  typedef struct {
    bool    timing;
    struct  timespec start_ts, end_ts;
    double  elapsed_time;
    unsigned long long nbyte;
    unsigned long long nflop;
  } benchmark;

  /*
   *  Macros
   */

  //** Figure num flop for common operations

  // number of flops for matmult (m x n) * (n x p)
  #define matmult_nflop(m,n,p) m*p*(2ll*n-1ll)

  //** Figure num byte for common operations

  // number of bytes in/out for single precision
  // matrix add (m x n) + (m x n)
  #define matadd_nbytef(m,n) 3*m*n*sizeof(float)

  /*
   *  Function prototypes
   */

  //** Initialization & destruction

  // initialize a new benchmark
  int benchmark_init(benchmark *b);

  // reset benchmark to all zeros
  int benchmark_reset(benchmark *b);

  // destroy a benchmark
  int benchmark_dest(benchmark *b);

  //** Timing

  // start timing
  int benchmark_start_timer(benchmark *b);

  // stop timing
  int benchmark_stop_timer(benchmark *b);

  // reset timer to zero
  void benchmark_reset_timer(benchmark *b);

  // check current time
  double benchmark_check_timer(benchmark b);

  //** Throughput

  // add bytes transfered
  void benchmark_add_byte(benchmark *b, unsigned long long nbyte);

  // check billion bytes transfered per second
  double benchmark_check_gbytes(benchmark b);

  //** Computation

  // add floating point operations
  void benchmark_add_flop(benchmark *b, unsigned long long nflop);

  // check billion floating point operations per second
  double benchmark_check_gflops(benchmark b);

  #ifdef __cplusplus
    }
  #endif

#endif
