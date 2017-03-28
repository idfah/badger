/***********************\
* Benchmarking routines *
*                       *
* by                    * 
*   Elliott Forney      *
*   3.15.2010           *
\***********************/

/*
 *  Libraries
 */

#include <stdbool.h>
#include <time.h>

#include "benchmark.h"

/*
 *  Function bodies
 */

//** Internal

// figure time in seconds
double figure_time(struct timespec start_ts, struct timespec end_ts)
{
  double start_time = (double)start_ts.tv_sec +           // seconds
                      ((double)start_ts.tv_nsec)*1.0e-9;  // nanoseconds
  double end_time   = (double)end_ts.tv_sec   +
                      ((double)end_ts.tv_nsec)  *1.0e-9;

  return end_time - start_time;
}

//** Initialization & destruction

// initialize a new benchmark
int benchmark_init(benchmark *b)
{
  benchmark_reset(b);
  return 0;
}

// reset benchmark to all zeros
int benchmark_reset(benchmark *b)
{
  b->elapsed_time = 0.0;
  b->timing = false;

  b->nbyte = 0ll;
  b->nflop = 0ll;

  return 0;
}

// destroy a benchmark
int benchmark_dest(benchmark *b)
{
  // nothing to do
  return 0;
}

//** Timing

// start timing
int benchmark_start_timer(benchmark *b)
{
  // error if timer already running
  if (b->timing == true)
    return -1;

  // mark timer as running
  b->timing = true;

  // look at the clock
  clock_gettime(CLOCK_REALTIME, &(b->start_ts));

  return 0;
}

// stop timing
int benchmark_stop_timer(benchmark *b)
{
  // error if timer not running
  if (b->timing != true)
    return -1;

  // mark timer as stopped
  b->timing = false;

  // look at clock
  clock_gettime(CLOCK_REALTIME, &(b->end_ts));

  // add time into elapsed
  b->elapsed_time += figure_time(b->start_ts, b->end_ts);

  return 0;
}

// reset timer to zero
void benchmark_reset_timer(benchmark *b)
{
  b->timing = false;
  b->elapsed_time = 0.0;
}

// check current time
double benchmark_check_timer(benchmark b)
{
  // if we are currently timing
  if (b.timing)
  {
    // look at clock
    struct timespec current_ts;
    clock_gettime(CLOCK_REALTIME, &current_ts);

    // add time into elapsed
    return b.elapsed_time + figure_time(b.start_ts, current_ts);
  }

  // if timer stopped, just return elapsed time
  return b.elapsed_time;
}

//** Throughput

// add bytes transfered
void benchmark_add_byte(benchmark *b, unsigned long long nbyte)
{
  b->nbyte += nbyte;
}

// check billion bytes transfered per second
double benchmark_check_gbytes(benchmark b)
{
  double bytes  = b.nbyte/b.elapsed_time;
  double gbytes = bytes*1.0e-9;

  return gbytes;
}

//** Computation

// add floating point operations
void benchmark_add_flop(benchmark *b, unsigned long long nflop)
{
  b->nflop += nflop;
}

// check billion floating point operations per second
double benchmark_check_gflops(benchmark b)
{
  double flops  = b.nflop/b.elapsed_time;
  double gflops = flops*1.0e-9;

  return gflops;
}
