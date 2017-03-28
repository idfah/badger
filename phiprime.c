/**********************************\
* Test benchmark                   *
*                                  *
* by                               *
*   Elliott Forney                 *
*   3.15.2010                      *
\**********************************/

/*
 *  Libraries
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>  
#include <unistd.h>
#include <getopt.h>
#include <math.h>

#include <cblas.h>

#include "matrix.h"
#include "benchmark.h"
#include "errcheck.h"

/*
 *  Macros
 */

// print command line usage
#define print_usage() fprintf(stdout, "Usage: %s [-m rows] [-n cols] [-s]\n", arg[0])

// acceptable maximum relative error tolerance
#define TOLERANCE 0.01

#define DEBUG 0

/*
 *  Global variables
 */

// flag for simple, one line output
bool simple_out = false;

/*
 *  Function prototypes
 */

// parse command line arguments
void parse_args(int narg, char **arg,
                long long *m, long long *n);

/*
 *  Function bodies
 */

// setup network
int main(int narg, char **arg)
{
  // default matrix dimensions
  long long m = 6400;
  long long n = 0;

  // parse command line arguments
  parse_args(narg, arg, &m, &n);

  // if n or p == 0 then set to m
  if (n == 0)
    n = m;

  // load test matrix b
  matrix a;
  matrix_init(&a, m, n);
  matrix_load_runif(a, 0, 1);

  // create matrix to hold result
  matrix result;
  matrix_init(&result, m, n);

  // run warm up
  matrix_phi_prime(result, a);

  // wait for all kernels to finish
  matrix_wait();

  // create a new benchmark timer
  benchmark ben;
  benchmark_init(&ben);

  // start timer
  benchmark_start_timer(&ben);

  // run multiplication
  matrix_phi_prime(result, a);

  // wait for all kernels to finish
  matrix_wait();

  // stop timer
  benchmark_stop_timer(&ben);

  // figure giga floating point operatins per second
  benchmark_add_flop(&ben, 3ll*m*n);
  double gflops = benchmark_check_gflops(ben);
  double time   = benchmark_check_timer(ben);

  // print time and gbytess
  printf("Time: %f  GFlopS: %f\n",
         time, gflops);

  // clean up
  matrix_dest(&a);
  matrix_dest(&result);
  benchmark_dest(&ben);

  // Come back soon now ya'hear!
  return 0;
}

// parse command line arguments
void parse_args(int narg, char **arg,
                long long *m, long long *n)
{
  int opt; // getopt output

  // for each argument
  while ((opt = getopt(narg, arg, "sm:n:")) != -1)
  {
    if (opt == 's')
      simple_out = true;

    else if (opt == 'm')
      *m = (long long)atoi(arg[optind-1]);

    else if (opt == 'n')
      *n = (long long)atoi(arg[optind-1]);

    // print usage and quit on unknown option
    else
    {
      print_usage();
      exit(1);
    }
  }

  // assume last non dash arg is m
  if (optind < narg)
    *m = (long long)atoi(arg[optind]);
}
