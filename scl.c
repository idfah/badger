/*************************\
* CUDA Accelerated Matrix *
* Scale Test benchmark    *
*                         *
* by                      *
*   Elliott Forney        *
*   3.15.2010             *
\*************************/

/*
 *  Libraries
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>  
#include <unistd.h>
#include <getopt.h>

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
                unsigned *m, unsigned *n);

/*
 *  Function bodies
 */

// setup network
int main(int narg, char **arg)
{
  unsigned i;

  // default matrix dimensions
  unsigned m = 6400;
  unsigned n = 0;

  // parse command line arguments
  parse_args(narg, arg, &m, &n);

  // if n or p == 0 then set to m
  if (n == 0)
    n = m;

  if (DEBUG > 0)
    printf("m: %d\nn: %d\n", m, n);

  // load test matrix a
  matrix a;
  matrix_init(&a, m, n);

  // load test scalar b
  float b = 3.14159f;

  // load test matrix c
  matrix c;
  matrix_init(&c, m, n);
  matrix_load_runif(c, 0, 1);

  // figure a with simple loop
  for (i = 0; i < a.rstride*a.cstride; ++i)
    a.cpu_data[i] = b * c.cpu_data[i];

  // create matrix to hold result
  matrix result;
  matrix_init(&result, m, n);
  matrix_load_zero(result);

  // run warm up
  matrix_scl(result, b, c);

  // wait for all kernels to finish
  matrix_wait();

  // create a new benchmark timer
  benchmark ben;
  benchmark_init(&ben);

  // start timer
  benchmark_start_timer(&ben);

  // run multiplication
  matrix_scl(result, b, c);

  // wait for all kernels to finish
  matrix_wait();

  // stop timer
  benchmark_stop_timer(&ben);

  if (DEBUG > 4)
  {
    printf("a:\n");
    matrix_print_padded(a);
    printf("b:\n%f", b);
    printf("c:\n");
    matrix_print_padded(c);
    printf("result:\n");
    matrix_print_padded(result);
  }

  // figure giga floating point operatins per second
  benchmark_add_byte(&ben, (2*m*n+1)*sizeof(float));
  double gbytes = benchmark_check_gbytes(ben);
  double time   = benchmark_check_timer(ben);

  // 
  float rmse = matrix_rmse(a, result);
  bool passed = true;
  if (rmse > TOLERANCE)
    passed = false;

  // if simple output requested
  if (simple_out)
    // simply print time, gbytess and error
    //printf("%f %f\n", time*1e3f, gbytes);
    printf("%f\n", gbytes);

  // if full output requested
  else
  {
    if (passed)
      printf("Test Passed!\n=======\n");
    else
      printf("Test Failed!\n=======\n");

    // print time and gbytess
    printf("Time: %f  GByteS: %f  RMSE: %f\n",
           time, gbytes, rmse);
  }

  // clean up
  matrix_dest(&a);
  matrix_dest(&c);
  matrix_dest(&result);
  benchmark_dest(&ben);

  // Come back soon now ya'hear!
  return 0;
}

// parse command line arguments
void parse_args(int narg, char **arg,
                unsigned *m, unsigned *n)
{
  int opt; // getopt output

  // for each argument
  while ((opt = getopt(narg, arg, "sm:n:")) != -1)
  {
    if (opt == 's')
      simple_out = true;

    else if (opt == 'm')
      *m = (unsigned)atoi(arg[optind-1]);

    else if (opt == 'n')
      *n = (unsigned)atoi(arg[optind-1]);

    // print usage and quit on unknown option
    else
    {
      print_usage();
      exit(1);
    }
  }

  // assume last non dash arg is m
  if (optind < narg)
    *m = (unsigned)atoi(arg[optind]);
}
