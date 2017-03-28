/********************************\
* Matrix Multiply Test Benchmark *
*                                *
* by                             *
*   Elliott Forney               *
*   3.15.2010                    *
\********************************/

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
#define print_usage() fprintf(stdout, "Usage: %s [-m a_rows] [-n a_cols] [-p b_cols] [-s]\n", arg[0])

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
                unsigned *m, unsigned *n,
                unsigned *p);

/*
 *  Function bodies
 */

// setup network
int main(int narg, char **arg)
{
  // default matrix dimensions
  unsigned m = 6400;
  unsigned n = 0;
  unsigned p = 0;

  // parse command line arguments
  parse_args(narg, arg, &m, &n, &p);

  // if n or p == 0 then set to m
  if (n == 0)
    n = m;
  if (p == 0)
    p = m;

  if (DEBUG > 0)
    printf("m: %d\nn: %d\np: %d\n", m, n, p);

  // create matrix to hold result
  matrix result;
  matrix_init(&result, m, p);
  matrix_load_runif(result, 500, 600);

  // load test matrix a
  matrix a;
  matrix_init(&a, m, p);
//  matrix_load_testa(a, n);

  // load test matrix b
  matrix b;
  matrix_init(&b, m, n);
//  matrix_load_testb(b);
  matrix_load_runif(b, 0, 1);

  // load test matrix c
  matrix c;
  matrix_init(&c, n, p);
  matrix_load_testc(c);
  matrix_load_runif(c, 0, 1);

  // figure a with ATLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              b.r, c.c, b.c, 1.0f,
              b.cpu_data, b.cstride, c.cpu_data, c.cstride,
              0.0f, a.cpu_data, a.cstride);

  // run warm up
  matrix_mult(result, b, c);

  // wait for all kernels to finish
  matrix_wait();

  // create a new benchmark timer
  benchmark ben;
  benchmark_init(&ben);

  // start timer
  benchmark_start_timer(&ben);

  // run multiplication
  matrix_mult(result, b, c);

  // wait for all kernels to finish
  matrix_wait();

  if (DEBUG > 5)
  {
    printf("a:\n");
    matrix_print_padded(a);
    printf("b:\n");
    matrix_print_padded(b);
    printf("c:\n");
    matrix_print_padded(c);
    printf("result:\n");
    matrix_print_padded(result);
  }

  // stop timer
  benchmark_stop_timer(&ben);

  // figure giga floating point operatins per second
  benchmark_add_flop(&ben, matmult_nflop(m,n,p));
  benchmark_add_byte(&ben, (m*n+n*p+m*p)*sizeof(float));
  double gbytes = benchmark_check_gbytes(ben);
  double gflops = benchmark_check_gflops(ben);
  double time   = benchmark_check_timer(ben);

  // 
  float rmse = matrix_rmse(a, result);
  bool passed = true;
  if (rmse > TOLERANCE)
    passed = false;

  // if simple output requested
  if (simple_out)
    // simply print time, gflops and error
    printf("%f %f\n", time*1e3f, gflops);

  // if full output requested
  else
  {
    if (passed)
      printf("Test Passed!\n=======\n");
    else
      printf("Test Failed!\n=======\n");

    // print time and gflops
    printf("Time: %f  GByteS: %f  GFlopS: %f  RMSE: %f\n",
           time, gbytes, gflops, rmse);
  }

  // clean up
  matrix_dest(&a);
  matrix_dest(&b);
  matrix_dest(&c);
  matrix_dest(&result);
  benchmark_dest(&ben);

  // Come back soon now ya'hear!
  return 0;
}

// parse command line arguments
void parse_args(int narg, char **arg,
                unsigned *m, unsigned *n,
                unsigned *p)
{
  int opt; // getopt output

  // for each argument
  while ((opt = getopt(narg, arg, "sm:n:p:")) != -1)
  {
    if (opt == 's')
      simple_out = true;

    else if (opt == 'm')
      *m = (unsigned)atoi(arg[optind-1]);

    else if (opt == 'n')
      *n = (unsigned)atoi(arg[optind-1]);

    else if (opt == 'p')
      *p = (unsigned)atoi(arg[optind-1]);
   
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
