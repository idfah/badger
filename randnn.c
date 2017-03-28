/*******************************\
* Apply Feedforward Neural      *
*   Network to random values    *
*   and benchmark it            *
* by                            *
*   Elliott Forney              *
*   4.29.2010                   *
\*******************************/

/*
 *  Libraries
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>  
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <math.h>

#include "ffnn.h"
#include "matrix.h"
#include "errcheck.h"
#include "benchmark.h"

/*
 *  Macros
 */

#define DEBUG 0

// print command line usage
#define print_usage() fprintf(stdout, "Usage: %s [-i num_inputs] [-o num_outputs] [-h num_hidden] [-d num_samples] [-n num_iter] [-s]\n", arg[0])

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
                long long *ni, long long *no,
                long long *nh, long long *ns,
                long long *niter);

/*
 *  Function bodies
 */

// setup network
int main(int narg, char **arg)
{
  // general purpose counter
  long long i, j;

  long long ni = 1023;
  long long no = 1023;
  long long nh = 1023;
  long long ns = 1023;
  long long niter = 100;


  parse_args(narg, arg, &ni, &no, &nh, &ns, &niter);

  // allocate space for network inputs
  float *input = (float*)malloc(ni*ns*sizeof(float));
  if (input == NULL)
    errcheck_cpu();

  // allocate space for network targets
  float *target = (float*)malloc(no*ns*sizeof(float));
  if (target == NULL)
    errcheck_cpu();

  // allocate space for network outputs
  float *output = (float*)malloc(no*ns*sizeof(float));
  if (output == NULL)
    errcheck_cpu();

  // initialize inputs to random values
  srand(9); // fixed seed for now
  for (i = 0; i < no*ns; ++i)
    input[i] = roundf((float)rand() / RAND_MAX);

  // initialize targets
  memcpy(target, input, no*ns*sizeof(float));

  // initialize a new network
  ffnn net;
  ffnn_init(&net, ni, no, nh);

  benchmark ben;
  benchmark_init(&ben);

  long long nflop =

    nh * (2ll*ni+1ll) * ns + 3 * nh * ns +

    no * (2ll*nh+1ll) * ns +

    2ll * no * ns +

    nh * (2ll*no-1ll) * ns +

    4ll * nh * ns +

    nh * (2ll*ns-1ll) * (ni+1ll) +

		no * (2ll*ns-1ll) * (nh+1ll) +

    2ll * (nh * (ni+1ll)) +

    2ll * (no * (nh+1ll));

  nflop *= niter;

	if (DEBUG > 0)
	  printf("nflop:  %lld\n", nflop);

  benchmark_add_flop(&ben, nflop);

  benchmark_start_timer(&ben);

  // train network
  ffnn_train_steepest(net, input, target, ns, 0.135f, 0.0f, niter);

  #ifdef GPU
    matrix_sync_to_cpu(net.hw);
    matrix_sync_to_cpu(net.vw);
  #endif

  benchmark_stop_timer(&ben);

  double gflops = benchmark_check_gflops(ben);
  double time   = benchmark_check_timer(ben);

  if (simple_out)
    printf("%f\n", gflops);
  else
    printf("Time: %f  GFlopS: %f\n", time, gflops);

  //ffnn_eval(net, input, output, n);

  if (DEBUG > 98)
  {
    printf("inputs:\n");
    for (i = 0; i < ni; ++i)
    {
      for (j = 0; j < ns; ++j)
        printf("%f ", input[i*ns+j]);
      printf("\n");
    }

    printf("targets:\n");
    for (i = 0; i < no; ++i)
    {
      for (j = 0; j < ns; ++j)
        printf("%f ", target[i*ns+j]);
      printf("\n");
    }

    printf("outputs:\n");
    for (i = 0; i < no; ++i)
    {
      for (j = 0; j < ns; ++j)
        printf("%f ", output[i*ns+j]);
      printf("\n");
    }
  }

  // destroy network
  ffnn_dest(&net);

  // free inputs, targets & outputs
  free(input);
  free(output);
  free(target);

  // Yay!
  return 0;
}

// parse command line arguments
void parse_args(int narg, char **arg,
                long long *ni, long long *no,
                long long *nh, long long *ns,
                long long *niter)
{
  int opt; // getopt output

  // for each argument
  while ((opt = getopt(narg, arg, "si:o:h:d:n:")) != -1)
  {
    if (opt == 's')
      simple_out = true;

    else if (opt == 'i')
      *ni = (long long)atoi(arg[optind-1]);

    else if (opt == 'o')
      *no = (long long)atoi(arg[optind-1]);

    else if (opt == 'h')
      *nh = (long long)atoi(arg[optind-1]);

    else if (opt == 'd')
      *ns = (long long)atoi(arg[optind-1]);

		else if (opt == 'n')
			*niter = (long long)atoi(arg[optind-1]);
   
    // print usage and quit on unknown option
    else
    {
      print_usage();
      exit(1);
    }
  }

  // print usage and quit if non dash args given
  if (optind < narg)
  {
    print_usage();
    exit(1);
  }
}
