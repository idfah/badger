/*******************************\
* Solve XOR problem with        *
*   Feedforward Neural Network  *
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
#include <time.h>
#include <math.h>

#include "ffnn.h"
#include "matrix.h"
#include "errcheck.h"

/*
 *  Macros
 */

#define DEBUG 99

// print command line usage
#define print_usage() fprintf(stdout, "Usage: %s [-n num_samples] [-h num_hidden] [-s]\n", arg[0])

/*
 *  Global variables
 */

// flag for simple, one line output
bool simple_out = false;

/*
 *  Function prototypes
 */

// parse command line arguments
void parse_args(int narg, char **arg, unsigned *n, unsigned *nh);

// determine rounded xor of floats a and b
float xorf(float a, float b);

/*
 *  Function bodies
 */

// setup network
int main(int narg, char **arg)
{
  // general purpose counter
  unsigned i, j;

  // number of network inputs, targets & outputs
  unsigned n = 10;

  // number of hidden units
  unsigned nh = 6;

  parse_args(narg, arg, &n, &nh);

  // allocate space for network inputs
  float *input = (float*)malloc(2*n*sizeof(float));
  if (input == NULL)
    errcheck_cpu();

  // allocate space for network targets
  float *target = (float*)malloc(n*sizeof(float));
  if (target == NULL)
    errcheck_cpu();

  // allocate space for network outputs
  float *output = (float*)malloc(n*sizeof(float));
  if (output == NULL)
    errcheck_cpu();

  // initialize inputs to random binary values
  srand(time(NULL)); // fixed seed for now
  for (i = 0; i < 2*n; ++i)
    input[i] = roundf((float)rand() / RAND_MAX);

  // initialize targets
  for (i = 0, j = 0; i < n; ++i, ++j)
    target[i] = xorf(input[j], input[j+n]);

  // initialize a new network
  ffnn net;
  ffnn_init(&net, 2, 1, nh);

  //ffnn_init_file(&net, "xor");

  // test forward pass 
  //ffnn_eval(net, input, output, n);

  ffnn_train_steepest(net, input, target, n, 0.135f, 0.0f, 300);
  ffnn_eval(net, input, output, n);

  if (DEBUG > 4)
  {
    printf("inputs:\n");
    for (i = 0; i < 2; ++i)
    {
      for (j = 0; j < n; ++j)
        printf("%f ", input[i*n+j]);
      printf("\n");
    }

    printf("targets:\n");
    for (i = 0; i < n; ++i)
      printf("%f ", target[i]);
    printf("\n");

    printf("outputs:\n");
    for (i = 0; i < n; ++i)
      printf("%f ", output[i]);
    printf("\n");
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
                unsigned *n, unsigned *nh)
{
  int opt; // getopt output

  // for each argument
  while ((opt = getopt(narg, arg, "sn:h:")) != -1)
  {
    if (opt == 's')
      simple_out = true;

    else if (opt == 'n')
      *n = (unsigned)atoi(arg[optind-1]);

    else if (opt == 'h')
      *nh = (unsigned)atoi(arg[optind-1]);
   
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

// determine rounded xor of floats a and b
float xorf(float a, float b)
{
  if ( ((a <= 0.5f) && (b <= 0.5f)) ||
       ((a >  0.5f) && (b >  0.5f)) )
    return 0.0f;

  else if ((a > 0.5f) || (b > 0.5f))
    return 1.0f;

  else
    return 0.0f;
}
