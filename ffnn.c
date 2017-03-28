/*****************************\
* Feed-Forward Neural Network *
*                             *
* by                          *
*   Elliott Forney            *
*   3.9.2010                  *
\*****************************/

/*
 *  Libraries
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "errcheck.h"
#include "matrix.h"
#include "ffnn.h"

/*
 *  Macros
 */

#define DEBUG 2

/*
 *  Function prototypes
 */

//
void eval(ffnn net, matrix x, matrix y);

/*
 *  Function bodies
 */

//
void eval(ffnn net, matrix x, matrix y)
{
  // forward pass
  //     Y     =     V         H         X
  // (no x ns) = (no x nh) (nh x ni) (ni x ns)

  // tac bias on to input matrix
  matrix_r1(&x);

  // figure output of hidden layer
  matrix z;
  matrix_init(&z, net.hw.r, x.c);
  matrix_mult_phi(z, net.hw, x);

  // take bias back off input
  matrix_r0(&x);

  // add bias to hidden layer output
  matrix_r1(&z);

  // output of visible layer
  matrix_mult(y, net.vw, z);

  // destroy intermediate
  matrix_dest(&z);
}

/*
 *  External function bodies
 */

//** Initialization & Destruction

// initialize a new network
void ffnn_init(ffnn *net, unsigned ni, unsigned no, unsigned nh)
{
  // set network size
  net->ni = ni;
  net->no = no;
  net->nh = nh;

  // initialize weights, cite Lecun paper here!!
  const float w_min = -sqrtf(3.0f/(ni+1.0f));
  const float w_max =  sqrtf(3.0f/(ni+1.0f));

  // initialize hidden layer
  matrix_init(&(net->hw), nh, ni+1);
  matrix_load_runif(net->hw, w_min, w_max);

  // initialize visible layer
  matrix_init(&(net->vw), no, nh+1);
  matrix_load_runif(net->vw, w_min, w_max);
}

// initialize a new network from file
void ffnn_init_file(ffnn *net, char *file_name)
{
  char *file_name_wext = (char*)malloc((strlen(file_name)+4)*sizeof(char));

  strcpy(file_name_wext, file_name);
  strcat(file_name_wext, ".hw");
  matrix_init_file(&(net->hw), file_name_wext);

  strcpy(file_name_wext, file_name);
  strcat(file_name_wext, ".vw");
  matrix_init_file(&(net->vw), file_name_wext);

  // set network size
  // (no x ns) = (no x nh) (nh x ni) (ni x ns)
  net->ni = (net->hw).c-1;
  net->no = (net->vw).r;
  net->nh = (net->hw).r;
}

// destroy an existing network
void ffnn_dest(ffnn *net)
{
  // free matrices
  matrix_dest(&(net->hw));
  matrix_dest(&(net->vw));

  // set everyting to zero for safety
  net->ni = 0;
  net->no = 0;
  net->nh = 0;
}

//** Evaluation

// evaluate network outputs for given intputs with ns samples
void ffnn_eval(ffnn net, float *input, float *output, unsigned ns)
{
  // initialize matrices
  matrix x, y;
  matrix_init(&x, net.ni, ns);
  matrix_init(&y, net.no, ns);

  // load inputs
  matrix_load_array(x, input);

  // evaluate network
  eval(net, x, y);

  // store results in output array
  matrix_unload_array(y, output);

  // clean up matrices
  matrix_dest(&x);
  matrix_dest(&y);
}

//** Training

// train net weights using steepest descent
void ffnn_train_steepest(ffnn net, float *input, float *target, unsigned ns,
                         float lr, float precision, float maxiter)
{

  // forward pass
  //     Y     =     V         H         X
  // (no x ns) = (no x nh) (nh x ni) (ni x ns)
  unsigned iter;

  // initialize matrices
  matrix x, y, z, g, delta, hg, vg;
  matrix_init(&x, net.ni, ns); // inputs
  matrix_init(&y, net.no, ns); // outputs
  matrix_init(&z, net.nh, ns); // hidden outputs
  matrix_init(&g, net.no, ns); // targets
  matrix_init(&delta, net.no, ns); // error derivative
  matrix_init(&hg, net.hw.r, net.hw.c); // hidden gradient
  matrix_init(&vg, net.vw.r, net.vw.c); // visible gradient

  // temporary matrices
  matrix t0, t1, t2, t3;
  matrix_init(&t0, z.c, z.r+1);
  matrix_init(&t1, net.vw.c-1, net.vw.r);
  matrix_init(&t2, t1.r, delta.c);
  matrix_init(&t3, x.c, x.r+1);

  // load inputs and targets
  matrix_load_array(x, input);
  matrix_load_array(g, target);

  //** forward pass

  // tac bias on to input matrix
  // x1 <- rbind(x,1)
  matrix_r1(&x);

  for (iter = 0; iter < maxiter; ++iter)
  {
    if (DEBUG > 1)
      printf("iter: %d\n", iter);

    // figure output of hidden layer
    // z <- net$phi(net$hw %*% x1)
    matrix_mult_phi(z, net.hw, x);

    // add bias to hidden layer output
    // z1 <- rbind(z,1)
    matrix_r1(&z);

    // output of visible layer
    // y <- net$vw %*% z1

    matrix_mult(y, net.vw, z);

    if (DEBUG > 0)
      printf("RMSE: %f\n", matrix_rmse(g, y));

    // figure delta
    // delta <- 2 * (y - r) / length(r)
    matrix_delta(delta, y, g);

    // delta %*% t(z1)
    matrix_trans(t0, z);

    matrix_mult(vg, delta, t0);

    // remove bias from hidden layer output
    matrix_r0(&z);

    // remove last col from vw
    matrix_c0v(&net.vw);

    // store transpose of vw
    matrix_trans(t1, net.vw);

    // restore last col to vw
    matrix_cv(&net.vw);

    // mult visible weights by delta
    // t(net$vw[...]) %*% delta
    matrix_mult(t2, t1, delta);

    // net$phi.prime(z)
    matrix_phi_prime(z, z);

    // (t(net$vw[...]) %*% delta) * net$phi.prime(z)
    matrix_pmult(z, t2, z);

// need to implement in gpu
//    matrix_pmult_phi_prime(z, t2, z);

    // t(x1)
    matrix_trans(t3, x);

    // ((t(net$vw[...]) %*% delta) * net$phi.prime(z)) %*% t(x1)
    matrix_mult(hg, z, t3);

    //matrix_scl(hg, -lr, hg);
    //matrix_add(net.hw, net.hw, hg);
    matrix_scl_add(net.hw, -lr, hg, net.hw);

    //matrix_scl(vg, -lr, vg);
    //matrix_add(net.vw, net.vw, vg);
    matrix_scl_add(net.vw, -lr, vg, net.vw);
  }

  // take bias back off input
  matrix_r0(&x);

  matrix_dest(&t0);
  matrix_dest(&t1);
  matrix_dest(&t2);
  matrix_dest(&t3);

  matrix_dest(&x);
  matrix_dest(&y);
  matrix_dest(&z);
  matrix_dest(&g);
}
