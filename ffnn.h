/*****************************\
* Feed-Forward Neural Network *
*                             *
* by                          *
*   Elliott Forney            *
*   3.9.2010                  *
\*****************************/

#ifndef FFNN_H
  #define FFNN_H

  // make c++ friendly
  #ifdef __cplusplus
    extern "C" {
  #endif

  /*
   *  Libraries
   */

  #include "matrix.h"

  /*
   *  Type definitions
   */

  // feedforward neural net
  typedef struct {
    unsigned ni; // number of inputs
    unsigned no; // number of outputs
    unsigned nh; // number of hidden units
    matrix   hw; // hidden  weights (nh x ni)
    matrix   vw; // visible weights (no x nh)
  } ffnn;

// (no x ns) = (no x nh) (nh x ni) (ni x ns)

  /*
   *  Function prototypes
   */

  //** Initialization & Destruction

  // initialize a new network
  void ffnn_init(ffnn *net, unsigned ni, unsigned no, unsigned nh);

  // initialize a new network from file
  void ffnn_init_file(ffnn *net, char *file_name);

  // destroy an existing network
  void ffnn_dest(ffnn *net);

  //** Evaluation

  // evaluate network outputs for given intputs with ns samples
  void ffnn_eval(ffnn net, float *input, float *output, unsigned ns);

  // evaluate output of hidden layer for given inputs
  //void ffnn_evalh(ffnn net, float *input, float *hidden_output);

  //** Training

  // train net weights using steepest descent
  void ffnn_train_steepest(ffnn net, float *input, float *target, unsigned ns,
                           float lr, float precision, float maxiter);

  //** Error measurement

  // return root mean squared error for given inputs and targets
  void ffnn_rmse(ffnn net, float *input, float *target);

  #ifdef __cplusplus
    }
  #endif

#endif
