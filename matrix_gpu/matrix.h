/*********************************\
* Cuda Accelerated Matrix Library *
*                                 *
* by                              *
*   Elliott Forney                *
*   3.9.2010                      *
\*********************************/

#ifndef MATRIX_H
  #define MATRIX_H

  // make c++ friendly
  #ifdef __cplusplus
    extern "C" {
  #endif

  /*
   *  Data types
   */

  // cpu/gpu sync state
  typedef enum {
    matrix_sync_cpu,    // cpu only is current
    matrix_sync_gpu,    // gpu only is current
    matrix_sync_both    // cpu and gpu are current
  } matrix_sync_state;

  // matrix data type
  typedef struct {
    unsigned r, c;            // num rows and cols
    unsigned cstride;         // num cols after zero padding
    unsigned rstride;         // num rows after zero padding
    float *cpu_data;          // matrix elements in row major
    float *gpu_data;          // matrix elements in row major
    float *cv;                // temp vector to hold last col
    matrix_sync_state *sync;  // sync state between cpu & gpu
  } matrix;

  /*
   *  Function prototypes
   */

  //** Initialization & Destruction

  // initialize matrix m of size r by c
  void matrix_init(matrix *m, unsigned r, unsigned c);

  // initialize matrix m from ascii table
  void matrix_init_file(matrix *m, char *file_name);

  // load zeros into matrix m
  void matrix_load_zero(matrix m);

  // load values from an array
  void matrix_load_array(matrix m, float *v);

  // load values from the random uniform distribution
  void matrix_load_runif(matrix m, float min, float max);

  // load test data into matrix m such that
  // a = b * c
  void matrix_load_testa(matrix m, unsigned n);
  void matrix_load_testb(matrix m);
  void matrix_load_testc(matrix m);

  // copy values to an array
  void matrix_unload_array(matrix m, float *v);

  // write values to a file
  void matrix_unload_file(matrix m, char *file_name);

  // destroy matrix m
  void matrix_dest(matrix *m);

  //** Synchronization

  // ensure current copy of matrix is on cpu
  void matrix_sync_to_cpu(matrix m);

  // ensure current copy of matrix is on gpu
  void matrix_sync_to_gpu(matrix m);

  // wait for any gpu kernels to finish
  void matrix_wait();

  //** Access

  // return pointer to cpu value at row r and col c
  float *matrix_at(matrix m, unsigned r, unsigned c);

  //** Addition/Subtraction

  // a = b + c
  void matrix_add(matrix a, matrix b, matrix c);

  // a = b - c
  void matrix_sub(matrix a, matrix b, matrix c);

  //** Multiplication

  // a = b * c
  void matrix_mult(matrix a, matrix b, matrix c);

  // a = phi(b * c)
  void matrix_mult_phi(matrix a, matrix b, matrix c);

  //** Transpose

  // a = b^T
  void matrix_trans(matrix a, matrix b);

  //** Pointwise miscellaneous

  // a = b^2
  void matrix_sqr(matrix a, matrix b);

  // scalar multiplication a = b*c
  void matrix_scl(matrix a, float b, matrix c);

  // scalar multiplication a = b*c
  void matrix_scl(matrix a, float b, matrix c);

  // scalar multiplication & pointwise addition a = b.*c+d
  void matrix_scl_add(matrix a, float b, matrix c, matrix d);

  // pointwise multiplication a = b.*c
  void matrix_pmult(matrix a, matrix b, matrix c);

  // 
  void matrix_phi_prime(matrix a, matrix b);

  //
  void matrix_delta(matrix delta, matrix y, matrix g);

  //** Combination/Separation

  // remove last row of m
  void matrix_r0(matrix *m);

  // append a row of 1's to m
  void matrix_r1(matrix *m);

  // remove and save last col of m
  void matrix_c0v(matrix *m);

  // restore last col of m
  void matrix_cv(matrix *m);

  //** Error measurement

  // rmse between values of actual and approx
  float matrix_rmse(matrix actual, matrix approx);

  // maximum relative error between approx and actual
  float matrix_relerr_max(matrix actual, matrix approx);

  //
  // float matrix_mse(matrix approx, matrix actual);

  //** Output

  // print m to standard out
  void matrix_print(matrix m);

  // print m including padding to standard out
  void matrix_print_padded(matrix m);

  #ifdef __cplusplus
    }
  #endif

#endif
