/******************************************\
* OpenMP/SSE/BLAS Optimized Matrix Library *
*                                          *
* by                                       *
*   Elliott Forney                         *
*   3.9.2010                               *
\******************************************/

#ifndef MATRIX_CPU_H
  #define MATRIX_CPU_H

  // make c++ friendly
  #ifdef __cplusplus
    extern "C" {
  #endif

  /*
   *  Data types
   */

  typedef struct matrix {
    unsigned r, c;    // num rows and cols
    unsigned cstride; // num cols after zero padding
    unsigned rstride; // num rows after zero padding
    float *cpu_data;  // matrix elements in row major
    float **drc;      // pointers to beginning of rows
    float *cv;        // temp vector to hold last col
  } matrix;

  /*
   *  Function prototypes
   */

  //** Initialization & Destruction

  // initialize matrix m of size r by c
  void matrix_init(matrix *m, const unsigned r, const unsigned c);

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

  // wait for asynchronous calls to finish
  void matrix_wait();

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

  // scalar multiplication & pointwise addition a = b.*c+d
  void matrix_scl_add(matrix a, float b, matrix c, matrix d);

  // pointwise multiplication a = b.*c
  void matrix_pmult(matrix a, matrix b, matrix c);

  // 
  void matrix_phi_prime(matrix a, matrix b);

  //
  void matrix_pmult_phi_prime(matrix a, matrix b, matrix c);

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

  //** Output

  // print m to standard out
  void matrix_print(matrix m);

  // print m including padding to standard out
  void matrix_print_padded(matrix m);

  #ifdef __cplusplus
    }
  #endif

#endif
