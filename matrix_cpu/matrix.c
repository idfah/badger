/******************************************\
* OpenMP/SSE/BLAS Optimized Matrix Library *
*                                          *
* by                                       *
*   Elliott Forney                         *
*   3.9.2010                               *
\******************************************/

/*
 *  Libraries
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// cblas
#include <cblas.h>

// OpenMP
#include <omp.h>

// SSE 4.1 intrinsics
#include <smmintrin.h>

// SSE 2 intrinsics
// #include <pmmintrin.h>

//
#include "errcheck.h"
#include "matrix.h"

/*
 *  Macros
 */

//** Debugging level
#define DEBUG 0

//** Initialization

// max number of chars per line in input table
#define line_buff_size 1024

//** Access

// row major indexing
// expects stride and data to be set
#define data_at(r,c) data[r*stride+c]

//** Addition/Subtraction

// when number of values in matrix is
// larger than add_lim use big kernel
#define add_lim 10000

//** Transpose

#define trans_lim 22500

//#define trans_tile 64

/*
 *  Global variables
 */

unsigned rand_state = 8;

/*
 *  Function prototypes
 */

//
void build_drc(matrix *m);

// sigmoid function
float phi(const float v);

// derivative of sigmoid function given phi(v)
float phi_prime(const float z);

// addition kernel for small matrices
void add_small(float *a, float *b, float *c, const unsigned n);

// addition kernel for big matrices
void add_big(float *m, float *b, float *c, const unsigned n);

// subtraction kernel for small matrices
void sub_small(float *a, float *b, float *c, const unsigned n);

// subtraction kernel for big matrices
void sub_big(float *a, float *b, float *c, const unsigned n);

// pointwise square kernel for small matrices
void sqr_small(float *a, float *b, const unsigned n);

// pointwise square kernel for big matrices
void sqr_big(float *a, float *b, const unsigned n);

// scalar multiply kernel for small matrices
void scl_small(float *a, float b, float *c, const unsigned n);

// scalar multiply kernel for big matrices
void scl_big(float *a, float b, float *c, const unsigned n);

// scalar multiplication & pointwise addition kernel for small matrices
void scl_add_small(float *a, float b, float *c, float *d, const unsigned n);

// scalar multiplication & pointwise addition kernel for big matrices
void scl_add_big(float *a, float b, float *c, float *d, const unsigned n);

// pointwise multiplication for small matrices
void pmult_small(float *a, float *b, float *c, const unsigned n);

// pointwise multiplication for big matrices
void pmult_big(float *a, float *b, float *c, const unsigned n);

/*
 *  Function bodies
 */

// build row-col indices for double index
void build_drc(matrix *m)
{
  // not done/needed yet
}

// sigmoid function
float phi(const float v)
{
  // recommended by Lecun, find citation!!
  const float scl = 2.0f/3.0f;
  return 1.7159f * tanh( scl * v );
}

// derivative of sigmoid function given phi(v)
float phi_prime(const float z)
{
  const float scl = 2.0f/3.0f;
  return scl * (1.7159f - z*z);
}

// addition kernel for small matrices
void add_small(float *a, float *b, float *c, const unsigned n)
{
  // pointer to last destination
  const float *a_end = a+n;

  // loop through each value in destination
  while (a < a_end)
  {
    // four values from b and c into sse registers
    __m128 mm_v1 = _mm_load_ps(b);
    __m128 mm_v2 = _mm_load_ps(c);

    // add our sse vectors
    mm_v1 = _mm_add_ps(mm_v1, mm_v2);

    // store result into a
    _mm_store_ps(a, mm_v1);

    // increment pointers by 4
    a += 4; b += 4; c += 4;
  }
}

// addition kernel for big matrices
void add_big(float *a, float *b, float *c, const unsigned n)
{
  unsigned i;

  // loop through a, b and c by 4 in parallel
  #pragma omp parallel for
  for (i = 0; i < n; i += 4)
  {
    // load four values into sse registers
    __m128 mm_v1 = _mm_load_ps(b+i); // what if they have different strides!!! :(
    __m128 mm_v2 = _mm_load_ps(c+i);

    // add our sse vectors
    mm_v1 = _mm_add_ps(mm_v1, mm_v2);

    // store result into a
    _mm_store_ps(a+i, mm_v1);
  }
}

// subtraction kernel for small matrices
void sub_small(float *a, float *b, float *c, const unsigned n)
{
  const float *a_end = a+n;

  while (a < a_end)
  {
    __m128 mm_v1 = _mm_load_ps(b);
    __m128 mm_v2 = _mm_load_ps(c);

    mm_v1 = _mm_sub_ps(mm_v1, mm_v2);

    _mm_store_ps(a, mm_v1);

    a += 4; b += 4; c += 4;
  }
}

// subtraction kernel for big matrices
void sub_big(float *a, float *b, float *c, const unsigned n)
{
  unsigned i;

  #pragma omp parallel for
  for (i = 0; i < n; i += 4)
  {
    __m128 mm_v1 = _mm_load_ps(b+i);
    __m128 mm_v2 = _mm_load_ps(c+i);

    mm_v1 = _mm_sub_ps(mm_v1, mm_v2);

    _mm_store_ps(a+i, mm_v1);
  }
}

// pointwise square kernel for small matrices
void sqr_small(float *a, float *b, const unsigned n)
{
  const float *a_end = a+n;

  while (a < a_end)
  {
    __m128 mm_v = _mm_load_ps(b);

    mm_v = _mm_mul_ps(mm_v, mm_v);

    _mm_store_ps(a, mm_v);

    a += 4; b += 4;
  }
}

// pointwise square kernel for big matrices
void sqr_big(float *a, float *b, const unsigned n)
{
  unsigned i;

  #pragma omp parallel for
  for (i = 0; i < n; i += 4)
  {
    __m128 mm_v = _mm_load_ps(b+i);

    mm_v = _mm_mul_ps(mm_v, mm_v);

    _mm_store_ps(a+i, mm_v);
  }
}

// scalar multiplication kernel for small matrices
void scl_small(float *a, float b, float *c, const unsigned n)
{
  const float *a_end = a+n;

  __m128 mm_s = _mm_set1_ps(b);

  while (a < a_end)
  {
    __m128 mm_v = _mm_load_ps(c);

    mm_v = _mm_mul_ps(mm_s, mm_v);

    _mm_store_ps(a, mm_v);

    a += 4; c += 4;
  }
}

// scalar multiplication kernel for big matrices
void scl_big(float *a, float b, float *c, const unsigned n)
{
  unsigned i;

  const __m128 mm_s = _mm_set1_ps(b);

  #pragma omp parallel for
  for (i = 0; i < n; i += 4)
  {
    __m128 mm_v = _mm_load_ps(c+i);

    mm_v = _mm_mul_ps(mm_s, mm_v);

    _mm_store_ps(a+i, mm_v);
  }
}

// scalar multiplication & pointwise addition kernel for small matrices
void scl_add_small(float *a, float b, float *c, float *d, unsigned n)
{
  const float *a_end = a+n;

  __m128 mm_s = _mm_set1_ps(b);

  while (a < a_end)
  {
    __m128 mm_v1 = _mm_load_ps(c);
    __m128 mm_v2 = _mm_load_ps(d);

    mm_v1 = _mm_mul_ps(mm_s,  mm_v1);
    mm_v1 = _mm_add_ps(mm_v1, mm_v2);

    _mm_store_ps(a, mm_v1);

    a += 4; c += 4; d += 4;
  }
}

// scalar multiplication & pointwise addition kernel for big matrices
void scl_add_big(float *a, float b, float *c, float *d, unsigned n)
{
  unsigned i;

  const __m128 mm_s = _mm_set1_ps(b);

  #pragma omp parallel for
  for (i = 0; i < n; i += 4)
  {
    __m128 mm_v1 = _mm_load_ps(c+i);
    __m128 mm_v2 = _mm_load_ps(d+i);

    mm_v1 = _mm_mul_ps(mm_s,  mm_v1);
    mm_v1 = _mm_add_ps(mm_v1, mm_v2);

    _mm_store_ps(a+i, mm_v1);
  }
}


// pointwise multiplication kernel for small matrices
void pmult_small(float *a, float *b, float *c, const unsigned n)
{
  // pointer to last destination
  const float *a_end = a+n;

  // loop through each value in destination
  while (a < a_end)
  {
    // four values from b and c into sse registers
    __m128 mm_v1 = _mm_load_ps(b);
    __m128 mm_v2 = _mm_load_ps(c);

    // mult our sse vectors
    mm_v1 = _mm_mul_ps(mm_v1, mm_v2);

    // store result into a
    _mm_store_ps(a, mm_v1);

    // increment pointers by 4
    a += 4; b += 4; c += 4;
  }
}

// pointwise multiplication kernel for big matrices
void pmult_big(float *a, float *b, float *c, const unsigned n)
{
  unsigned i;

  // loop through a, b and c by 4 in parallel
  #pragma omp parallel for
  for (i = 0; i < n; i += 4)
  {
    // load four values into sse registers
    __m128 mm_v1 = _mm_load_ps(b+i);
    __m128 mm_v2 = _mm_load_ps(c+i);

    // mult our sse vectors
    mm_v1 = _mm_mul_ps(mm_v1, mm_v2);

    // store result into a
    _mm_store_ps(a+i, mm_v1);
  }
}

/*
 *  External function bodies
 */

//** Initialization & Destruction

// initialize matrix m of size r by c
void matrix_init(matrix *m, const unsigned r, const unsigned c)
{
  // set matrix dimensions
  m->r = r;
  m->c = c;

  // zero pad to nearest mult of 64, lcm 4 for sse, 64 for trans
  // leave one extra row/col to append bias terms
//  m->rstride = r + (64 - (r % 64)); // lcm(trans_tile, 4)
//  m->cstride = c + (64 - (c % 64));

  // 64 tiles speeds up transpose but hurts everything else
  m->rstride = r + (4 - (r % 4));
  m->cstride = c + (4 - (c % 4));

  // allocate space for matrix, 16 byte aligned
  m->cpu_data = (float*)_mm_malloc((m->rstride)*(m->cstride)*
                                   sizeof(float), 16);
  if (m->cpu_data == NULL)
    errcheck_cpu();

  // allocate space for last column holder
  m->cv = (float*)malloc((m->r)*sizeof(float));
  if (m->cv == NULL)
    errcheck_cpu();

  // zero out data for padding
  memset(m->cpu_data, '\0', (m->rstride)*(m->cstride)*sizeof(float));
  memset(m->cv, '\0', (m->r)*sizeof(float));

  // build row-col indices
  build_drc(m);
}

// initialize matrix m from ascii table
void matrix_init_file(matrix *m, char *file_name)
{
  // will need to do this dynamically if
  // we ever want to handle big tables!!!
  float table_data[line_buff_size][line_buff_size];
  unsigned r = 0, c = 0;

  // open file & setup input buffer
  FILE *table = fopen(file_name, "r");
  char  line_buffer[line_buff_size];

  // check if we were even able to open table file
  if (file_name == NULL)
    errcheck_cpu();

  // buckle-up, don't drink and code
  memset(table_data, '\0', line_buff_size*line_buff_size*sizeof(float));

  // for each line in table (row)
  while (fgets(line_buffer, line_buff_size, table))
  {
    if (DEBUG > 5)
      printf("line buffer: %s\n", line_buffer);

    // set up string tokenizer on input buffer
    char *cur_val = strtok(line_buffer, " "); // Note reentrant or thread safe!!

    // don't increment num rows on blank line
    if (cur_val != NULL)
    {
      c = 0; // new row, reset col counter

      // for each token (col)
      while (cur_val != NULL)
      {
        // convert from char to float
        table_data[r][c] = atof(cur_val);

        if (DEBUG > 5)
          printf("converting %d %d %f\n", r, c, table_data[r][c]);

        // get next token
        cur_val = strtok(NULL, " ");
        ++c; // increment num cols
      }
      ++r; // increment num rows
    }
  }

  // close file descriptor
  fclose(table);

  // initialize m
  matrix_init(m, r, c);

  // setup for data_at
  unsigned mr, mc;
  float *data = m->cpu_data;
  const unsigned stride = m->cstride;

  // loop through collected data and put into m
  for (mr = 0; mr < r; ++mr)
    for (mc = 0; mc < c; ++mc)
      data_at(mr,mc) = table_data[mr][mc];
}

// load zeros into matrix m
void matrix_load_zero(matrix m)
{
  float *data = m.cpu_data;
  const float *data_end = data + (m.rstride*m.cstride);

  // loop through all values and set to zero
  // memset?
  while (data < data_end)
    *(data++) = 0.0f;
}

// load values from an array
void matrix_load_array(matrix m, float *v)
{
  unsigned r, c, i = 0;

  // set data and stride for data_at
  float *data = m.cpu_data;
  const unsigned stride = m.cstride;

  for (r = 0; r < m.r; ++r)
    for (c = 0; c < m.c; ++c)
      data_at(r,c) = v[i++];
}

// load values from the random uniform distribution
void matrix_load_runif(matrix m, float min, float max)
{
  unsigned r, c;

  // set data and stride for data_at
  float *data = m.cpu_data;
  const unsigned stride = m.cstride;

  const float range = max - min;

  // needs work to be multithreaded!

  // set each non-padding value from random uniform
  for (r = 0; r < m.r; ++r)
    for (c = 0; c < m.c; ++c)
      data_at(r,c) = rand_r(&rand_state) * range / RAND_MAX + min;
}

void matrix_load_testa(matrix m, unsigned n)
{
  unsigned r, c;

  // set data and stride for data_at
  float *data = m.cpu_data;
  const unsigned stride = m.cstride;

  #pragma omp parallel for private(c)
  for (r = 0; r < m.r; ++r)
    for (c = 0; c < m.c; ++c)
      //data_at(r,c) = (float)n * (float)r*c;
      data_at(r,c) = (float)n;
}

void matrix_load_testb(matrix m)
{
  unsigned r, c;

  // set data and stride for data_at
  float *data = m.cpu_data;
  const unsigned stride = m.cstride;

  #pragma omp parallel for private(c)
  for (r = 0; r < m.r; ++r)
    for (c = 0; c < m.c; ++c)
      //data_at(r,c) = (float)r;
      data_at(r,c) = 1.0f;
}

void matrix_load_testc(matrix m)
{
  unsigned r, c;

  // set data and stride for data_at
  float *data = m.cpu_data;
  const unsigned stride = m.cstride;

  #pragma omp parallel for private(c)
  for (r = 0; r < m.r; ++r)
    for (c = 0; c < m.c; ++c)
      //data_at(r,c) = (float)c;
      data_at(r,c) = 1.0f;
}

// copy values to an array
void matrix_unload_array(matrix m, float *v)
{
  unsigned r, c, i = 0;

  float *data = m.cpu_data;
  const unsigned stride = m.cstride;

  for (r = 0; r < m.r; ++r)
    for (c = 0; c < m.c; ++c)
      v[i++] = data_at(r,c);
}

// write values to a file
void matrix_unload_file(matrix m, char *file_name)
{
  unsigned r, c;
  
  float *data = m.cpu_data;
  const unsigned stride = m.cstride;
  
  // open file for writing
  FILE *table = fopen(file_name, "w");

  // check if we were even able to open table file
  if (file_name == NULL)
    errcheck_cpu();

  for (r = 0; r < m.r; ++r)
  {
    for (c = 0; c < m.c-1; ++c)
      fprintf(table, "%.16f ", data_at(r,c));
    fprintf(table, "%.16f\n", data_at(r,c));
  }

  fclose(table);
}

// destroy matrix m
void matrix_dest(matrix *m)
{
  // free matrix data
  free(m->cpu_data);
  free(m->cv);

  // free row/col indices
  //free(m->drc);

  // set everything to zero for safety
  m->r = 0;
  m->c = 0;
  m->rstride = 0;
  m->cstride = 0;
  m->cpu_data = NULL;
}

//** Synchronization

// wait for asynchronous calls to finish
void matrix_wait()
{
  // nada
}

//** Addition/Subtraction

// a = b + c
void matrix_add(matrix a, matrix b, matrix c)
{
  #ifndef NO_ERRCHECK
    if ( (a.r != b.r) || (b.r != c.r) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Rows don't match for addition!\n", __LINE__);
      exit(CPU_ERROR);
    }

    if ( (a.c != b.c) || (b.c != c.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Cols don't match for addition!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  const unsigned n = a.r * a.cstride;

/* naive
  unsigned i;
  float *aval = a.cpu_data;
  float *bval = b.cpu_data;
  float *cval = c.cpu_data;
  for (i = 0; i < n; ++i)
    aval[i] = bval[i] + cval[i]; */

  if (n < add_lim)
    add_small(a.cpu_data, b.cpu_data, c.cpu_data, n);
  else
    add_big(a.cpu_data, b.cpu_data, c.cpu_data, n);
}

// a = b - c
void matrix_sub(matrix a, matrix b, matrix c)
{
  #ifndef NO_ERRCHECK
    if ( (a.r != b.r) || (b.r != c.r) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Rows don't match for subtraction!\n", __LINE__);
      exit(CPU_ERROR);
    }

    if ( (a.c != b.c) || (b.c != c.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Cols don't match for subtraction!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  const unsigned n = a.r * a.cstride;

  if (n < add_lim)
    sub_small(a.cpu_data, b.cpu_data, c.cpu_data, n);
  else
    sub_big(a.cpu_data, b.cpu_data, c.cpu_data, n);
}

//** Multiplication

// a = b * c
void matrix_mult(matrix a, matrix b, matrix c)
{
  #ifndef NO_ERRCHECK
    if ( (b.c != c.r) || (a.r != b.r) || (a.c != c.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Dimensions don't match for multiplication!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  // use cblas for multiplication, not going to beat this.. for now ;)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              b.r, c.c, b.c, 1.0f,
              b.cpu_data, b.cstride, c.cpu_data, c.cstride,
              0.0f, a.cpu_data, a.cstride);
}

// a = phi(b * c)
void matrix_mult_phi(matrix a, matrix b, matrix c)
{
  #ifndef NO_ERRCHECK
    if ( (b.c != c.r) || (a.r != b.r) || (a.c != c.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Dimensions don't match for mult_phi!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  // use cblas for multiplication, not going to beat this.. for now ;)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              b.r, c.c, b.c, 1.0f,
              b.cpu_data, b.cstride, c.cpu_data, c.cstride,
              0.0f, a.cpu_data, a.cstride);

  // apply phi

  unsigned row, col;

  float *adata = a.cpu_data;
  const unsigned astride = a.cstride;

  // need to optimize small kern and put into functions
  if ((a.r*a.cstride) < add_lim)
  {
    const __m128 mm_s1 = _mm_set1_ps(1.7159f);
    const __m128 mm_s2 = _mm_set1_ps(2.0f/3.0f);
    for (row = 0; row < a.r; ++row)
      for (col = 0; col < a.c; col += 4)
      {
        unsigned i;
        const unsigned base = row*astride+col;

        // inside scalar mult
        __m128 mm_v = _mm_load_ps(adata+base);
        mm_v = _mm_mul_ps(mm_v, mm_s2);
        _mm_store_ps(adata+base, mm_v);

        // apply tanh
        for (i = base; i < base+4; ++i)
          adata[i] = tanh(adata[i]);

        // outside scalar mult
        mm_v = _mm_load_ps(adata+base);
        mm_v = _mm_mul_ps(mm_v, mm_s1);
        _mm_store_ps(adata+base, mm_v);
      }
  }
  else
  {
    const __m128 mm_s1 = _mm_set1_ps(1.7159f);
    const __m128 mm_s2 = _mm_set1_ps(2.0f/3.0f);
    #pragma omp parallel for private(col)
    for (row = 0; row < a.r; ++row)
      for (col = 0; col < a.c; col += 4)
      {
        unsigned i;
        const unsigned base = row*astride+col;

        // inside scalar mult
        __m128 mm_v = _mm_load_ps(adata+base);
        mm_v = _mm_mul_ps(mm_v, mm_s2);
        _mm_store_ps(adata+base, mm_v);

        // apply tanh
        for (i = base; i < base+4; ++i)
          adata[i] = tanh(adata[i]);

        // outside scalar mult
        mm_v = _mm_load_ps(adata+base);
        mm_v = _mm_mul_ps(mm_v, mm_s1);
        _mm_store_ps(adata+base, mm_v);
      }
  }

/*
  unsigned i;
  #pragma omp parallel for
  for (i = 0; i < a.r*a.cstride; ++i)
    adata[i] = phi(adata[i]);
*/
}

//** Transpose

// a = b^T
void matrix_trans(matrix a, matrix b)
{
  #ifndef NO_ERRCHECK
    if ( (a.r != b.c) || (a.c != b.r) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Source and destination dimensions don't match for transpose!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  unsigned r, c;

  float *adata = a.cpu_data;
  float *bdata = b.cpu_data;

  const unsigned astride = a.cstride;
  const unsigned bstride = b.cstride;

  // need to seperate into smaller functions
  if ((a.r*a.c) < trans_lim)
  {
    for (r = 0; r < a.r; r += 4)
      for (c = 0; c < a.c; c += 4)
      {
        // load 4x4 tile
        float *base = bdata + c*bstride+r;
        __m128 mm_v1 = _mm_load_ps(base            );
        __m128 mm_v2 = _mm_load_ps(base +   bstride);
        __m128 mm_v3 = _mm_load_ps(base + 2*bstride);
        __m128 mm_v4 = _mm_load_ps(base + 3*bstride);

        // transpose 4x4 tile
        _MM_TRANSPOSE4_PS(mm_v1, mm_v2, mm_v3, mm_v4);

        // store 4x4 tile back into a
        base = adata + r*astride+c;
        _mm_store_ps(base,             mm_v1);
        _mm_store_ps(base +   astride, mm_v2);
        _mm_store_ps(base + 2*astride, mm_v3);
        _mm_store_ps(base + 3*astride, mm_v4);
      }
  }
  else
  {
    #pragma omp parallel for private(c)
    for (r = 0; r < a.r; r += 4)
      for (c = 0; c < a.c; c += 4)
      {
        // load 4x4 tile
        float *base = bdata + c*bstride+r;
        __m128 mm_v1 = _mm_load_ps(base            );
        __m128 mm_v2 = _mm_load_ps(base +   bstride);
        __m128 mm_v3 = _mm_load_ps(base + 2*bstride);
        __m128 mm_v4 = _mm_load_ps(base + 3*bstride);

        // transpose 4x4 tile
        _MM_TRANSPOSE4_PS(mm_v1, mm_v2, mm_v3, mm_v4);

        // store 4x4 tile back into a
        base = adata + r*astride+c;
        _mm_store_ps(base,             mm_v1);
        _mm_store_ps(base +   astride, mm_v2);
        _mm_store_ps(base + 2*astride, mm_v3);
        _mm_store_ps(base + 3*astride, mm_v4);
      }
  }

  /* tiled version, fast but big tiles hurt everything else
    #pragma omp parallel for private(tc,r,c)
    for (tr = 0; tr < a.r; tr += trans_tile)
      for (tc = 0; tc < a.c; tc += trans_tile)
        for (r = tr; r < tr+trans_tile; r += 4)
          for (c = tc; c < tc+trans_tile; c += 4)
          {
            // load 4x4 tile
            float *base = bdata + c*bstride+r;
            __m128 mm_v1 = _mm_load_ps(base            );
            __m128 mm_v2 = _mm_load_ps(base +   bstride);
            __m128 mm_v3 = _mm_load_ps(base + 2*bstride);
            __m128 mm_v4 = _mm_load_ps(base + 3*bstride);

            // transpose 4x4 tile
            _MM_TRANSPOSE4_PS(mm_v1, mm_v2, mm_v3, mm_v4);

            // store 4x4 tile back into a
            base = adata + r*astride+c;
            _mm_store_ps(base,             mm_v1);
            _mm_store_ps(base +   astride, mm_v2);
            _mm_store_ps(base + 2*astride, mm_v3);
            _mm_store_ps(base + 3*astride, mm_v4);
          } */
}

//** Pointwise miscellaneous

// a = b^2
void matrix_sqr(matrix a, matrix b)
{
  #ifndef NO_ERRCHECK
    if ( (a.r != b.r) || (a.c != b.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Source and destination dimensions don't match for square!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  const unsigned n = a.r * a.cstride;

  if (n < add_lim)
    sqr_small(a.cpu_data, b.cpu_data, n);
  else
    sqr_big(a.cpu_data, b.cpu_data, n);
}

// scalar multiplication a = b*c
void matrix_scl(matrix a, float b, matrix c)
{
  #ifndef NO_ERRCHECK
    if ( (a.r != c.r) || (a.c != c.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Source and destination dimensions don't match for scale!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  const unsigned n = a.r * a.cstride;

  if (n < add_lim)
    scl_small(a.cpu_data, b, c.cpu_data, n);
  else
    scl_big(a.cpu_data, b, c.cpu_data, n);
}

// scalar multiplication & pointwise addition a = b.*c+d
void matrix_scl_add(matrix a, float b, matrix c, matrix d)
{
  #ifndef NO_ERRCHECK
    if ( (a.r != c.r) || (a.c != c.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Source and destination dimensions don't match for scl_add!\n", __LINE__);
      exit(CPU_ERROR);
    }
    if ( (a.r != d.r) || (a.c != d.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Source and destination dimensions don't match for scl_add!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  const unsigned n = a.r * a.cstride;

  if (n < add_lim)
    scl_add_small(a.cpu_data, b, c.cpu_data, d.cpu_data, n);
  else
    scl_add_big(a.cpu_data, b, c.cpu_data, d.cpu_data, n);
}

// pointwise multiplication a = b.*c
void matrix_pmult(matrix a, matrix b, matrix c)
{
  #ifndef NO_ERRCHECK
    if ( (a.r != b.r) || (b.r != c.r) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Rows don't match for pmult!\n", __LINE__);
      exit(CPU_ERROR);
    }

    if ( (a.c != b.c) || (b.c != c.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Cols don't match for pmult!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  const unsigned n = a.r * a.cstride;

  if (n < add_lim)
    pmult_small(a.cpu_data, b.cpu_data, c.cpu_data, n);
  else
    pmult_big(a.cpu_data, b.cpu_data, c.cpu_data, n);
}

// 
void matrix_phi_prime(matrix a, matrix b)
{
  unsigned r, c;

  const unsigned astride = a.cstride;
  const unsigned bstride = b.cstride;

  float *adata = a.cpu_data;
  float *bdata = b.cpu_data;

  const __m128 mm_s1 = _mm_set1_ps(2.0f/3.0f);
  const __m128 mm_s2 = _mm_set1_ps(1.7159f);

  // need to optimize small kern and put into functions
  if ((a.r*a.cstride) < add_lim)
  {
    for (r = 0; r < a.r; ++r)
      for (c = 0; c < a.c; c += 4)
      {
        __m128 mm_v1 = _mm_load_ps(bdata + r*bstride+c);

        mm_v1 = _mm_mul_ps(mm_v1, mm_v1);

        mm_v1 = _mm_sub_ps(mm_s2, mm_v1);

        mm_v1 = _mm_mul_ps(mm_s1, mm_v1);

        _mm_store_ps(adata + r*astride+c, mm_v1);
      }
  }
  else
  {
    #pragma omp parallel for private(c)
    for (r = 0; r < a.r; ++r)
      for (c = 0; c < a.c; c += 4)
      {
        __m128 mm_v1 = _mm_load_ps(bdata + r*bstride+c);

        mm_v1 = _mm_mul_ps(mm_v1, mm_v1);

        mm_v1 = _mm_sub_ps(mm_s2, mm_v1);

        mm_v1 = _mm_mul_ps(mm_s1, mm_v1);

        _mm_store_ps(adata + r*astride+c, mm_v1);
      }
  }

//  const float scl = 2.0f/3.0f;
//  return scl * (1.7159f - z*z);

/*
  #pragma omp parallel for private(c)
  for (r = 0; r < a.r; ++r)
    for (c = 0; c < a.c; ++c)
      adata[r*astride + c] =
        phi_prime(bdata[r*bstride + c]);
*/
}

// a = b .* phi_prime(c)
void matrix_pmult_phi_prime(matrix a, matrix b, matrix c)
{
  unsigned i;

  float *adata = a.cpu_data;
  float *bdata = b.cpu_data;
  float *cdata = c.cpu_data;

  const __m128 mm_s1 = _mm_set1_ps(2.0f/3.0f);
  const __m128 mm_s2 = _mm_set1_ps(1.7159f);

  #pragma omp parallel for
  for (i = 0; i < a.r*a.cstride; i += 4)
    {
      __m128 mm_v1 = _mm_load_ps(bdata+i);
      __m128 mm_v2 = _mm_load_ps(cdata+i);

      mm_v2 = _mm_mul_ps(mm_v2, mm_v2);

      mm_v2 = _mm_sub_ps(mm_s2, mm_v2);

      mm_v2 = _mm_mul_ps(mm_s1, mm_v2);

      mm_v2 = _mm_mul_ps(mm_v1, mm_v2);

      _mm_store_ps(adata + i, mm_v2);
    }
}

//
void matrix_delta(matrix delta, matrix y, matrix g)
{
  unsigned r, c;

  #ifndef NO_ERRCHECK
    if ( (delta.r != y.r) || (y.r != g.r) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Rows don't match for delta!\n", __LINE__);
      exit(CPU_ERROR);
    }

    if ( (delta.c != y.c) || (y.c != g.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Cols don't match for delta!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  const unsigned dstride = delta.cstride;
  const unsigned gstride = g.cstride;
  const unsigned ystride = y.cstride;

  float *ddata = delta.cpu_data;
  float *gdata = g.cpu_data;
  float *ydata = y.cpu_data;

  // need to optimize small kern and put into functions
  if ((delta.r*delta.cstride) < add_lim)
  {
    const __m128 mm_s = _mm_set1_ps(2.0f/(g.r*g.c));
    for (r = 0; r < delta.r; ++r)
      for (c = 0; c < delta.c; c += 4)
      {
        __m128 mm_v1 = _mm_load_ps(gdata + r*gstride+c);
        __m128 mm_v2 = _mm_load_ps(ydata + r*ystride+c);

        mm_v1 = _mm_sub_ps(mm_v2, mm_v1);
        mm_v1 = _mm_mul_ps(mm_v1, mm_s);

        _mm_store_ps(ddata + r*dstride+c, mm_v1);
      }
  }
  else
  {
    const __m128 mm_s = _mm_set1_ps(2.0f/(g.r*g.c));
    #pragma omp parallel for private (c)
    for (r = 0; r < delta.r; ++r)
      for (c = 0; c < delta.c; c += 4)
      {
        __m128 mm_v1 = _mm_load_ps(gdata + r*gstride+c);
        __m128 mm_v2 = _mm_load_ps(ydata + r*ystride+c);

        mm_v1 = _mm_sub_ps(mm_v2, mm_v1);
        mm_v1 = _mm_mul_ps(mm_v1, mm_s);

        _mm_store_ps(ddata + r*dstride+c, mm_v1);
      }
  }

/*
  const float denom = 2.0f / (g.r*g.c);

  #pragma omp parallel for private(c)
  for (r = 0; r < delta.r; ++r)
    for (c = 0; c < delta.c; ++c)
    {
      const float gval = gdata[r*gstride + c];
      const float yval = ydata[r*ystride + c];

      // wow, sse could be used all over here!
      ddata[r*dstride + c] = 
        (yval - gval) * denom;
    }
*/
}

//** Combination/Separation

// remove last row of m
void matrix_r0(matrix *m)
{
  --(m->r);

  float *data = (m->cpu_data)+((m->r)*(m->cstride));
  const float *data_end = data + m->c;

  while (data < data_end)
    *(data++) = 0.0f;
}

// append a row of 1's to m
void matrix_r1(matrix *m)
{
  float *data = (m->cpu_data)+((m->r)*(m->cstride));
  const float *data_end = data + m->c;

  while (data < data_end)
    *(data++) = 1.0f;

  ++(m->r);
}

//** Error measurement

// rmse between values of actual and approx
float matrix_rmse(matrix actual, matrix approx)
{
  // if dims don't match throw error!
  
  unsigned r, c;
  const unsigned stride = actual.cstride;
  const unsigned len = actual.r*actual.c;
  float *d1 = actual.cpu_data;
  float *d2 = approx.cpu_data;
  float err = 0.0f;

  // need to use omp, maybe reduction? 
  for (r = 0; r < actual.r; ++r)
    for (c = 0; c < actual.c; ++c)
    {
      const unsigned i = r*stride+c;
      err += (d1[i] - d2[i])*(d1[i] - d2[i]);
    }

  return sqrt(err/len);
}

// remove and save last col of m
void matrix_c0v(matrix *m)
{
  unsigned r;
  const unsigned stride = m->cstride;
  const unsigned c = --(m->c);
  float *data = m->cpu_data;

  for (r = 0; r < m->r; ++r)
  {
    float *cur = data + r*stride+c;
    m->cv[r] = *cur;
    *cur = 0.0f;
  }
}

// restore last col of m
void matrix_cv(matrix *m)
{
  unsigned r;

  const unsigned stride = m->cstride;
  const unsigned c = (m->c)++;
  float *data = m->cpu_data;

  for (r = 0; r < m->r; ++r)
    data[r*stride+c] = m->cv[r];
}

//** Output

// print m to standard out
void matrix_print(matrix m)
{
  unsigned r, c;
  const unsigned stride = m.cstride;
  float *data = m.cpu_data;

  // print each value to stdout
  for (r = 0; r < m.r; ++r)
  {
    for (c = 0; c < m.c; ++c)
      printf("%f ", data_at(r,c));
    printf("\n");
  }
}

// print m including padding to standard out
void matrix_print_padded(matrix m)
{
  unsigned r, c;

  // set data and stride for data_at
  const unsigned stride = m.cstride;
  float *data = m.cpu_data;

  // print each value to stdout
  for (r = 0; r < m.rstride; ++r)
  {
    for (c = 0; c < m.cstride; ++c)
      printf("%f ", data_at(r,c));
    printf("\n");
  }
}
