/*********************************\
* Cuda Accelerated Matrix Library *
*                                 *
* by                              *
*   Elliott Forney                *
*   3.9.2010                      *
\*********************************/

/* need to fix all kernels to handle matrices with diff strides
   or else ensure that all matrices will have same stride in r0,v1,c0v,cv */

/*
 *  Libraries
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <omp.h>

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

// when rows/cols < rlim/clim use small kernel
// must be greater than or equal to add_big_tpb
#define add_rlim 512
#define add_clim 512

// threads per block for small addition kernel
#define add_small_tpb 64

// threads per block for big addition kernel
// this must be smaller than and divide
// lcm(mult_tn,2*mult_tp) evenly
#define add_big_tpb 128

// stripe size for big addition kernel
#define add_big_stripe 4

//** Multiplication

// tile sizes
#define mult_tm 20
#define mult_tn 16
#define mult_tp 64

/* small mult tile sizes, uncomment for debugging 
#define mult_tm 4
#define mult_tn 4
#define mult_tp 4 */

//** Transpose

// tile sizes
#define trans_tile_r 4
#define trans_tile_c 32

// virtualization
#define trans_stripe 8

/* small tiles, uncomment for debugging
#define trans_tile_r 2
#define trans_tile_c 4
#define trans_stripe 2 */

//** Combination/Separation

// threads per block for add/remove row/col kernels
#define r0_tpb 64

/*
 *  Global variables
 */

//** random generator stuff needs to be redone, non-reentrant

unsigned rand_state = 7;

/*
 *  Kernel function prototypes
 */

// sigmoid function
__device__ float phi(float v);

// derivative of sigmoid function given phi(v)
__device__ float phi_prime(float z);

// addition kernel for small matrices
__global__ void add_small_kern(float *a, float *b, float *c, unsigned n);

// addition kernel for big matrices
__global__ void add_big_kern(float *a, float *b, float *c, unsigned n, unsigned stride);

// subtraction kernel for small matrices
__global__ void sub_small_kern(float *a, float *b, float *c, unsigned n);

// subtraction kernel for big matrices
__global__ void sub_big_kern(float *a, float *b, float *c, unsigned n, unsigned stride);

// matrix multiplication kernel
__global__ void mult_kern(float *a, unsigned a_c,
                          float *b, unsigned b_c, unsigned b_csub,
                          float *c, unsigned c_c);

// matrix multiplication plus component-wise function apply
__global__ void mult_phi_kern(float *a, unsigned a_c,
                              float *b, unsigned b_c, unsigned b_csub,
                              float *c, unsigned c_c);

// transpose kernel
__global__ void trans_kern(float *a, float *b, unsigned nrow,
                           unsigned astride, unsigned bstride);

// square kernel for small matrices
__global__ void sqr_small_kern(float *a, float *b, unsigned n);

// square kernel for large matrices
__global__ void sqr_big_kern(float *a, float *b, unsigned n, unsigned stride);

// scalar multiply kernel for small matrices
__global__ void scl_small_kern(float *a, float b, float *c, unsigned n);

// scalar multiply kernel for big matrices
__global__ void scl_big_kern(float *a, float b, float *c, unsigned n, unsigned stride);

// scalar multiply & pointwise addition kernel for small matrices
__global__ void scl_add_small_kern(float *a, float b, float *c,
                                   float *d, unsigned n);

// scalar multiply & pointwise addition kernel for big matrices
__global__ void scl_add_big_kern(float *a, float b, float *c,
                                 float *d, unsigned n, unsigned stride);

// pointwise multiply kernel for small matrices
__global__ void pmult_small_kern(float *a, float *b, float *c, unsigned n);

// pointwise multiply kernel for big matrices
__global__ void pmult_big_kern(float *a, float *b, float *c, unsigned n, unsigned stride);

//
__global__ void phi_prime_small_kern(float *a, float *b, unsigned n);

//
__global__ void phi_prime_big_kern(float *a, float *b, unsigned n, unsigned stride);

//
__global__ void delta_small_kern(float *a, float *b, float *c, float denom, unsigned n);

//
__global__ void delta_big_kern(float *a, float *b, float *c, float denom,
                               unsigned n, unsigned stride);

// remove last row
__global__ void r0_kern(float *m, unsigned nrow, unsigned ncol);

// add row of ones kernel
__global__ void r1_kern(float *m, unsigned nrow, unsigned ncol);

//
__global__ void c0v_kern(float *a, float *b, unsigned nrow, unsigned stride);

//
__global__ void cv_kern(float *a, float *b, unsigned nrow, unsigned stride);

// zero out all values in m for small matrices
__global__ void zero_small_kern(float *m, unsigned n);

// zero out all values in m for big matrices
__global__ void zero_big_kern(float *m, unsigned n, unsigned stride);

/*
 *  Kernel function bodies
 */

// sigmoid function
__device__ float phi(float v)
{
  // recommended by Lecun, find citation!!
  return 1.7159f * tanh( (2.0f/3.0f) * v );
}

// derivative of sigmoid function given phi(v)
__device__ float phi_prime(float z)
{
  return (2.0f/3.0f) * (1.7159f - z*z);
}

// addition kernel for small matrices
__global__ void add_small_kern(float *a, float *b, float *c, unsigned n)
{
  // unique id for each thread 0, ..., (nthreads-1)
  const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  // if inside matrix
  if (id < n)
    // sum one value
    a[id] = b[id] + c[id];
}

// addition kernel for big matrices
__global__ void add_big_kern(float *a, float *b, float *c, unsigned n, unsigned stride)
{ 
  unsigned i;

  // unique id for each block, strided according to stripe size
  const unsigned block_index = blockIdx.x + blockIdx.y * gridDim.x * add_big_stripe;

  // unique id for each thread 
  const unsigned id = threadIdx.x + block_index * blockDim.x;

  // each thread sums down a column stripe times
  #pragma unroll    
  for (i = id; i < id+add_big_stripe*stride; i += stride)
    if (i < n)  // if inside matrix
      a[i] = b[i] + c[i]; // sum value
}

// subtraction kernel for small matrices
__global__ void sub_small_kern(float *a, float *b, float *c, unsigned n)
{
  // unique id for each thread 0, ..., (nthreads-1)
  const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  // if inside matrix
  if (id < n)
    // subtract one value
    a[id] = b[id] - c[id];
}

// subtraction kernel for big matrices
__global__ void sub_big_kern(float *a, float *b, float *c, unsigned n, unsigned stride)
{ 
  unsigned i;

  // unique id for each block, strided according to stripe size
  const unsigned block_index = blockIdx.x + blockIdx.y * gridDim.x * add_big_stripe;

  // unique id for each thread 
  const unsigned id = threadIdx.x + block_index * blockDim.x;

  // each thread sums down a column stripe times
  #pragma unroll 
  for (i = id; i < id+add_big_stripe*stride; i += stride)
    if (i < n)  // if inside matrix
      a[i] = b[i] - c[i]; // subtract value
}

// matrix multiplication kernel
// expects padded to tile sizes
__global__ void mult_kern(float *a, unsigned a_c,
                          float *b, unsigned b_c, unsigned b_csub,
                          float *c, unsigned c_c)
{
  // loop counter
  unsigned i, j;

  // starting positions in respective tiles
  float *acur, *bcur, *ccur;
  {
    const unsigned block_pos_r = blockIdx.y*mult_tm;
    const unsigned block_pos_c = blockIdx.x*2*mult_tp;

    acur = a + block_pos_c + block_pos_r * a_c +
           threadIdx.x + threadIdx.y * mult_tn;
    bcur = b + (block_pos_r + threadIdx.y) * b_c + threadIdx.x;
    ccur = c + block_pos_c + threadIdx.x + threadIdx.y * mult_tn;
  }

  // end of last tile in b
  const float *bend = bcur + b_csub;

  // current a values
  // two way virtualization
  float aval_v1[mult_tm];
  float aval_v2[mult_tm];

  // initialize a values to zero
  #pragma unroll
  for (i = 0; i < mult_tm; ++i)
  {
    aval_v1[i] = 0.0f;
    aval_v2[i] = 0.0f;
  }

  // for each tile read from b
  do
  {
    // allocate shared space for tile in b
    __shared__ float bs[mult_tn][mult_tm+1];

    // put tile from b into shared memory
    #pragma unroll
    for (i = 0; i < mult_tm; i += (mult_tp/mult_tn))
      bs[threadIdx.x][threadIdx.y+i] = bcur[i*b_c];

    // move b's tile across
    bcur += mult_tn;

    // synchronize to ensure bll elements are read
    __syncthreads();

    // for each row in tile of c
    #pragma unroll
    for (i = 0; i < mult_tn; ++i)
    {
      // do mults and adds
      #pragma unroll
      for (j = 0; j < mult_tm; ++j)
      {
        aval_v1[j] += bs[i][j] * ccur[0];
        aval_v2[j] += bs[i][j] * ccur[mult_tp];
      }

      ccur += c_c;
    }

    __syncthreads();
  }
  while (bcur < bend); // until last tile in b

  // copy results to global memory
  #pragma unroll
  for (i = 0; i < mult_tm; ++i, acur += a_c)
  {
    acur[0]       = aval_v1[i];
    acur[mult_tp] = aval_v2[i];
  }
}

// matrix multiplication plus component-wise function apply
__global__ void mult_phi_kern(float *a, unsigned a_c,
                              float *b, unsigned b_c, unsigned b_csub,
                              float *c, unsigned c_c)
{
  // loop counter
  unsigned i, j;

  // starting positions in respective tiles
  float *acur, *bcur, *ccur;
  {
    const unsigned block_pos_r = blockIdx.y*mult_tm;
    const unsigned block_pos_c = blockIdx.x*2*mult_tp;

    acur = a + block_pos_c + block_pos_r * a_c +
           threadIdx.x + threadIdx.y * mult_tn;
    bcur = b + (block_pos_r + threadIdx.y) * b_c + threadIdx.x;
    ccur = c + block_pos_c + threadIdx.x + threadIdx.y * mult_tn;
  }

  // end of last tile in b
  const float *bend = bcur + b_csub;

  // current a values
  // two way virtualization
  float aval_v1[mult_tm];
  float aval_v2[mult_tm];

  // initialize a values to zero
  #pragma unroll
  for (i = 0; i < mult_tm; ++i)
  {
    aval_v1[i] = 0.0f;
    aval_v2[i] = 0.0f;
  }

  // for each tile read from b
  do
  {
    // allocate shared space for tile in b
    __shared__ float bs[mult_tn][mult_tm+1];

    // put tile from b into shared memory
    #pragma unroll
    for (i = 0; i < mult_tm; i += (mult_tp/mult_tn))
      bs[threadIdx.x][threadIdx.y+i] = bcur[i*b_c];

    // move b's tile across
    bcur += mult_tn;

    // synchronize to ensure bll elements are read
    __syncthreads();

    // for each row in tile of c
    #pragma unroll
    for (i = 0; i < mult_tn; ++i)
    {
      // do mults and adds
      #pragma unroll
      for (j = 0; j < mult_tm; ++j)
      {
        aval_v1[j] += bs[i][j] * ccur[0];
        aval_v2[j] += bs[i][j] * ccur[mult_tp];
      }

      ccur += c_c;
    }

    __syncthreads();
  }
  while (bcur < bend); // until last tile in b

  // copy results to global memory
  #pragma unroll
  for (i = 0; i < mult_tm; ++i, acur += a_c)
  {
    acur[0]       = phi(aval_v1[i]);
    acur[mult_tp] = phi(aval_v2[i]);
  }
}

// transpose kernel
// expects padded to tile size
__global__ void trans_kern(float *a, float *b, unsigned nrow,
                           unsigned astride, unsigned bstride)
{
  unsigned i, blockIdx_x, blockIdx_y;

  // diagonal reordering to prevent partition camping
  // borrowed from NVIDIA CUDA SDK, Thanks!
  if (nrow == astride)
  {
    blockIdx_y = blockIdx.x;
    blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
  }
  else
  {
    const unsigned bid = blockIdx.x + gridDim.x*blockIdx.y;
    blockIdx_y = bid%gridDim.y;
    blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
  }

  //
  const unsigned tile_r_stripe = trans_tile_r * trans_stripe;
  const unsigned tid_y_stripe  = threadIdx.y  * trans_stripe;

  // starting row and col in a
  const unsigned block_row = blockIdx_y * tile_r_stripe;
  const unsigned block_col = blockIdx_x * trans_tile_c;

  // thread's row and col in b
  unsigned row = block_col + tid_y_stripe;
  unsigned col = block_row + threadIdx.x;

  // perform tile transpose in shared memory
  __shared__ float tile[trans_tile_c][tile_r_stripe+1];

  unsigned base = row*bstride + col;

  // read transposed values in from b
  #pragma unroll
  for (i = 0; i < trans_stripe; ++i)
    tile[threadIdx.x][tid_y_stripe+i] = b[base+i*bstride];

  // wait for all threads to finish reading into shared mem
  __syncthreads();

  // thread's row and col in a
  row = block_row + tid_y_stripe;
  col = block_col + threadIdx.x;

  base = row*astride + col;

  // write tile into a
  #pragma unroll
  for (i = 0; i < trans_stripe; ++i)
    a[base+i*astride] = tile[tid_y_stripe+i][threadIdx.x];
}

// square kernel for small matrices
__global__ void sqr_small_kern(float *a, float *b, unsigned n)
{
  // unique id for each thread 0, ..., (nthreads-1)
  const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  // if inside matrix
  if (id < n)
  {
    // square single value
    float bval = b[id];
    a[id] = bval*bval;
  }
}

// square kernel for large matrices
__global__ void sqr_big_kern(float *a, float *b, unsigned n, unsigned stride)
{
  unsigned i;

  // unique id for each block, strided according to stripe size
  const unsigned block_index = blockIdx.x + blockIdx.y * gridDim.x * add_big_stripe;

  // unique id for each thread 
  const unsigned id = threadIdx.x + block_index * blockDim.x;

  // each thread operates down a column stripe times
  #pragma unroll    
  for (i = id; i < id+add_big_stripe*stride; i += stride)
    if (i < n)
    {
      float bval = b[i];
      a[i] = bval*bval;
    }
}

// scalar multiply kernel for small matrices
__global__ void scl_small_kern(float *a, float b, float *c, unsigned n)
{
  // unique id for each thread 0, ..., (nthreads-1)
  const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  // if inside matrix
  if (id < n)
    a[id] = b*c[id];
}

// scalar multiply kernel for big matrices
__global__ void scl_big_kern(float *a, float b, float *c, unsigned n, unsigned stride)
{
  unsigned i;

  // unique id for each block, strided according to stripe size
  const unsigned block_index = blockIdx.x + blockIdx.y * gridDim.x * add_big_stripe;

  // unique id for each thread 
  const unsigned id = threadIdx.x + block_index * blockDim.x;

  // each thread operates down a column stripe times
  #pragma unroll    
  for (i = id; i < id+add_big_stripe*stride; i += stride)
    if (i < n)
      a[i] = b*c[id];
}

// scalar multiply & pointwise addition kernel for small matrices
__global__ void scl_add_small_kern(float *a, float b, float *c,
                                   float *d, unsigned n)
{
  // unique id for each thread 0, ..., (nthreads-1)
  const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  // if inside matrix
  if (id < n)
    a[id] = b*c[id]+d[id];
}

// scalar multiply & pointwise addition kernel for big matrices
__global__ void scl_add_big_kern(float *a, float b, float *c,
                                 float *d, unsigned n, unsigned stride)
{
  unsigned i;

  // unique id for each block, strided according to stripe size
  const unsigned block_index = blockIdx.x + blockIdx.y * gridDim.x * add_big_stripe;

  // unique id for each thread 
  const unsigned id = threadIdx.x + block_index * blockDim.x;

  // each thread operates down a column stripe times
  #pragma unroll    
  for (i = id; i < id+add_big_stripe*stride; i += stride)
    if (i < n)
      a[i] = b*c[id]+d[id];
}


// pointwise multiply kernel for small matrices
__global__ void pmult_small_kern(float *a, float *b, float *c, unsigned n)
{
  // unique id for each thread 0, ..., (nthreads-1)
  const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  // if inside matrix
  if (id < n)
    // mult one value
    a[id] = b[id] * c[id];
}

// pointwise multiply kernel for big matrices
__global__ void pmult_big_kern(float *a, float *b, float *c, unsigned n, unsigned stride)
{
  unsigned i;

  // unique id for each block, strided according to stripe size
  const unsigned block_index = blockIdx.x + blockIdx.y * gridDim.x * add_big_stripe;

  // unique id for each thread 
  const unsigned id = threadIdx.x + block_index * blockDim.x;

  // each thread mults down a column stripe times
  #pragma unroll    
  for (i = id; i < id+add_big_stripe*stride; i += stride)
    if (i < n)  // if inside matrix
      a[i] = b[i] * c[i]; // mult value
}

//
__global__ void phi_prime_small_kern(float *a, float *b, unsigned n)
{
  // unique id for each thread 0, ..., (nthreads-1)
  const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  // if inside matrix
  if (id < n)
    // sum one value
    a[id] = phi_prime(b[id]);
}

//
__global__ void phi_prime_big_kern(float *a, float *b, unsigned n, unsigned stride)
{
  unsigned i;

  // unique id for each block, strided according to stripe size
  const unsigned block_index = blockIdx.x + blockIdx.y * gridDim.x * add_big_stripe;

  // unique id for each thread 
  const unsigned id = threadIdx.x + block_index * blockDim.x;

  // each thread applys down a column stripe times
  #pragma unroll    
  for (i = id; i < id+add_big_stripe*stride; i += stride)
    if (i < n)  // if inside matrix
      a[i] = phi_prime(b[i]);
}

//
__global__ void delta_small_kern(float *a, float *b, float *c, float denom, unsigned n)
{
  // unique id for each thread 0, ..., (nthreads-1)
  const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  // if inside matrix
  if (id < n)
    a[id] = (b[id] - c[id]) * denom;
}

//
__global__ void delta_big_kern(float *a, float *b, float *c, float denom,
                               unsigned n, unsigned stride)
{
  unsigned i;

  // unique id for each block, strided according to stripe size
  const unsigned block_index = blockIdx.x + blockIdx.y * gridDim.x * add_big_stripe;

  // unique id for each thread 
  const unsigned id = threadIdx.x + block_index * blockDim.x;

  #pragma unroll    
  for (i = id; i < id+add_big_stripe*stride; i += stride)
    if (i < n)  // if inside matrix
      a[i] = (b[i] - c[i]) * denom;
}

// remove last row kernel
// m should point to beginning of last row
__global__ void r0_kern(float *m, unsigned ncol)
{
  // unique id for each thread 0, ..., (nthreads-1)
  const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  // if not past last col
  if (id < ncol)
    // set a single value to zero
    m[id] = 0.0f;
}

// add row of ones kernel
// m should point to first non-existant row
__global__ void r1_kern(float *m, unsigned ncol)
{
  // unique id for each thread 0, ..., (nthreads-1)
  const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  // if not past last col
  if (id < ncol)
    // set a single value to one
    m[id] = 1.0f;
}

//
__global__ void c0v_kern(float *a, float *b, unsigned nrow, unsigned stride)
{
  const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < nrow)
  {
    float *aval = a + stride * id;
    b[id] = *aval;
    *aval = 0.0f;
  }
}

//
__global__ void cv_kern(float *a, float *b, unsigned nrow, unsigned stride)
{
  const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < nrow)
    a[stride * id] = b[id];
}

// zero out all values in m for small matrices
__global__ void zero_small_kern(float *m, unsigned n)
{
  // unique id for each thread 0, ..., (nthreads-1)
  const unsigned id = threadIdx.x + blockIdx.x * blockDim.x;

  // if inside matrix
  if (id < n)
    // write zero
    m[id] = 0.0f;
}

// zero out all values in m for big matrices
__global__ void zero_big_kern(float *m, unsigned n, unsigned stride)
{
  unsigned i;

  // unique id for each block, strided according to stripe size
  const unsigned block_index = blockIdx.x + blockIdx.y * gridDim.x * add_big_stripe;

  // unique id for each thread 
  const unsigned id = threadIdx.x + block_index * blockDim.x;

  // each thread operates down a column stripe times
  #pragma unroll    
  for (i = id; i < id+add_big_stripe*stride; i += stride)
    if (i < n)
      // write zero
      m[i] = 0.0f;
}

/*
 *  Internal function prototypes
 */

// zero out all values on cpu, does not set sync state!
void zero_cpu(matrix m);

// zero out all values on gpu, does not set sync state!
void zero_gpu(matrix m);

/*
 *  Internal function bodies
 */

// zero out all values on cpu, does not set sync state!
void zero_cpu(matrix m)
{
  memset(m.cpu_data, '\0', (m.rstride)*(m.cstride)*sizeof(float));
}

// zero out all values on gpu, does not set sync state!
void zero_gpu(matrix m)
{
  const unsigned row = m.rstride;
  const unsigned col = m.cstride;

  // if matrix dimensions are smaller than limits
  if ((row < add_rlim) || (col < add_clim))
  {
    // set threads per block as parameterized
    const dim3 block_size(add_small_tpb, 1, 1);

    // treat like vector, num blocks is n / tpb
    const dim3 grid_size((add_small_tpb + row*col - 1) /
                          add_small_tpb, 1, 1);

    // call small zero kernel
    zero_small_kern<<<grid_size, block_size>>>(m.gpu_data, row*col);
    errcheck_gpu();
  }
  else
  {
    // set threads per block as parameterized
    const dim3 block_size(add_big_tpb, 1, 1);

    // across rows we have ncol / tpb blocks
    const unsigned grid_size_x = (add_big_tpb + col - 1) / add_big_tpb;

    // down cols we have nrow / stripe blocks
    const unsigned grid_size_y = (add_big_stripe + row - 1) / add_big_stripe;
    const dim3 grid_size(grid_size_x, grid_size_y, 1);

    // call big zero kernel
    zero_big_kern<<<grid_size, block_size>>>(m.gpu_data, row*col, col);
    errcheck_gpu();
  }
}

/*
 *  External function bodies
 */

//** Initialization & Destruction

// initialize matrix m of size r by c
extern "C" void matrix_init(matrix *m, unsigned r, unsigned c)
{
  // set matrix dimensions
  m->r = r;
  m->c = c;

  // leave extra room for zero padding to make multiple of tile sizes
  // always leave an extra row & col for bias terms

  m->rstride = r + (80  - (r % 80));  // lcm(mult_tn,mult_tm)
  m->cstride = c + (128 - (c % 128)); // lcm(mult_tn,2*mult_tp) 
/*
  m->rstride = r + (640 - (r % 640));
  m->cstride = c + (640 - (c % 640));
*/
  /* uncomment to debug without extra row & col of padding
  if ((r % 80) == 0)
    m->rstride = r;
  else
    m->rstride = r + (80  - (r % 80));
  if ((c % 128) == 0)
    m->cstride = c;
  else
    m->cstride = c + (128 - (c % 128)); */

  /* uncomment to debug with small tiles 4,4,4
  m->rstride = r + (8 - (r % 8));
  m->cstride = c + (8 - (c % 8)); */

  if (DEBUG > 1) {
    printf("matrix size:  %d %d\n", m->r, m->c);
    printf("with padding: %d %d\n", m->rstride, m->cstride);
  }

  // allocate space for sync state
  // done dynamically so we can pass copies of matrices
  // and utilize them without screwing up state
  m->sync = (matrix_sync_state*)malloc(sizeof(matrix_sync_state));
  if (m->sync == NULL)
    errcheck_cpu();

  // set initial sync to cpu
  *(m->sync) = matrix_sync_cpu;

  // allocate space for matrix on gpu
  cudaMalloc((void**)&(m->gpu_data), (m->rstride)*(m->cstride)*sizeof(float));
  errcheck_gpu();

  // allocate space for last column holder
  cudaMalloc((void**)&(m->cv), (m->r)*sizeof(float));
  errcheck_gpu();

  // allocate space for matrix on cpu
  m->cpu_data = (float*)malloc((m->rstride)*(m->cstride)*sizeof(float));
  if (m->cpu_data == NULL)
    errcheck_cpu();

  // zero out all data for safe padding
  zero_cpu(*m);
  zero_gpu(*m);
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
  memset(*table_data, '\0', line_buff_size*line_buff_size*sizeof(float));

  // for each line in table - row
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

      // for each token - col
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
extern "C" void matrix_load_zero(matrix m)
{
  float *data = m.cpu_data;
  const float *data_end = data + (m.rstride*m.cstride);

  // trashing, set sync set to cpu
  *(m.sync) = matrix_sync_cpu;

  // loop through all values and set to zero
  // use memset?
  while (data < data_end)
    *(data++) = 0.0f;
}

// load values from an array
extern "C" void matrix_load_array(matrix m, float *v)
{
  unsigned r, c, i = 0;

  // set data and stride for data_at
  float *data = m.cpu_data;
  const unsigned stride = m.cstride;

  *(m.sync) = matrix_sync_cpu;
  matrix_load_zero(m);

  for (r = 0; r < m.r; ++r)
    for (c = 0; c < m.c; ++c)
      data_at(r,c) = v[i++];
}

// load values from the random uniform distribution
extern "C" void matrix_load_runif(matrix m, float min, float max)
{
  unsigned r, c;

  // set data and stride for data_at
  float *data = m.cpu_data;
  const unsigned stride = m.cstride;

  const float range = max - min;

  *(m.sync) = matrix_sync_cpu;
  matrix_load_zero(m);

  // needs work to be multithreaded!
  // should prolly use GPU?!

  // set each non-padding value from random uniform
  for (r = 0; r < m.r; ++r)
    for (c = 0; c < m.c; ++c)
      data_at(r,c) = rand_r(&rand_state) * range / RAND_MAX + min;
}

extern "C" void matrix_load_testa(matrix m, unsigned n)
{
  unsigned r, c;

  // set data and stride for data_at
  float *data = m.cpu_data;
  const unsigned stride = m.cstride;

  *(m.sync) = matrix_sync_cpu;
  matrix_load_zero(m);

  #pragma omp parallel for private(c)
  for (r = 0; r < m.r; ++r)
    for (c = 0; c < m.c; ++c)
      //data_at(r,c) = (float)n * (float)r*c;
      data_at(r,c) = (float)n;
}

extern "C" void matrix_load_testb(matrix m)
{
  unsigned r, c;

  // set data and stride for data_at
  float *data = m.cpu_data;
  const unsigned stride = m.cstride;

  *(m.sync) = matrix_sync_cpu;
  matrix_load_zero(m);

  #pragma omp parallel for private(c)
  for (r = 0; r < m.r; ++r)
    for (c = 0; c < m.c; ++c)
      data_at(r,c) = (float)r;
      //data_at(r,c) = 1.0f;
}

extern "C" void matrix_load_testc(matrix m)
{
  unsigned r, c;

  // set data and stride for data_at
  float *data = m.cpu_data;
  const unsigned stride = m.cstride;

  *(m.sync) = matrix_sync_cpu;
  matrix_load_zero(m);

  #pragma omp parallel for private(c)
  for (r = 0; r < m.r; ++r)
    for (c = 0; c < m.c; ++c)
      //data_at(r,c) = (float)c;
      data_at(r,c) = 1.0f;
}

// copy values to an array
extern "C" void matrix_unload_array(matrix m, float *v)
{
  unsigned r, c, i = 0;

  float *data = m.cpu_data;
  const unsigned stride = m.cstride;

  matrix_sync_to_cpu(m);

  for (r = 0; r < m.r; ++r)
    for (c = 0; c < m.c; ++c)
      v[i++] = data_at(r,c);
}

// write values to a file
extern "C" void matrix_unload_file(matrix m, char *file_name)
{
  unsigned r, c;

  float *data = m.cpu_data;
  const unsigned stride = m.cstride;

  // open file for writing
  FILE *table = fopen(file_name, "w");

  // check if we were even able to open table file
  if (file_name == NULL)
    errcheck_cpu();

  matrix_sync_to_cpu(m);

  for (r = 0; r < m.r; ++r)
  {
    for (c = 0; c < m.c-1; ++c)
      fprintf(table, "%.16f ", data_at(r,c));
    fprintf(table, "%.16f\n", data_at(r,c));
  }

  fclose(table);
}

// destroy matrix m
extern "C" void matrix_dest(matrix *m)
{
  free(m->cpu_data);      // free matrix on cpu
  free(m->sync);          // free sync state
  cudaFree(m->gpu_data);  // free matrix on gpu
  cudaFree(m->cv);        // free col holder

  // set everything to zero for safety
  m->r        = 0;
  m->c        = 0;
  m->rstride  = 0;
  m->cstride  = 0;
  m->sync     = NULL;
  m->cpu_data = NULL;
  m->gpu_data = NULL;
  m->cv       = NULL;
}

//** cpu/gpu synchronization

// ensure current copy of matrix is on cpu
void matrix_sync_to_cpu(matrix m)
{
  // if not already on cpu
  if (*(m.sync) != matrix_sync_cpu)
  {
    // copy from device memory to host memory
    cudaMemcpy(m.cpu_data, m.gpu_data,
               sizeof(float)*(m.rstride)*(m.cstride),
               cudaMemcpyDeviceToHost);
    errcheck_gpu();

    // set sync state to cpu
    *(m.sync) = matrix_sync_cpu;
  }
}

// ensure current copy of matrix is on gpu
void matrix_sync_to_gpu(matrix m)
{
  // if not already on gpu
  if (*(m.sync) != matrix_sync_gpu)
  {
    // copy from host memory to device memory
    cudaMemcpy(m.gpu_data, m.cpu_data,
               sizeof(float)*(m.rstride)*(m.cstride),
               cudaMemcpyHostToDevice);
    errcheck_gpu();

    // set sync state to gpu
    *(m.sync) = matrix_sync_gpu;
  }
}

// wait for any gpu kernels to finish
void matrix_wait()
{
  cudaDeviceSynchronize();
  errcheck_gpu();
}

//** Access

// return pointer to cpu value at row r and col c
float *matrix_at(matrix m, unsigned r, unsigned c)
{
  // check not out of bounds!

  matrix_sync_to_cpu(m);
  return (m.cpu_data)+(r*m.cstride+c);
}

//** Addition/Subtraction

// a = b + c
extern "C" void matrix_add(matrix a, matrix b, matrix c)
{
  // sync matrices to gpu
  *(a.sync) = matrix_sync_gpu; // trashing, just set
  matrix_sync_to_gpu(b);
  matrix_sync_to_gpu(c);

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

  const unsigned row = b.r;
  const unsigned col = b.cstride;

  // if matrix dimensions are smaller than limits
  if ((row < add_rlim) || (col < add_clim))
  {
    // set threads per block as parameterized
    const dim3 block_size(add_small_tpb, 1, 1);

    // treat like vector, num blocks is n / tpb
    const dim3 grid_size((add_small_tpb + row*col - 1) /
                          add_small_tpb, 1, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    add_small_kern<<<grid_size, block_size>>>(a.gpu_data, b.gpu_data,
                                              c.gpu_data, row*col);
    errcheck_gpu();
  }
  else
  {
    // set threads per block as parameterized
    const dim3 block_size(add_big_tpb, 1, 1);

    // across rows we have ncol / tpb blocks
    const unsigned grid_size_x = (add_big_tpb + col - 1) / add_big_tpb;

    // down cols we have nrow / stripe blocks
    const unsigned grid_size_y = (add_big_stripe + row - 1) / add_big_stripe;
    const dim3 grid_size(grid_size_x, grid_size_y, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    add_big_kern<<<grid_size, block_size>>>(a.gpu_data, b.gpu_data,
                                            c.gpu_data, row*col, col);
    errcheck_gpu();
  }
}

// a = b - c
extern "C" void matrix_sub(matrix a, matrix b, matrix c)
{
  // sync matrices to gpu
  *(a.sync) = matrix_sync_gpu; // trashing, just set
  matrix_sync_to_gpu(b);
  matrix_sync_to_gpu(c);

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

  const unsigned row = b.r;
  const unsigned col = b.cstride;

  // if matrix dimensions are smaller than limits
  if ((row < add_rlim) || (col < add_clim))
  {
    // set threads per block as parameterized
    const dim3 block_size(add_small_tpb, 1, 1);

    // treat like vector, num blocks is n / tpb
    const dim3 grid_size((add_small_tpb + row*col - 1) /
                          add_small_tpb, 1, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    sub_small_kern<<<grid_size, block_size>>>(a.gpu_data, b.gpu_data,
                                              c.gpu_data, row*col);
  }
  else
  {
    // set threads per block as parameterized
    const dim3 block_size(add_big_tpb, 1, 1);

    // across rows we have ncol / tpb blocks
    const unsigned grid_size_x = (add_big_tpb + col - 1) / add_big_tpb;

    // down cols we have nrow / stripe blocks
    const unsigned grid_size_y = (add_big_stripe + row - 1) / add_big_stripe;
    const dim3 grid_size(grid_size_x, grid_size_y, 1);

    if (DEBUG > 0)
    {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    sub_big_kern<<<grid_size, block_size>>>(a.gpu_data, b.gpu_data,
                                            c.gpu_data, row*col, col);
    errcheck_gpu();
  }
}

//** Multiplication

// a = b * c
extern "C" void matrix_mult(matrix a, matrix b, matrix c)
{
  *(a.sync) = matrix_sync_gpu; // trashing, just set
  matrix_sync_to_gpu(b);
  matrix_sync_to_gpu(c);

  #ifndef NO_ERRCHECK
    if ( (b.c != c.r) || (a.r != b.r) || (a.c != c.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Dimensions don't match for multiplication!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  const dim3 block_size(mult_tn, mult_tp/mult_tn, 1);

  // remember, x is col and y is row here
  const unsigned grid_size_r = (mult_tm + a.r - 1) / mult_tm;
  const unsigned grid_size_c = (2*mult_tp + a.c - 1) / (2*mult_tp);

  //const unsigned grid_size_r = a.rstride / mult_tm;
  //const unsigned grid_size_c = a.cstride / (2*mult_tp);
  const dim3 grid_size(grid_size_c, grid_size_r, 1);

  if (DEBUG > 0) {
    printf("block size: %d %d\n", block_size.x, block_size.y);
    printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
  }

  mult_kern<<<grid_size, block_size>>>(a.gpu_data, a.cstride,
                                       b.gpu_data, b.cstride, b.c,
                                       c.gpu_data, c.cstride);
  errcheck_gpu();
}

// a = phi(b * c)
extern "C" void matrix_mult_phi(matrix a, matrix b, matrix c)
{
  *(a.sync) = matrix_sync_gpu; // trashing, just set
  matrix_sync_to_gpu(b);
  matrix_sync_to_gpu(c);

  #ifndef NO_ERRCHECK
    if ( (b.c != c.r) || (a.r != b.r) || (a.c != c.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Dimensions don't match for mult-apply!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  const dim3 block_size(mult_tn, mult_tp/mult_tn, 1);

  // remember, x is col and y is row here
  const unsigned grid_size_r = (mult_tm + a.r - 1) / mult_tm;
  const unsigned grid_size_c = (2*mult_tp + a.c - 1) / (2*mult_tp);

  //const unsigned grid_size_r = a.rstride / mult_tm;
  //const unsigned grid_size_c = a.cstride / (2*mult_tp);
  const dim3 grid_size(grid_size_c, grid_size_r, 1);

  if (DEBUG > 0) {
    printf("block size: %d %d\n", block_size.x, block_size.y);
    printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
  }

  mult_phi_kern<<<grid_size, block_size>>>(a.gpu_data, a.cstride,
                                           b.gpu_data, b.cstride, b.c,
                                           c.gpu_data, c.cstride);
  errcheck_gpu();
}

//** Transpose

// a = b^T
// a = b^T
extern "C" void matrix_trans(matrix a, matrix b)
{
  *(a.sync) = matrix_sync_gpu; // trashing, just set
  matrix_sync_to_gpu(b);

  #ifndef NO_ERRCHECK
    if ( (a.r != b.c) || (a.c != b.r) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Source and destination dimensions don't match for transpose!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  // divide into grid of trans_tile sized block
  const dim3 block_size(trans_tile_c, trans_tile_r, 1);

  const unsigned grid_size_x = (trans_tile_c + a.c - 1) / trans_tile_c;
  const unsigned grid_size_y = ((trans_tile_r * trans_stripe) + a.r - 1) /
                                (trans_tile_r * trans_stripe);
  const dim3 grid_size(grid_size_x, grid_size_y, 1);

  if (DEBUG > 0) {
    printf("block size: %d %d\n", block_size.x, block_size.y);
    printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
  }

  // call transpose kernel
  trans_kern<<<grid_size, block_size>>>(a.gpu_data, b.gpu_data, a.r,
                                        a.cstride, b.cstride);
  errcheck_gpu();
}

//** Pointwise miscellaneous

// a = b^2
extern "C" void matrix_sqr(matrix a, matrix b)
{
  // sync matrices to gpu
  *(a.sync) = matrix_sync_gpu; // trashing, just set
  matrix_sync_to_gpu(b);

  #ifndef NO_ERRCHECK
    if ( (a.r != b.r) || (a.c != b.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Source and destination dimensions don't match for square!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  const unsigned row = b.r;
  const unsigned col = b.cstride;

  // if matrix dimensions are smaller than limits
  if ((row < add_rlim) || (col < add_clim))
  {
    // set threads per block as parameterized
    const dim3 block_size(add_small_tpb, 1, 1);

    // treat like vector, num blocks is n / tpb
    const dim3 grid_size((add_small_tpb + row*col - 1) /
                          add_small_tpb, 1, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    sqr_small_kern<<<grid_size, block_size>>>(a.gpu_data, b.gpu_data,
                                              row*col);
    errcheck_gpu();
  }
  else
  {
    // set threads per block as parameterized
    const dim3 block_size(add_big_tpb, 1, 1);

    // across rows we have ncol / tpb blocks
    const unsigned grid_size_x = (add_big_tpb + col - 1) / add_big_tpb;

    // down cols we have nrow / stripe blocks
    const unsigned grid_size_y = (add_big_stripe + row - 1) / add_big_stripe;
    const dim3 grid_size(grid_size_x, grid_size_y, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    sqr_big_kern<<<grid_size, block_size>>>(a.gpu_data, b.gpu_data,
                                            row*col, col);
    errcheck_gpu();
  }
}

// scalar multiplication a = b*c
extern "C" void matrix_scl(matrix a, float b, matrix c)
{
  *(a.sync) = matrix_sync_gpu; // trashing, just set
  matrix_sync_to_gpu(c);

  #ifndef NO_ERRCHECK
    if ( (a.r != c.r) || (a.c != c.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Source and destination dimensions don't match for scalar mult!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  const unsigned row = c.r;
  const unsigned col = c.cstride;

  // if matrix dimensions are smaller than limits
  if ((row < add_rlim) || (col < add_clim))
  {
    // set threads per block as parameterized
    const dim3 block_size(add_small_tpb, 1, 1);

    // treat like vector, num blocks is n / tpb
    const dim3 grid_size((add_small_tpb + row*col - 1) /
                          add_small_tpb, 1, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    scl_small_kern<<<grid_size, block_size>>>(a.gpu_data, b, c.gpu_data,
                                              row*col);
    errcheck_gpu();
  }
  else
  {
    // set threads per block as parameterized
    const dim3 block_size(add_big_tpb, 1, 1);

    // across rows we have ncol / tpb blocks
    const unsigned grid_size_x = (add_big_tpb + col - 1) / add_big_tpb;

    // down cols we have nrow / stripe blocks
    const unsigned grid_size_y = (add_big_stripe + row - 1) / add_big_stripe;
    const dim3 grid_size(grid_size_x, grid_size_y, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    scl_big_kern<<<grid_size, block_size>>>(a.gpu_data, b, c.gpu_data,
                                            row*col, col);
    errcheck_gpu();
  }
}

// scalar multiplication & pointwise addition a = b.*c+d
void matrix_scl_add(matrix a, float b, matrix c, matrix d)
{
  *(a.sync) = matrix_sync_gpu; // trashing, just set
  matrix_sync_to_gpu(c);
  matrix_sync_to_gpu(d);

  #ifndef NO_ERRCHECK
    if ( (a.r != c.r) || (a.c != c.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Source and destination dimensions don't match for scalar mult!\n", __LINE__);
      exit(CPU_ERROR);
    }
    if ( (a.r != d.r) || (a.c != d.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Source and destination dimensions don't match for scalar mult!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  const unsigned row = c.r;
  const unsigned col = c.cstride;

  // if matrix dimensions are smaller than limits
  if ((row < add_rlim) || (col < add_clim))
  {
    // set threads per block as parameterized
    const dim3 block_size(add_small_tpb, 1, 1);

    // treat like vector, num blocks is n / tpb
    const dim3 grid_size((add_small_tpb + row*col - 1) /
                          add_small_tpb, 1, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    scl_add_small_kern<<<grid_size, block_size>>>(a.gpu_data, b,
                                                  c.gpu_data, d.gpu_data,
                                                  row*col);
    errcheck_gpu();
  }
  else
  {
    // set threads per block as parameterized
    const dim3 block_size(add_big_tpb, 1, 1);

    // across rows we have ncol / tpb blocks
    const unsigned grid_size_x = (add_big_tpb + col - 1) / add_big_tpb;

    // down cols we have nrow / stripe blocks
    const unsigned grid_size_y = (add_big_stripe + row - 1) / add_big_stripe;
    const dim3 grid_size(grid_size_x, grid_size_y, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    scl_add_big_kern<<<grid_size, block_size>>>(a.gpu_data, b,
                                                c.gpu_data, d.gpu_data,
                                                row*col, col);
    errcheck_gpu();
  }
}

// pointwise multiplication a = b.*c
extern "C" void matrix_pmult(matrix a, matrix b, matrix c)
{
  *(a.sync) = matrix_sync_gpu; // trashing, just set
  matrix_sync_to_gpu(b);
  matrix_sync_to_gpu(c);

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

  const unsigned row = b.r;
  const unsigned col = b.cstride;

  // if matrix dimensions are smaller than limits
  if ((row < add_rlim) || (col < add_clim))
  {
    // set threads per block as parameterized
    const dim3 block_size(add_small_tpb, 1, 1);

    // treat like vector, num blocks is n / tpb
    const dim3 grid_size((add_small_tpb + row*col - 1) /
                          add_small_tpb, 1, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    pmult_small_kern<<<grid_size, block_size>>>(a.gpu_data, b.gpu_data,
                                                c.gpu_data, row*col);
    errcheck_gpu();
  }
  else
  {
    // set threads per block as parameterized
    const dim3 block_size(add_big_tpb, 1, 1);

    // across rows we have ncol / tpb blocks
    const unsigned grid_size_x = (add_big_tpb + col - 1) / add_big_tpb;

    // down cols we have nrow / stripe blocks
    const unsigned grid_size_y = (add_big_stripe + row - 1) / add_big_stripe;
    const dim3 grid_size(grid_size_x, grid_size_y, 1);

    if (DEBUG > 0)
    {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    pmult_big_kern<<<grid_size, block_size>>>(a.gpu_data, b.gpu_data,
                                              c.gpu_data, row*col, col);
    errcheck_gpu();
  }
}

// 
extern "C" void matrix_phi_prime(matrix a, matrix b)
{
  *(a.sync) = matrix_sync_gpu; // trashing, just set
  matrix_sync_to_gpu(b);

  #ifndef NO_ERRCHECK
    if ( (a.r != b.r) || (a.c != b.c) ) {
      fprintf(stderr, __FILE__ " %d: "
              "Source and destination dimensions don't match for phi_prime!\n", __LINE__);
      exit(CPU_ERROR);
    }
  #endif

  const unsigned row = b.r;
  const unsigned col = b.cstride;

  // if matrix dimensions are smaller than limits
  if ((row < add_rlim) || (col < add_clim))
  {
    // set threads per block as parameterized
    const dim3 block_size(add_small_tpb, 1, 1);

    // treat like vector, num blocks is n / tpb
    const dim3 grid_size((add_small_tpb + row*col - 1) /
                          add_small_tpb, 1, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    phi_prime_small_kern<<<grid_size, block_size>>>(a.gpu_data, b.gpu_data,
                                                    row*col);
    errcheck_gpu();
  }
  else
  {
    // set threads per block as parameterized
    const dim3 block_size(add_big_tpb, 1, 1);

    // across rows we have ncol / tpb blocks
    const unsigned grid_size_x = (add_big_tpb + col - 1) / add_big_tpb;

    // down cols we have nrow / stripe blocks
    const unsigned grid_size_y = (add_big_stripe + row - 1) / add_big_stripe;
    const dim3 grid_size(grid_size_x, grid_size_y, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    phi_prime_big_kern<<<grid_size, block_size>>>(a.gpu_data, b.gpu_data,
                                                  row*col, col);
    errcheck_gpu();
  }
}

//
extern "C" void matrix_delta(matrix delta, matrix y, matrix g)
{
  *(delta.sync) = matrix_sync_gpu; // trashing, just set
  matrix_sync_to_gpu(y);
  matrix_sync_to_gpu(g);

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

  const unsigned row = y.r;
  const unsigned col = y.cstride;

  // if matrix dimensions are smaller than limits
  if ((row < add_rlim) || (col < add_clim))
  {
    // set threads per block as parameterized
    const dim3 block_size(add_small_tpb, 1, 1);

    // treat like vector, num blocks is n / tpb
    const dim3 grid_size((add_small_tpb + row*col - 1) /
                          add_small_tpb, 1, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    delta_small_kern<<<grid_size, block_size>>>(delta.gpu_data, y.gpu_data, g.gpu_data,
                                                2.0f / (float)(g.r*g.c), row*col);
    errcheck_gpu();
  }
  else
  {
    // set threads per block as parameterized
    const dim3 block_size(add_big_tpb, 1, 1);

    // across rows we have ncol / tpb blocks
    const unsigned grid_size_x = (add_big_tpb + col - 1) / add_big_tpb;

    // down cols we have nrow / stripe blocks
    const unsigned grid_size_y = (add_big_stripe + row - 1) / add_big_stripe;
    const dim3 grid_size(grid_size_x, grid_size_y, 1);

    if (DEBUG > 0) {
      printf("block size: %d %d\n", block_size.x, block_size.y);
      printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
    }

    delta_big_kern<<<grid_size, block_size>>>(delta.gpu_data, y.gpu_data, g.gpu_data,
                                              2.0f / (float)(g.r*g.c), row*col, col);
    errcheck_gpu();
  }
}

//** Combination/Separation

// remove last row of m
extern "C" void matrix_r0(matrix *m)
{
  matrix_sync_to_gpu(*m);

  const dim3 block_size(r0_tpb, 1, 1);
  const dim3 grid_size((r0_tpb + m->c - 1) / r0_tpb, 1, 1);
  
  if (DEBUG > 0) {
    printf("block size: %d %d\n", block_size.x, block_size.y);
    printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
  }

  // decriment number of rows
  --(m->r);

  // pass r0_kern gpu data beginning at last row
  r0_kern<<<grid_size, block_size>>>((m->gpu_data)+((m->r)*(m->cstride)), m->c);
  errcheck_gpu();
}

// append a row of 1's to m
// Note:  we currently always leave enough padding so that we
//        can add one extra row.  Can't do this more than once!
extern "C" void matrix_r1(matrix *m)
{
  matrix_sync_to_gpu(*m);

  const dim3 block_size(r0_tpb, 1, 1);
  const dim3 grid_size((r0_tpb + m->c - 1) / r0_tpb, 1, 1);
  
  if (DEBUG > 0) {
    printf("block size: %d %d\n", block_size.x, block_size.y);
    printf("grid size:  %d %d\n", grid_size.x, grid_size.y);
  }

  // pass r1_kern gpu data beginning at beginning of row after end
  r1_kern<<<grid_size, block_size>>>((m->gpu_data)+((m->r)*(m->cstride)), m->c);
  errcheck_gpu();

  // increment number of rows
  ++(m->r);
}

// remove and save last col of m
extern "C" void matrix_c0v(matrix *m)
{
  matrix_sync_to_gpu(*m);

  const dim3 block_size(r0_tpb, 1, 1);

  const dim3 grid_size((r0_tpb + m->r - 1) / r0_tpb, 1, 1);

  --(m->c);

  c0v_kern<<<grid_size, block_size>>>((m->gpu_data) + (m->c), m->cv, m->r, m->cstride);
  errcheck_gpu();
}

// restore last col of m
extern "C" void matrix_cv(matrix *m)
{
  matrix_sync_to_gpu(*m);

  const dim3 block_size(r0_tpb, 1, 1);

  const dim3 grid_size((r0_tpb + m->r - 1) / r0_tpb, 1, 1);

  cv_kern<<<grid_size, block_size>>>((m->gpu_data) + (m->c), m->cv, m->r, m->cstride);
  errcheck_gpu();

  ++(m->c);
}

//** Error Measurement

// rmse between values of actual and approx
extern "C" float matrix_rmse(matrix actual, matrix approx)
{
  // if dims don't match throw error!

  // 
  matrix_sync_to_cpu(actual);
  matrix_sync_to_cpu(approx);

  //
  unsigned r, c;
  const unsigned len = actual.r*actual.c;
  float *d1 = actual.cpu_data;
  float *d2 = approx.cpu_data;
  float err = 0.0f;

//  #pragma omp parallel for shared(err) private(c,d1,d2)
  for (r = 0; r < actual.r; ++r)
    for (c = 0; c < actual.c; ++c)
    {
      unsigned i = r*actual.cstride+c;
      err += (d1[i] - d2[i])*(d1[i] - d2[i]);
    }

  return sqrt(err/len);
}

// return maximum relative error between approx and actual
extern "C" float matrix_relerr_max(matrix actual, matrix approx)
{
  unsigned i;

  // if matrices are different sizes
  // then return -1.0f
  if ( (approx.r != actual.r) ||
       (approx.c != actual.c) )
    return -1.0f;

  // synchronize both matrices to cpu
  matrix_sync_to_cpu(approx);
  matrix_sync_to_cpu(actual);

  // figure relative error
  float max_err = 0.0f;
  for (i = 0; i < (approx.rstride)*(approx.cstride); ++i)
  {
    // needs work here !!!
    float actual_cur = actual.cpu_data[i];
    if (fabs(actual_cur) > 1.0e-10f)
    {
      float rel_err = fabs(actual_cur - approx.cpu_data[i]) /
                           actual_cur;

      if (rel_err > max_err)
        max_err = rel_err;
    }
  }

  return max_err;
}

//** Output

// print m to standard out
extern "C" void matrix_print(matrix m)
{
  unsigned r, c;
  const unsigned stride = m.cstride;
  float *data = m.cpu_data;

  // sync matrix to cpu
  matrix_sync_to_cpu(m);

  // print each value to stdout
  for (r = 0; r < m.r; ++r)
  {
    for (c = 0; c < m.c; ++c)
      printf("%f ", data_at(r,c));
    printf("\n");
  }
}

// print m including padding to standard out
extern "C" void matrix_print_padded(matrix m)
{
  unsigned r, c;

  // set data and stride for data_at
  const unsigned stride = m.cstride;
  float *data = m.cpu_data;

  // sync matrix to cpu
  matrix_sync_to_cpu(m);

  // print each value to stdout
  for (r = 0; r < m.rstride; ++r)
  {
    for (c = 0; c < m.cstride; ++c)
      printf("%f ", data_at(r,c));
    printf("\n");
  }
}
