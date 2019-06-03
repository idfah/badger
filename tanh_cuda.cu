/*
 *  Libraries
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

#include "benchmark.h"

/*
 *  Macros
 */

//#define DEBUG

#ifndef mult_tm
  #define mult_tm 20
#endif
#ifndef mult_tn
  #define mult_tn 16
#endif
#ifndef mult_tp
  #define mult_tp 64
#endif

/*
 *  CUDA kernel bodies
 */

__global__ void full_kern(float *a, unsigned a_r, unsigned a_c,
                          float *b, unsigned b_r, unsigned b_c,
                          float *c, unsigned c_r, unsigned c_c)
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
  const float *bend = bcur + b_c;

  // current a values
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

  // aopy results to global memory
  #pragma unroll
  for (i = 0; i < mult_tm; ++i, acur += a_c)
  {
    acur[0] = tanhf(aval_v1[i]);
    acur[mult_tp] = tanhf(aval_v2[i]);
  }
}

/*
 *  Function bodies
 */

#define tn 64
void figure_gold(float *gold, float *m1, float *m2, unsigned n)
{
  unsigned ti, tj, tk;
  unsigned i, j, k;

  #pragma omp parallel for private(tj,tk,i,j,k)
  for (ti = 0; ti < n; ti += tn)
    for (tj = 0; tj < n; tj += tn)
      for (tk = 0; tk < n; tk += tn)
        for (i = ti; i < ti+tn; ++i)
          for (j = tj; j < tj+tn; ++j)
          {
            float dot = gold[i*n+j];
            for (k = tk; k < tk+tn; ++k)
              dot += m1[i*n+k] * m2[k*n+j];

            gold[i*n+j] = dot;
          }

  #pragma omp parallel
  for (i = 0; i < n*n; ++i)
    gold[i] = tanhf(gold[i]);
}


int main(int narg, char **arg)
{
  // counter & size of vector
  unsigned i, n = 6400;

  // timers
  benchmark bm;
  benchmark_init(&bm);

  // grab n from command line if given
  if (narg > 1)
    n = atoi(arg[1]);

  const size_t size = n*n*sizeof(float);

  #ifdef DEBUG
    printf("n: %d\n", n);
  #endif

  // allocate space for first matrix
  float *m1 = (float*)malloc(size);
  if (m1 == NULL) {
    fprintf(stderr, "Could not allocate space for m1!\n");
    exit(1);
  }

  // allocate space for second matrix
  float *m2 = (float*)malloc(size);
  if (m2 == NULL) {
    fprintf(stderr, "Could not allocate space for m2!\n");
    exit(1);
  }

  // allocate space for result matrix
  float *result = (float*)malloc(size);
  if (result == NULL) {
    fprintf(stderr, "Could not allocate space for result!\n");
    exit(1);
  }

  // allocate space for gold matrix
  float *gold = (float*)malloc(size);
  if (gold == NULL) {
    fprintf(stderr, "Could not allocate space for gold!\n");
    exit(1);
  }

  // seed random number generator
  srand(7);

  // fill m1 and m2 with random numbers from [-7,7]
  for (i = 0; i < n*n; ++i)
  {
    m1[i] = ((((float)rand()) / RAND_MAX) * 14.0f) - 7.0f;
    m2[i] = ((((float)rand()) / RAND_MAX) * 14.0f) - 7.0f;

    #ifdef DEBUG
      printf("m1[%d]:  %f\n", i, m1[i]);
    #endif
  }

  // figure gold values
  printf("start\n");
  figure_gold(gold, m1, m2, n);
  printf("end\n");

  // allocate space for matrices on gpu
  float *d_m1;
  cudaMalloc((void**)&d_m1,  size);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Could not allocate d_m1!\n");
    exit(1);
  }

  float *d_m2;
  cudaMalloc((void**)&d_m2, size);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Could not allocate d_m2!\n");
    exit(1);
  }

  float *d_result;
  cudaMalloc((void**)&d_result, size);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Could not allocate d_result!\n");
    exit(1);
  }

  // copy m1 & m2 to device
  cudaMemcpy(d_m1, m1, size, cudaMemcpyHostToDevice);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Could not copy m1 to device!\n");
    exit(1);
  }

  cudaMemcpy(d_m2, m2, size, cudaMemcpyHostToDevice);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Could not copy m2 to device!\n");
    exit(1);
  }

  // set grid and block dims
  const dim3 block_size(mult_tn, mult_tp/mult_tn, 1);
  const unsigned num_cblk_r = n / mult_tm;
  const unsigned num_cblk_c = n / (2*mult_tp);
  const dim3 grid_size(num_cblk_c, num_cblk_r, 1);

  // warm up
  full_kern<<<grid_size, block_size>>>(d_result, n, n,
                                       d_m1, n, n,
                                       d_m2, n, n);
  cudaDeviceSynchronize();

  // gold run
  benchmark_start_timer(&bm);
  full_kern<<<grid_size, block_size>>>(d_result, n, n,
                                       d_m1, n, n,
                                       d_m2, n, n);
  cudaDeviceSynchronize();
  benchmark_stop_timer(&bm);

  cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

  float max_err = 0.0f;
  for (i = i; i < n*n; ++i)
  {
    float actual_cur = gold[i];
    if (fabs(actual_cur) > 1.0e-10f)
    {
      float rel_err = fabs(actual_cur - result[i] / 
                           actual_cur);

      if (rel_err > max_err)
        max_err = rel_err;
    }
  }

  // set number of floating point ops
  benchmark_add_flop(&bm, n*n*(2ll*n));

  printf("Full tanh\n=======\n");
  printf("Time: %f  GFlopS: %f  Error: %f\n\n",
         benchmark_check_timer(bm),
         benchmark_check_gflops(bm),
         max_err);

  free(m1);
  free(m2);
  free(result);
  free(gold);

  cudaFree(d_m1);
  cudaFree(d_m2);
  cudaFree(d_result);

  return 0;
}
