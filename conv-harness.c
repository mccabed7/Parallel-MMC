/* Test and timing harness program for developing a multichannel
   multikernel convolution (as used in deep learning networks)

   Note there are some simplifications around this implementation,
   in particular with respect to computing the convolution at edge
   pixels of the image.

   Author: David Gregg
   Date:   March 2022

   Version 1.7 : Adjusted types for mixed-type computation

   Version 1.6 : Modified the code so that the input tensor is float

   Version 1.5 : Modified the code so that the input and kernel
                 are tensors of 16-bit integer values

   Version 1.4 : Modified the random generator to reduce the range
                 of generated values;

   Version 1.3 : Fixed which loop variables were being incremented
                 in write_out();
                 Fixed dimensions of output and control_output 
                 matrices in main function

   Version 1.2 : Changed distribution of test data to (hopefully) 
                 eliminate random walk of floating point error;
                 Also introduced checks to restrict kernel-order to
                 a small set of values

   Version 1.1 : Fixed bug in code to create 4d matrix
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h>
// #include <emmintrin.h>
// #include <pmmintrin.h>
// #include <xmmintrin.h>

/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
/*#define DEBUGGING(_x) _x */
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)


/* write 3d matrix to stdout */
void write_out(int16_t *** a, int dim0, int dim1, int dim2)
{
  int i, j, k;

  for ( i = 0; i < dim0; i++ ) {
    printf("Outer dimension number %d\n", i);
    for ( j = 0; j < dim1; j++ ) {
      for ( k = 0; k < dim2 - 1; k++ ) {
        printf("%d, ", a[i][j][k]);
      }
      // print end of line
      printf("%f\n", a[i][j][dim2-1]);
    }
  }
}


/* create new empty 4d float matrix */
float **** new_empty_4d_matrix_float(int dim0, int dim1, int dim2, int dim3)
{
  float **** result = malloc(dim0 * sizeof(float***));
  float *** mat1 = malloc(dim0 * dim1 * sizeof(float**));
  float ** mat2 = malloc(dim0 * dim1 * dim2 * sizeof(float*));
  float * mat3 = malloc(dim0 * dim1 * dim2 *dim3 * sizeof(float));
  int i, j, k;

  
  for ( i = 0; i < dim0; i++ ) {
    result[i] = &(mat1[i*dim1]);
    for ( j = 0; j < dim1; j++ ) {
      result[i][j] = &(mat2[i*dim1*dim2 + j*dim2]);
      for ( k = 0; k < dim2; k++ ) {
        result[i][j][k] = &(mat3[i*dim1*dim2*dim3+j*dim2*dim3+k*dim3]);
      }
    }
  }

  return result;
}

/* create new empty 3d matrix */
float *** new_empty_3d_matrix_float(int dim0, int dim1, int dim2)
{
  float **** mat4d;
  float *** mat3d;

  // create a 4d matrix with single first dimension
  mat4d = new_empty_4d_matrix_float(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* create new empty 4d int16_t matrix */
int16_t **** new_empty_4d_matrix_int16(int dim0, int dim1, int dim2, int dim3)
{
  int16_t **** result = malloc(dim0 * sizeof(int16_t***));
  int16_t *** mat1 = malloc(dim0 * dim1 * sizeof(int16_t**));
  int16_t ** mat2 = malloc(dim0 * dim1 * dim2 * sizeof(int16_t*));
  int16_t * mat3 = malloc(dim0 * dim1 * dim2 *dim3 * sizeof(int16_t));
  int i, j, k;

  
  for ( i = 0; i < dim0; i++ ) {
    result[i] = &(mat1[i*dim1]);
    for ( j = 0; j < dim1; j++ ) {
      result[i][j] = &(mat2[i*dim1*dim2 + j*dim2]);
      for ( k = 0; k < dim2; k++ ) {
        result[i][j][k] = &(mat3[i*dim1*dim2*dim3+j*dim2*dim3+k*dim3]);
      }
    }
  }

  return result;
}

/* create new empty 3d matrix */
int16_t *** new_empty_3d_matrix_int16(int dim0, int dim1, int dim2)
{
  int16_t **** mat4d;
  int16_t *** mat3d;

  // create a 4d matrix with single first dimension
  mat4d = new_empty_4d_matrix_int16(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* take a copy of the matrix and return in a newly allocated matrix */
int16_t **** copy_4d_matrix(int16_t **** source_matrix, int dim0,
                            int dim1, int dim2, int dim3)
{
  int i, j, k, l;
  int16_t **** result = new_empty_4d_matrix_int16(dim0, dim1, dim2, dim3);

  for ( i = 0; i < dim0; i++ ) {
    for ( j = 0; j < dim1; j++ ) {
      for ( k = 0; k < dim2; k++ ) {
        for ( l = 0; l < dim3; l++ ) {
          result[i][j][k][l] = source_matrix[i][j][k][l];
        }
      }
    }
  }
  return result;
}

/* create a matrix and fill it with random numbers */
int16_t **** gen_random_4d_matrix_int16(int dim0, int dim1, int dim2, int dim3)
{
int16_t **** result;
int i, j, k, l;
struct timeval seedtime;
  int seed;

  result = new_empty_4d_matrix_int16(dim0, dim1, dim2, dim3);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  const int range = 1 << 10; // 2^10
  //const int bias = 1 << 16; // 2^16
  int16_t offset = 0.0;
  for ( i = 0; i < dim0; i++ ) {
    for ( j = 0; j < dim1; j++ ) {
      for ( k = 0; k < dim2; k++ ) {
        for ( l = 0; l < dim3; l++ ) {
          // generate uniform random integer with mean of zero
          long long rand = random();
          // now cut down the range and bias the mean to reduce
          // the likelihood of large floating point round-off errors
          int reduced_range = (rand % range);
          result[i][j][k][l] = reduced_range;
        }
      }
    }
  }

  return result;
}


/* create a matrix and fill it with random numbers */
float **** gen_random_4d_matrix_float(int dim0, int dim1, int dim2, int dim3)
{
float **** result;
int i, j, k, l;
struct timeval seedtime;
  int seed;

  result = new_empty_4d_matrix_float(dim0, dim1, dim2, dim3);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  const int range = 1 << 12; // 2^12
  const int bias = 1 << 10; // 2^16
  int16_t offset = 0.0;
  for ( i = 0; i < dim0; i++ ) {
    for ( j = 0; j < dim1; j++ ) {
      for ( k = 0; k < dim2; k++ ) {
        for ( l = 0; l < dim3; l++ ) {
          // generate uniform random integer with mean of zero
          long long rand = random();
          // now cut down the range and bias the mean to reduce
          // the likelihood of large floating point round-off errors
          int reduced_range = (rand % range);
          result[i][j][k][l] = reduced_range + bias;
        }
      }
    }
  }

  return result;
}


/* create a matrix and fill it with random numbers */
float *** gen_random_3d_matrix_float(int dim0, int dim1, int dim2)
{
  float **** mat4d;
  float *** mat3d;

  // create a 4d matrix with single first dimension
  mat4d = gen_random_4d_matrix_float(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* create a matrix and fill it with random numbers */
int16_t *** gen_random_3d_matrix_int16(int dim0, int dim1, int dim2)
{
  int16_t **** mat4d;
  int16_t *** mat3d;

  // create a 4d matrix with single first dimension
  mat4d = gen_random_4d_matrix_int16(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* check the sum of absolute differences is within reasonable epsilon */
void check_result(float *** result, float *** control,
                  int dim0, int dim1, int dim2)
{
  int i, j, k;
  double sum_abs_diff = 0.0;
  const double EPSILON = 0.0625;

  //printf("SAD\n");
  
  for ( i = 0; i < dim0; i++ ) {
    for ( j = 0; j < dim1; j++ ) {
      for ( k = 0; k < dim2; k++ ) {
        double diff = fabs(control[i][j][k] - result[i][j][k]);
        assert( diff >= 0.0 );
        sum_abs_diff = sum_abs_diff + diff;
      }
    }
  }

  if ( sum_abs_diff > EPSILON ) {
    fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
            sum_abs_diff, EPSILON);
  }
  else {
    printf("COMMENT: sum of absolute differences (%f)  within acceptable range (%f)\n", sum_abs_diff, EPSILON);
  }
}

/* the slow but correct version of matmul written by David */
void multichannel_conv(float *** image, int16_t **** kernels,
		       float *** output, int width, int height,
		       int nchannels, int nkernels, int kernel_order)
{
  int h, w, x, y, c, m;

  for ( m = 0; m < nkernels; m++ ) {
    for ( w = 0; w < width; w++ ) {
      for ( h = 0; h < height; h++ ) {
        double sum = 0.0;
        for ( c = 0; c < nchannels; c++ ) {
          for ( x = 0; x < kernel_order; x++) {
            for ( y = 0; y < kernel_order; y++ ) {
              sum += image[w+x][h+y][c] * kernels[m][c][x][y];
            }
          }
          output[m][w][h] = (float) sum;
        }
      }
    }
  }
}

static inline __m256d mul_4(__m256d k4, __m256d sum4, __m128 i4) {
  return _mm256_add_pd(sum4, _mm256_mul_pd(k4, _mm256_cvtps_pd(i4)));
}

static inline __m256d mul_8(__m256d k4_1, __m256d k4_2, __m256d sum4, float *i, int c) {
  sum4 = mul_4(k4_1, sum4, _mm_loadu_ps(&(i[c])));
  return mul_4(k4_2, sum4, _mm_loadu_ps(&(i[c+4])));
}

// static inline __m128 mm512_combine_4(__m256d a, __m256d b, __m256d c, __m256d d) {
//   __m256d low  = _mm256_castpd512_pd256(a);
//   __m256d high;
//   low = _mm256_extractf64x4_pd(a, 0);
//   high = _mm256_extractf64x4_pd(a, 1);
//   // __m512d l = _mm512_shuffle_pd(a, b, 0xdd); // 7, 5, 3, 1,     7, 5, 3, 1
//   // __m512d r = _mm512_shuffle_pd(a, b, 0x88); // 6, 4, 2, 0,     6, 4, 2, 0
//   return _mm256_cvtpd_ps(_mm256_setzero_pd());
// }

static inline void mul_4h_8c_sum(int w, int height, int kernel_order, int nchannels, float *** restrict image, float * restrict outputmw, int16_t *** restrict kernelsm) {

  __m256d sum8_1, sum8_2, sum8_3, sum8_4, k4_1, k4_2;
  __m256 i8;
  __m128i k8i;
  float *i_1, *i_2, *i_3, *i_4, **imagewx;
  int16_t *k;
  int h, x, y, c, yh, wx;
  // #pragma omp target teams distribute parallel for schedule(static)
  for ( h = 0; h < height-3; h+=4 ) {
    // sum = 0.0;
    // for ( c = 0; c < nchannels; c++ ) {
    // 	sum += (float) (image[w][h][c] * kernels[m][0][0][c]);
    // }
    // output[m][w][h] = sum;

    sum8_1 = _mm256_setzero_pd();
    sum8_2 = _mm256_setzero_pd();
    sum8_3 = _mm256_setzero_pd();
    sum8_4 = _mm256_setzero_pd();


    for ( x = 0; x < kernel_order; x++) {
      imagewx = image[w+x];
      for ( y = 0; y < kernel_order; y++ ) {
        yh = y+h;
        i_1 = imagewx[yh];
        i_2 = imagewx[yh+1];
        i_3 = imagewx[yh+2];
        i_4 = imagewx[yh+3];
        k = kernelsm[x][y];

        for ( c = 0; c < nchannels; c += 8) {
          // since kernel is 16 bit ints, one vector has 8 values
          k8i = _mm_load_si128((__m128i_u*)&(k[c]));

          k4_1 = _mm256_cvtepi32_pd(_mm_unpacklo_epi16(k8i, k8i));
          k4_2 = _mm256_cvtepi32_pd(_mm_unpackhi_epi16(k8i, k8i));

          sum8_1 = mul_8(k4_1, k4_2, sum8_1, i_1, c);
          sum8_2 = mul_8(k4_1, k4_2, sum8_2, i_2, c);
          sum8_3 = mul_8(k4_1, k4_2, sum8_3, i_3, c);
          sum8_4 = mul_8(k4_1, k4_2, sum8_4, i_4, c);
          // i8 = _mm256_loadu_ps(&(i_1[c]));
          // sum8_1 = mul_4(k4_1, sum8_1, _mm256_extractf128_ps(i8, 0));
          // sum8_1 = mul_4(k4_2, sum8_1, _mm256_extractf128_ps(i8, 1)); // non compounding sum
          // i8 = _mm256_loadu_ps(&(i_2[c]));
          // sum8_2 = mul_4(k4_1, sum8_2, _mm256_extractf128_ps(i8, 0));
          // sum8_2 = mul_4(k4_2, sum8_2, _mm256_extractf128_ps(i8, 1));
          // i8 = _mm256_loadu_ps(&(i_3[c]));
          // sum8_3 = mul_4(k4_1, sum8_3, _mm256_extractf128_ps(i8, 0));
          // sum8_3 = mul_4(k4_2, sum8_3, _mm256_extractf128_ps(i8, 1));
          // i8 = _mm256_loadu_ps(&(i_4[c]));
          // sum8_4 = mul_4(k4_1, sum8_4, _mm256_extractf128_ps(i8, 0));
          // sum8_4 = mul_4(k4_2, sum8_4, _mm256_extractf128_ps(i8, 1));
        }
      }
    }
    
    sum8_1 = _mm256_hadd_pd(sum8_1, sum8_2);
    sum8_3 = _mm256_hadd_pd(sum8_3, sum8_4);
    sum8_4 = _mm256_hadd_pd(sum8_1, sum8_3);
    // sum8_1 = mm512_hadd(sum8_1, sum8_2);
    // sum8_3 = mm512_hadd(sum8_3, sum8_4);
    // sum8_4 = mm512_hadd(sum8_1, sum8_3);
    // sum8_1 = mm512_hadd(sum8_4, _mm512_setzero_pd());
    
    // _mm256_store_ps(&output[m][w][h], _mm256_cvtpd_ps(_mm256_castpd512_pd256(sum8_1)));

    _mm_storeu_ps(&(outputmw[h]), _mm256_cvtpd_ps(sum8_4));
    // outputmw[h] = _mm256_reduce_add_pd(sum8_1);
    // outputmw[h+1] = _mm256_reduce_add_pd(sum8_2);
    // outputmw[h+2] = _mm256_reduce_add_pd(sum8_3);
    // outputmw[h+3] = _mm256_reduce_add_pd(sum8_4);
  }
  for (;h<height; h++) {
    sum8_1 = _mm256_setzero_pd();
    for ( x = 0; x < kernel_order; x++) {
      for ( y = 0; y < kernel_order; y++ ) {
        i_1 = image[x+w][h+y];
        k = kernelsm[x][y];

        for ( c = 0; c < nchannels; c += 8) {
          k8i = _mm_load_si128((__m128i_u*)&(k[c]));
          
          k4_1 = _mm256_cvtepi32_pd(_mm_unpacklo_epi16(k8i, k8i));
          k4_2 = _mm256_cvtepi32_pd(_mm_unpackhi_epi16(k8i, k8i));
          sum8_1 = mul_8(k4_1, k4_2, sum8_1, i_1, c);

          // i8 = _mm256_loadu_ps(&(i_1[c]));
          // sum8_1 = mul_4(k4_1, sum8_1, _mm256_extractf128_ps(i8, 0));
          // sum8_1 = mul_4(k4_2, sum8_1, _mm256_extractf128_ps(i8, 1)); // non compounding sum // non compounding sum
        }
      }
    }
    sum8_1 = _mm256_hadd_pd(sum8_1, sum8_1);
    sum8_1 = _mm256_hadd_pd(sum8_1, sum8_1);
    _mm_store1_ps(&(outputmw[h]), _mm256_cvtpd_ps(sum8_1));
  }
}

static inline void matrix_order_1_conv(float *** restrict image, int16_t **** restrict kernels, float *** restrict output,
				int width, int height, int nchannels, int nkernels)
{
  	int h, w, c, m;
	int16_t *k;
	float *i_1, *i_2, *i_3, *i_4;
	double sum;
	__m256d k8, sum8_1, sum8_2, sum8_3, sum8_4, k4_1, k4_2;
  __m256 i8;
	__m128i k8i;

  #pragma omp parallel for collapse(2) schedule(static)
  // #pragma omp target teams distribute parallel for schedule(static) // bad
	for ( m = 0; m < nkernels; m++ ) {
		for ( w = 0; w < width; w++ ) {
			for ( h = 0; h < height-3; h+=4 ) {
				// sum = 0.0;
				// for ( c = 0; c < nchannels; c++ ) {
				// 	sum += (float) (image[w][h][c] * kernels[m][0][0][c]);
				// }
				// output[m][w][h] = sum;

				sum8_1 = _mm256_setzero_pd();
				sum8_2 = _mm256_setzero_pd();
				sum8_3 = _mm256_setzero_pd();
				sum8_4 = _mm256_setzero_pd();

				i_1 = image[w][h];
				i_2 = image[w][h+1];
				i_3 = image[w][h+2];
				i_4 = image[w][h+3];
				k = kernels[m][0][0];

				for ( c = 0; c < nchannels; c += 8) {
					// since kernel is 16 bit ints, one vector has 8 values
					// k4i = _mm_load_si128((__m128i_u*)&(k[c]));
					// // k4_1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(k4i, k4i), 16));
					// // k4_2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(k4i, k4i), 16));
          
          // k8 = _mm256_cvtepi64_pd(_mm256_cvtepi16_epi64(k4i)); // could try from 256 to double
          // // k4_1 = _mm256_cvtepi32_pd(_mm_srai_epi32(_mm_unpacklo_epi16(k4i, k4i), 16));
          // // k4_2 = _mm256_cvtepi32_pd(_mm_srai_epi32(_mm_unpackhi_epi16(k4i, k4i), 16));

					// sum8_1 = mul_8(k8, sum8_1, i_1, c); // non compounding sum
					// sum8_2 = mul_8(k8, sum8_2, i_2, c);
					// sum8_3 = mul_8(k8, sum8_3, i_3, c);
					// sum8_4 = mul_8(k8, sum8_4, i_4, c);
          k8i = _mm_load_si128((__m128i_u*)&(k[c]));
          k4_1 = _mm256_cvtepi32_pd(_mm_unpackhi_epi16(k8i, k8i));
          k4_2 = _mm256_cvtepi32_pd(_mm_unpacklo_epi16(k8i, k8i));

          sum8_1 = mul_8(k4_1, k4_2, sum8_1, i_1, c);
          sum8_2 = mul_8(k4_1, k4_2, sum8_2, i_2, c);
          sum8_3 = mul_8(k4_1, k4_2, sum8_3, i_3, c);
          sum8_4 = mul_8(k4_1, k4_2, sum8_4, i_4, c);
          // k4_1 = _mm256_cvtepi32_pd(_mm_unpacklo_epi16(k8i, k8i));
          // k4_2 = _mm256_cvtepi32_pd(_mm_unpackhi_epi16(k8i, k8i));
          // i8 = _mm256_loadu_ps(&(i_1[c]));
          // sum8_1 = mul_4(k4_1, sum8_1, _mm256_extractf128_ps(i8, 0));
          // sum8_1 = mul_4(k4_2, sum8_1, _mm256_extractf128_ps(i8, 1)); // non compounding sum
          // i8 = _mm256_loadu_ps(&(i_2[c]));
          // sum8_2 = mul_4(k4_1, sum8_2, _mm256_extractf128_ps(i8, 0));
          // sum8_2 = mul_4(k4_2, sum8_2, _mm256_extractf128_ps(i8, 1));
          // i8 = _mm256_loadu_ps(&(i_3[c]));
          // sum8_3 = mul_4(k4_1, sum8_3, _mm256_extractf128_ps(i8, 0));
          // sum8_3 = mul_4(k4_2, sum8_3, _mm256_extractf128_ps(i8, 1));
          // i8 = _mm256_loadu_ps(&(i_4[c]));
          // sum8_4 = mul_4(k4_1, sum8_4, _mm256_extractf128_ps(i8, 0));
          // sum8_4 = mul_4(k4_2, sum8_4, _mm256_extractf128_ps(i8, 1));
				}
        
				sum8_1 = _mm256_hadd_pd(sum8_1, sum8_2);
        sum8_3 = _mm256_hadd_pd(sum8_3, sum8_4);
        sum8_4 = _mm256_hadd_pd(sum8_1, sum8_3);
        // sum8_1 = mm512_hadd(sum8_1, sum8_2);
        // sum8_3 = mm512_hadd(sum8_3, sum8_4);
        // sum8_4 = mm512_hadd(sum8_1, sum8_3);
        // sum8_1 = mm512_hadd(sum8_4, _mm512_setzero_pd());
        
        // _mm256_store_ps(&output[m][w][h], _mm256_cvtpd_ps(_mm256_castpd512_pd256(sum8_1)));

        _mm_storeu_ps(&(output[m][w][h]), _mm256_cvtpd_ps(sum8_4));
        
        // output[m][w][h] = _mm256_reduce_add_pd(sum8_1);
        // output[m][w][h+1] = _mm256_reduce_add_pd(sum8_2);
        // output[m][w][h+2] = _mm256_reduce_add_pd(sum8_3);
        // output[m][w][h+3] = _mm256_reduce_add_pd(sum8_4);
			}
      		
			for (;h<height; h++) {
        sum8_1 = _mm256_setzero_pd();

				i_1 = image[w][h];
				k = kernels[m][0][0];

				for ( c = 0; c < nchannels; c += 8) {
          k8i = _mm_load_si128((__m128i_u*)&(k[c]));
          
          k4_1 = _mm256_cvtepi32_pd(_mm_unpacklo_epi16(k8i, k8i));
          k4_2 = _mm256_cvtepi32_pd(_mm_unpackhi_epi16(k8i, k8i));
          sum8_1 = mul_8(k4_1, k4_2, sum8_1, i_1, c);
          // i8 = _mm256_loadu_ps(&(i_1[c]));
          // sum8_1 = mul_4(k4_1, sum8_1, _mm256_extractf128_ps(i8, 0));
          // sum8_1 = mul_4(k4_2, sum8_1, _mm256_extractf128_ps(i8, 1)); // non compounding sum // non compounding sum
          // k9i = _mm_load_si128((__m128i_u*)&(k[c]));
					// k4_1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(k4i, k4i), 16));
					// k4_2 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(k4i, k4i), 16));
          
          // k8 = _mm256_cvtepi64_pd(_mm256_cvtepi16_epi64(k4i)); // could try from 256 to double
          // k4_1 = _mm256_cvtepi32_pd(_mm_srai_epi32(_mm_unpacklo_epi16(k4i, k4i), 16));
          // k4_2 = _mm256_cvtepi32_pd(_mm_srai_epi32(_mm_unpackhi_epi16(k4i, k4i), 16));

					// sum8_1 = mul_8(k8, sum8_1, i_1, c); // non compounding sum
					// since kernel is 16 bit ints, one vector has 8 values
					// k4i = _mm_loadu_si128((__m128i_u*)&(k[c]));
					// k4_1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(k4i, _mm_setzero_si128()));
					// k4_2 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(k4i, _mm_setzero_si128()));

					// sum4_1 = mul_8_4_4(k4_1, k4_2, sum4_1, i4_1, i4_2, i_1, c);
				}

				// sum4_1 = _mm_hadd_ps(sum4_1, sum4_1);
				// sum4_1 = _mm_hadd_ps(sum4_1, sum4_1);

				// float sum1;
				// _mm_store_ss(&sum1, sum4_1);
        sum8_1 = _mm256_hadd_pd(sum8_1, sum8_1);
        sum8_1 = _mm256_hadd_pd(sum8_1, sum8_1);
        _mm_store1_ps(&(output[m][w][h]), _mm256_cvtpd_ps(sum8_1));
      }
 		}
	}
}
/* create new empty 4d float matrix */
int16_t **** reorganise_kernels(int16_t **** old_kernels, int nkernels, int nchannels, int kernel_order)
{
  // new_empty_4d_matrix_int16(nkernels, kernel_order, kernel_order, nchannels);
  
  int16_t **** result = malloc(nkernels * sizeof(int16_t***));
  int16_t *** mat1 = malloc(nkernels * kernel_order * sizeof(int16_t**));
  int16_t ** mat2 = malloc(nkernels * kernel_order * kernel_order * sizeof(int16_t*));
  int16_t * mat3 = _mm_malloc(nkernels * kernel_order * kernel_order *nchannels * sizeof(int16_t), 256);
  int i, j, k, l;
  #pragma omp parallel for
  // #pragma omp target teams distribute parallel for
  for ( i = 0; i < nkernels; i++ ) {
    result[i] = &(mat1[i*kernel_order]);
    for ( j = 0; j < kernel_order; j++ ) {
      result[i][j] = &(mat2[i*kernel_order*kernel_order + j*kernel_order]);
      for ( k = 0; k < kernel_order; k++ ) {
        result[i][j][k] = &(mat3[i*kernel_order*kernel_order*nchannels+j*kernel_order*nchannels+k*nchannels]);
        for ( l = 0; l < nchannels; l++ ) {
          result[i][j][k][l] = old_kernels[i][l][j][k];
        }
      }
    }
  }

  return result;
}

/* the fast version of matmul written by the student */
void student_conv(float *** restrict image, int16_t **** restrict kernels, float *** restrict output,
               int width, int height, int nchannels, int nkernels,
               int kernel_order)
{
  int16_t ****better_kernels = reorganise_kernels(kernels, nkernels, nchannels, kernel_order);
  	//float *** flipped_image = flip_3d_matrix_float(image, width+kernel_order, height+kernel_order, nchannels); 
	switch (kernel_order)
	{
		case 1:
			matrix_order_1_conv(image, better_kernels, output, width, height, nchannels, nkernels);
			return;
		default:
			break;
	}

  	int h, w, x, y, c, m;
	
  #pragma omp parallel for collapse(2) schedule(static)
  // #pragma omp target teams distribute parallel for collapse(2) schedule(static)
	for ( m = 0; m < nkernels; m++ ) {
    
		for ( w = 0; w < width; w++ ) {
      mul_4h_8c_sum(w, height, kernel_order, nchannels, image, output[m][w], better_kernels[m]);
			// for ( h = 0; h < height; h++ ) {
			// 	double sum_vec = 0.0;
			// 	double sum_vec_arr[4];
			// 	__m256d sum4 = _mm256_setzero_pd();

			// 	// #pragma omp collapse(2) // negligeable speedup
			// 	for ( x = 0; x < kernel_order; x++) {
			// 		for ( y = 0; y < kernel_order; y++ ) {
			// 			for (c = 0; c < nchannels; c += 8) {
			// 				// since the kernel is 16 bit ints, one 128-bit vector has 8 values 
			// 				// extracting to 128 bit in vector for simplicity. Could maybe extract to 256-bit vector for 16 shorts at a time,
			// 				// but seperating those 16 shorts into 4 separate 256 double vectors was troublesome and I gave up.

			// 				__m128i k4i = _mm_load_si128((__m128i_u*)&(kernels[m][x][y][c]));
      //         __m512d k8 = _mm512_cvtepi64_pd(_mm512_cvtepi16_epi64(k4i));
			// 				// __m256d k4_1 = _mm256_cvtepi32_pd(_mm_srai_epi32(_mm_unpacklo_epi16(k4i, k4i), 16));
			// 				// __m256d k4_2 = _mm256_cvtepi32_pd(_mm_srai_epi32(_mm_unpackhi_epi16(k4i, k4i), 16));

      //         __m512d i8 = _mm512_cvtps_pd(_mm256_loadu_ps(&(image[w+x][h+y][c])));
			// 				// __m256d i4_1 = _mm256_cvtps_pd(_mm_loadu_ps(&(image[w+x][h+y][c])));
			// 				// __m256d i4_2 = _mm256_cvtps_pd(_mm_loadu_ps(&(image[w+x][h+y][c+4])));

      //         sum_vec += _mm512_reduce_add_pd(_mm512_mul_pd(i8,k8));
			// 				// sum4 = _mm256_add_pd(sum4, _mm256_add_pd(_mm256_mul_pd(i4_1, k4_1), _mm256_mul_pd(i4_2, k4_2)));
			// 			}
			// 		}
			// 	}
      //   output[m][w][h] = sum_vec;
			// 	// _mm256_storeu_pd(sum_vec_arr, sum4); // No load single function for __m256d
			// 	// output[m][w][h] = (float) (sum_vec_arr[0] + sum_vec_arr[1] + sum_vec_arr[2] + sum_vec_arr[3]); // not using hadd because it refused to work, feel free to try fix
			// }
		}
	}
}

int main(int argc, char ** argv)
{
  //float image[W][H][C];
  //float kernels[M][C][K][K];
  //float output[M][W][H];
  
  float *** image;
  int16_t **** kernels;
  float *** control_output, *** output;
  long long mul_time, student_time;
  int width, height, kernel_order, nchannels, nkernels;
  struct timeval start_time;
  struct timeval stop_time;

  if ( argc != 6 ) {
    fprintf(stderr, "Usage: conv-harness <image_width> <image_height> <kernel_order> <number of channels> <number of kernels>\n");
    exit(1);
  }
  else {
    width = atoi(argv[1]);
    height = atoi(argv[2]);
    kernel_order = atoi(argv[3]);
    nchannels = atoi(argv[4]);
    nkernels = atoi(argv[5]);
  }
  switch ( kernel_order ) {
  case 1:
  case 3:
  case 5:
  case 7: break;
  default:
    fprintf(stderr, "FATAL: kernel_order must be 1, 3, 5 or 7, not %d\n",
            kernel_order);
    exit(1);
  }

  /* allocate the matrices */
  image = gen_random_3d_matrix_float(width+kernel_order, height + kernel_order,
                               nchannels);
  kernels = gen_random_4d_matrix_int16(nkernels, nchannels, kernel_order, kernel_order);
  output = new_empty_3d_matrix_float(nkernels, width, height);
  control_output = new_empty_3d_matrix_float(nkernels, width, height);

  //DEBUGGING(write_out(A, a_dim1, a_dim2));

  /* record starting time of David's code*/
  gettimeofday(&start_time, NULL);

  /* use a simple multichannel convolution routine to produce control result */
  multichannel_conv(image, kernels, control_output, width,
                    height, nchannels, nkernels, kernel_order);

    /* record David's finishing time */
  gettimeofday(&stop_time, NULL);
  mul_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("David conv time: %lld microseconds\n", mul_time);

  /* record starting time of student's code*/
  gettimeofday(&start_time, NULL);

  /* perform student's multichannel convolution */
  student_conv(image, kernels, output, width,
                    height, nchannels, nkernels, kernel_order);

  /* record finishing time */
  gettimeofday(&stop_time, NULL);
  student_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("Student conv time: %lld microseconds\n", student_time);
  printf("Approx Speedup = %f\n", (double)(mul_time/(double)student_time));

  DEBUGGING(write_out(output, nkernels, width, height));

  /* now check that the student's multichannel convolution routine
     gives the same answer as the known working version */
  check_result(output, control_output, nkernels, width, height);

  return 0;
}
