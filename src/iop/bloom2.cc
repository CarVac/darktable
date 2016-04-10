/*
    This file is part of darktable,
    copyright (c) 2010-2012 Henrik Andersson.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/
//Matrix class
#ifndef MATRIX_H
#define MATRIX_H
#include <limits>
#include <algorithm>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using std::vector;

#include "assert.h" //Included later so NDEBUG has an effect

template <class T> class matrix
{
private:
  T *data;
  int num_rows;
  int num_cols;
  inline void slow_transpose_to(const matrix<T> &target) const;
  inline void fast_transpose_to(const matrix<T> &target) const;
  inline void transpose4x4_SSE(float *A, float *B, const int lda, const int ldb) const;
  inline void transpose_block_SSE4x4(float *A, float *B, const int n, const int m, const int lda,
                                     const int ldb, const int block_size) const;
  inline void transpose_scalar_block(float *A, float *B, const int lda, const int ldb,
                                     const int block_size) const;
  inline void transpose_block(float *A, float *B, const int n, const int m, const int lda, const int ldb,
                              const int block_size) const;

public:
  matrix(const int nrows = 0, const int ncols = 0);
  matrix(const matrix<T> &toCopy);
  ~matrix();
  void set_size(const int nrows, const int ncols);
  void free();
  int nr() const;
  int nc() const;
  T &operator()(const int row, const int col) const;
  // template <class U> //Never gets called if use matrix<U>
  matrix<T> &operator=(const matrix<T> &toCopy);
  template <class U> matrix<T> &operator=(const U value);
  template <class U> const matrix<T> add(const matrix<U> &rhs) const;
  template <class U> const matrix<T> add(const U value) const;
  template <class U> const matrix<T> &add_this(const U value);
  template <class U> const matrix<T> subtract(const matrix<U> &rhs) const;
  template <class U> const matrix<T> subtract(const U value) const;
  template <class U> const matrix<T> pointmult(const matrix<U> &rhs) const;
  template <class U> const matrix<T> mult(const U value) const;
  template <class U> const matrix<T> &mult_this(const U value);
  template <class U> const matrix<T> divide(const U value) const;
  inline void transpose_to(const matrix<T> &target) const;
  double sum();
  T max();
  T min();
  double mean();
  double variance();
};

template <class T> double sum(matrix<T> &mat);

template <class T> T max(matrix<T> &mat);

template <class T> T min(matrix<T> &mat);

template <class T> double mean(matrix<T> &mat);

template <class T> double variance(matrix<T> &mat);

template <class T, class U> const matrix<T> operator+(const matrix<T> &mat1, const matrix<U> &mat2);

template <class T, class U> const matrix<T> operator+(const U value, const matrix<T> &mat);

template <class T, class U> const matrix<T> operator+(const matrix<T> &mat, const U value);

template <class T, class U> const matrix<T> operator+=(matrix<T> &mat, const U value);

template <class T, class U> const matrix<T> operator-(const matrix<T> &mat1, const matrix<U> &mat2);

template <class T, class U> const matrix<T> operator-(const matrix<T> &mat, const U value);

template <class T, class U> const matrix<T> operator%(const matrix<T> &mat1, const matrix<U> &mat2);

template <class T, class U> const matrix<T> operator*(const U value, const matrix<T> &mat);

template <class T, class U> const matrix<T> operator*(const matrix<T> &mat, const U value);

template <class T, class U> const matrix<T> operator*=(matrix<T> &mat, const U value);

template <class T, class U> const matrix<T> operator/(const matrix<T> &mat1, const U value);

// IMPLEMENTATION:

template <class T> matrix<T>::matrix(const int nrows, const int ncols)
{
  assert(nrows >= 0 && ncols >= 0);
  num_rows = nrows;
  num_cols = ncols;
  if(nrows == 0 || ncols == 0)
  {
    data = nullptr;
  }
  else
  {
    data = new T[nrows * ncols];
  }
}

template <class T> matrix<T>::matrix(const matrix<T> &toCopy)
{
  if(this == &toCopy) return;

  num_rows = toCopy.num_rows;
  num_cols = toCopy.num_cols;
  data = new T[num_rows * num_cols];

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++) data[row * num_cols + col] = toCopy.data[row * num_cols + col];
}

template <class T> matrix<T>::~matrix()
{
  delete[] data;
}

template <class T> void matrix<T>::set_size(const int nrows, const int ncols)
{
  assert(nrows >= 0 && ncols >= 0);
  num_rows = nrows;
  num_cols = ncols;
  delete[] data;
  data = new(std::nothrow) T[nrows * ncols];
  //if(data == nullptr) std::cout << "matrix::set_size memory could not be alloc'd" << std::endl;
}

template <class T> void matrix<T>::free()
{
  set_size(0, 0);
}

template <class T> int matrix<T>::nr() const
{
  return num_rows;
}

template <class T> int matrix<T>::nc() const
{
  return num_cols;
}

template <class T> T &matrix<T>::operator()(const int row, const int col) const
{
  assert(row < num_rows && col < num_cols);
  return data[row * num_cols + col];
}

template <class T> // template<class U>
matrix<T> &matrix<T>::operator=(const matrix<T> &toCopy)
{
  if(this == &toCopy) return *this;

  set_size(toCopy.nr(), toCopy.nc());

#ifdef _OPENMP
#pragma omp parallel for shared(toCopy)
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++) data[row * num_cols + col] = toCopy.data[row * num_cols + col];
  return *this;
}

template <class T> template <class U> matrix<T> &matrix<T>::operator=(const U value)
{

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++) data[row * num_cols + col] = value;
  return *this;
}

template <class T> template <class U> const matrix<T> matrix<T>::add(const matrix<U> &rhs) const
{
  assert(num_rows == rhs.num_rows && num_cols == rhs.num_cols);
  matrix<T> result(num_rows, num_cols);

  T *pdata = data;
  int pnum_cols = num_cols;
#ifdef _OPENMP
#pragma omp parallel for shared(pdata, pnum_cols, result, rhs)
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++)
      result.data[row * num_cols + col] = data[row * num_cols + col] + rhs.data[row * num_cols + col];
  return result;
}

template <class T> template <class U> const matrix<T> matrix<T>::add(const U value) const
{
  matrix<T> result(num_rows, num_cols);

  T *pdata = data;
  int pnum_cols = num_cols;
#ifdef _OPENMP
#pragma omp parallel for shared(pdata, pnum_cols)
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++)
      result.data[row * num_cols + col] = data[row * num_cols + col] + value;
  return result;
}

template <class T> template <class U> const matrix<T> &matrix<T>::add_this(const U value)
{
  T *pdata = data;
  int pnum_cols = num_cols;
#ifdef _OPENMP
#pragma omp parallel for shared(pdata, pnum_cols)
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++) data[row * num_cols + col] += value;
  return *this;
}

template <class T> template <class U> const matrix<T> matrix<T>::subtract(const matrix<U> &rhs) const
{
  assert(num_rows == rhs.num_rows && num_cols == rhs.num_cols);
  matrix<T> result(num_rows, num_cols);

  T *pdata = data;
  int pnum_cols = num_cols;
#ifdef _OPENMP
#pragma omp parallel for shared(pdata, pnum_cols, result, rhs)
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++)
      result.data[row * num_cols + col] = data[row * num_cols + col] - rhs.data[row * num_cols + col];
  return result;
}

template <class T> template <class U> const matrix<T> matrix<T>::subtract(const U value) const
{
  matrix<T> result(num_rows, num_cols);

  T *pdata = data;
  int pnum_cols = num_cols;
#ifdef _OPENMP
#pragma omp parallel for shared(pdata, pnum_cols, result)
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++)
      result.data[row * num_cols + col] = data[row * num_cols + col] + value;
  return result;
}

template <class T> template <class U> const matrix<T> matrix<T>::pointmult(const matrix<U> &rhs) const
{
  matrix<T> result(num_rows, num_cols);

#ifdef _OPENMP
#pragma omp parallel for shared(result, rhs)
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++)
      result.data[row * num_cols + col] = data[row * num_cols + col] * rhs.data[row * num_cols + col];
  return result;
}

template <class T> template <class U> const matrix<T> matrix<T>::mult(const U value) const
{
  matrix<T> result(num_rows, num_cols);

#ifdef _OPENMP
#pragma omp parallel for shared(result)
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++)
      result.data[row * num_cols + col] = data[row * num_cols + col] * value;
  return result;
}

template <class T> template <class U> const matrix<T> &matrix<T>::mult_this(const U value)
{

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++) data[row * num_cols + col] *= value;
  return *this;
}

template <class T> template <class U> const matrix<T> matrix<T>::divide(const U value) const
{
  matrix<T> result(num_rows, num_cols);

#ifdef _OPENMP
#pragma omp parallel for shared(result)
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++)
      result.data[row * num_cols + col] = data[row * num_cols + col] / value;
  return result;
}

template <class T> inline void matrix<T>::slow_transpose_to(const matrix<T> &target) const
{
  assert(target.num_rows == num_cols && target.num_cols == num_rows);

#ifdef _OPENMP
#pragma omp parallel for shared(target)
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++) target.data[col * num_rows + row] = data[row * num_cols + col];
}

/*
template <> inline void matrix<float>::fast_transpose_to(const matrix<float> &target) const
{
  assert(target.num_rows == num_cols && target.num_cols == num_rows);

  transpose_block_SSE4x4(data, target.data, num_rows, num_cols, num_cols, num_rows, 16);
}
*/

// There is no fast transpose in the general case
template <class T> inline void matrix<T>::fast_transpose_to(const matrix<T> &target) const
{
  slow_transpose_to(target);
}

template <class T> inline void matrix<T>::transpose_to(const matrix<T> &target) const
{
  slow_transpose_to(target);
}

/*
template <> inline void matrix<float>::transpose_to(const matrix<float> &target) const
{
  // Fast transpose only work with matricies with dimensions of multiples of 16
  if((num_rows % 16 != 0) || (num_cols % 16 != 0))
    slow_transpose_to(target);
  else
    fast_transpose_to(target);
}
*/

/*
template <class T>
inline void matrix<T>::transpose4x4_SSE(float *A, float *B, const int lda, const int ldb) const
{
  __m128 row1 = _mm_load_ps(&A[0 * lda]);
  __m128 row2 = _mm_load_ps(&A[1 * lda]);
  __m128 row3 = _mm_load_ps(&A[2 * lda]);
  __m128 row4 = _mm_load_ps(&A[3 * lda]);
  _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
  _mm_store_ps(&B[0 * ldb], row1);
  _mm_store_ps(&B[1 * ldb], row2);
  _mm_store_ps(&B[2 * ldb], row3);
  _mm_store_ps(&B[3 * ldb], row4);
}
*/

/*
// block_size = 16 works best
template <class T>
inline void matrix<T>::transpose_block_SSE4x4(float *A, float *B, const int n, const int m, const int lda,
                                              const int ldb, const int block_size) const
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i = 0; i < n; i += block_size)
    for(int j = 0; j < m; j += block_size)
    {
      int max_i2 = i + block_size < n ? i + block_size : n;
      int max_j2 = j + block_size < m ? j + block_size : m;
      for(int i2 = i; i2 < max_i2; i2 += 4)
        for(int j2 = j; j2 < max_j2; j2 += 4)
          transpose4x4_SSE(&A[i2 * lda + j2], &B[j2 * ldb + i2], lda, ldb);
    }
}
*/

template <class T>
inline void matrix<T>::transpose_scalar_block(float *A, float *B, const int lda, const int ldb,
                                              const int block_size) const
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i = 0; i < block_size; i++)
  {
    for(int j = 0; j < block_size; j++)
    {
      B[j * ldb + i] = A[i * lda + j];
    }
  }
}

template <class T>
inline void matrix<T>::transpose_block(float *A, float *B, const int n, const int m, const int lda,
                                       const int ldb, const int block_size) const
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i = 0; i < n; i += block_size)
  {
    for(int j = 0; j < m; j += block_size)
    {
      transpose_scalar_block(&A[i * lda + j], &B[j * ldb + i], lda, ldb, block_size);
    }
  }
}

template <class T> double matrix<T>::sum()
{
  double sum = 0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++) sum += data[row * num_cols + col];
  return sum;
}

template <class T> T matrix<T>::max()
{
  T shared_max;

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    T max = std::numeric_limits<T>::min();
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(int row = 0; row < num_rows; row++)
      for(int col = 0; col < num_cols; col++) max = std::max(data[row * num_cols + col], max);
#ifdef _OPENMP
#pragma omp critical
#endif
    {
      shared_max = std::max(shared_max, max);
    }
  }
  return shared_max;
}


template <class T> T matrix<T>::min()
{
  T shared_min;

// T* pdata = data;
// int pnum_cols = num_cols;
#ifdef _OPENMP
#pragma omp parallel // shared(pdata, pnum_cols)
#endif
  {
    T min = std::numeric_limits<T>::max();
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(int row = 0; row < num_rows; row++)
      for(int col = 0; col < num_cols; col++) min = std::min(data[row * num_cols + col], min);
#ifdef _OPENMP
#pragma omp critical
#endif
    {
      shared_min = std::min(shared_min, min);
    }
  }
  return shared_min;
}

template <class T> double matrix<T>::mean()
{
  assert(num_rows > 0 && num_cols > 0);
  double size = num_rows * num_cols;
  return sum() / size;
}

template <class T> double matrix<T>::variance()
{
  double m = mean();
  double size = num_rows * num_cols;
  double variance = 0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : variance)
#endif
  for(int row = 0; row < num_rows; row++)
    for(int col = 0; col < num_cols; col++) variance += pow(data[row * num_cols + col] - m, 2);
  return variance / size;
}

// Non object functions

template <class T> double sum(matrix<T> &mat)
{
  return mat.sum();
}

template <class T> T max(matrix<T> &mat)
{
  return mat.max();
}

template <class T> T min(matrix<T> &mat)
{
  return mat.min();
}


template <class T> double mean(matrix<T> &mat)
{
  return mat.mean();
}

template <class T> double variance(matrix<T> &mat)
{
  return mat.variance();
}

template <class T, class U> const matrix<T> operator+(const matrix<T> &mat1, const matrix<U> &mat2)
{
  return mat1.add(mat2);
}

template <class T, class U> const matrix<T> operator+(const U value, const matrix<T> &mat)
{
  return mat.add(value);
}

template <class T, class U> const matrix<T> operator+(const matrix<T> &mat, const U value)
{
  return mat.add(value);
}

template <class T, class U> const matrix<T> operator+=(matrix<T> &mat, const U value)
{
  return mat.add_this(value);
}

template <class T, class U> const matrix<T> operator-(const matrix<T> &mat1, const matrix<U> &mat2)
{
  return mat1.subtract(mat2);
}

template <class T, class U> const matrix<T> operator-(const matrix<T> &mat, const U value)
{
  return mat.subtact(value);
}

template <class T, class U> const matrix<T> operator%(const matrix<T> &mat1, const matrix<U> &mat2)
{
  return mat1.pointmult(mat2);
}

template <class T, class U> const matrix<T> operator*(const U value, const matrix<T> &mat)
{
  return mat.mult(value);
}

template <class T, class U> const matrix<T> operator*(const matrix<T> &mat, const U value)
{
  return mat.mult(value);
}

template <class T, class U> const matrix<T> operator*=(matrix<T> &mat, const U value)
{
  return mat.mult_this(value);
}

template <class T, class U> const matrix<T> operator/(const matrix<T> &mat, const U value)
{
  return mat.divide(value);
}


// This uses a convolution forward and backward with a particular
// 4-length, 1-dimensional kernel to mimic a gaussian.
// In the first pass, it starts at 0, then goes out 4 standard deviations
// onto 0-clamped padding, then convolves back to the start.
// Naturally this attenuates the edges, so it does the same to all ones,
// and divides the image by that.

// Based on the paper "Recursive implementation of the Gaussian filter"
// in Signal Processing 44 (1995) 139-151
// Referencing code from here:
// https://github.com/halide/Halide/blob/e23f83b9bde63ed64f4d9a2fbe1ed29b9cfbf2e6/test/generator/gaussian_blur_generator.cpp
void iir_gaussblur(matrix<float> &image, const float sigma)
{
  const int height = image.nr();
  const int width = image.nc();

  // We set the padding to be 4 standard deviations so as to catch as much as possible.
  int paddedWidth = width + 4 * sigma + 3;
  int paddedHeight = height + 4 * sigma + 3;

  double q; // constant for computing coefficients
  if(sigma < 2.5)
  {
    q = 3.97156 - 4.14554 * sqrt(1 - 0.26891 * sigma);
  }
  else
  {
    q = 0.98711 * sigma - 0.96330;
  }

  double denom = 1.57825 + 2.44413 * q + 1.4281 * q * q + 0.422205 * q * q * q;
  double coeff[4];

  coeff[1] = (2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q) / denom;
  coeff[2] = (-1.4281 * q * q - 1.26661 * q * q * q) / denom;
  coeff[3] = (0.422205 * q * q * q) / denom;
  coeff[0] = 1 - (coeff[1] + coeff[2] + coeff[3]);

  // We blur ones in order to cancel the edge attenuation.

  // First we do horizontally.
  vector<double> attenuationX(paddedWidth);
  // Set up the boundary
  attenuationX[0] = coeff[0]; // times 1
  attenuationX[1] = coeff[0] + coeff[1] * attenuationX[0];
  attenuationX[2] = coeff[0] + coeff[1] * attenuationX[1] + coeff[2] * attenuationX[0];
  // Go over the image width
  for(int i = 3; i < width; i++)
  {
    attenuationX[i] = coeff[0] + // times 1
                      coeff[1] * attenuationX[i - 1] + coeff[2] * attenuationX[i - 2]
                      + coeff[3] * attenuationX[i - 3];
  }
  // Fill in the padding (which is all zeros)
  for(int i = width; i < paddedWidth; i++)
  {
    // All zeros, so no coeff[0]*1 here.
    attenuationX[i]
        = coeff[1] * attenuationX[i - 1] + coeff[2] * attenuationX[i - 2] + coeff[3] * attenuationX[i - 3];
  }
  // And go back.
  for(int i = paddedWidth - 3 - 1; i >= 0; i--)
  {
    attenuationX[i] = coeff[0] * attenuationX[i] + coeff[1] * attenuationX[i + 1]
                      + coeff[2] * attenuationX[i + 2] + coeff[3] * attenuationX[i + 3];
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i = 0; i < width; i++)
  {
    if(attenuationX[i] <= 0)
    {
      //std::cout << "gonna blow X" << std::endl;
    }
    else // we can invert this
    {
      attenuationX[i] = 1.0 / attenuationX[i];
    }
  }

  // And now vertically.
  vector<double> attenuationY(paddedHeight);
  // Set up the boundary
  attenuationY[0] = coeff[0]; // times 1
  attenuationY[1] = coeff[0] + coeff[1] * attenuationY[0];
  attenuationY[2] = coeff[0] + coeff[1] * attenuationY[1] + coeff[2] * attenuationY[0];
  // Go over the image height
  for(int i = 3; i < height; i++)
  {
    attenuationY[i] = coeff[0] + // times 1
                      coeff[1] * attenuationY[i - 1] + coeff[2] * attenuationY[i - 2]
                      + coeff[3] * attenuationY[i - 3];
  }
  // Fill in the padding (which is all zeros)
  for(int i = height; i < paddedHeight; i++)
  {
    // All zeros, so no coeff[0]*1 here.
    attenuationY[i]
        = coeff[1] * attenuationY[i - 1] + coeff[2] * attenuationY[i - 2] + coeff[3] * attenuationY[i - 3];
  }
  // And go back.
  for(int i = paddedHeight - 3 - 1; i >= 0; i--)
  {
    attenuationY[i] = coeff[0] * attenuationY[i] + coeff[1] * attenuationY[i + 1]
                      + coeff[2] * attenuationY[i + 2] + coeff[3] * attenuationY[i + 3];
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i = 0; i < height; i++)
  {
    if(attenuationY[i] <= 0)
    {
      //std::cout << "gonna blow Y" << std::endl;
    }
    else
    {
      attenuationY[i] = 1.0 / attenuationY[i];
    }
  }

#ifdef _OPENMP
#pragma omp parallel shared(image, coeff, attenuationX)
#endif
  {
    // X direction blurring.
    // We slice by individual rows.
    vector<double> image_x(paddedWidth);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(int row = 0; row < height; row++)
    {
      // Copy data into the temp.
      for(int col = 0; col < width; col++)
      {
        image_x[col] = double(image(row, col));
      }
      // Set up the boundary
      image_x[0] = coeff[0] * image_x[0];
      image_x[1] = coeff[0] * image_x[1] + coeff[1] * image_x[0];
      image_x[2] = coeff[0] * image_x[2] + coeff[1] * image_x[1] + coeff[2] * image_x[0];
      // Iterate over the main part of the image, except for the setup
      for(int col = 3; col < width; col++)
      {
        image_x[col] = coeff[0] * image_x[col]
                         + coeff[1] * image_x[col - 1]
                         + coeff[2] * image_x[col - 2]
                         + coeff[3] * image_x[col - 3];
      }
      // Iterate over the zeroed tail
      for(int col = width; col < paddedWidth; col++)
      {
        image_x[col] = coeff[1] * image_x[col - 1]
                         + coeff[2] * image_x[col - 2]
                         + coeff[3] * image_x[col - 3];
      }
      // And go back
      for(int col = paddedWidth - 3 - 1; col >= 0; col--)
      {
        image_x[col] = coeff[0] * image_x[col]
                         + coeff[1] * image_x[col + 1]
                         + coeff[2] * image_x[col + 2]
                         + coeff[3] * image_x[col + 3];
      }
// And undo the attenuation, copying back from the temp.
#ifdef _OPENMP
#pragma omp simd
#endif
      for(int col = 0; col < width; col++)
      {
        image(row, col) = image_x[col] * attenuationX[col];
      }
    }
  }

#ifdef _OPENMP
#pragma omp parallel shared(image, coeff, attenuationY)
#endif
  {
    // Y direction blurring. We slice into columns a whole number of cache lines wide.
    // Each cache line is 8 doubles wide.
    matrix<double> image_y;
    int thickness = 8; // of the slice
    image_y.set_size(paddedHeight, thickness);
    int slices = ceil(width / float(thickness));
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for(int slice = 0; slice < slices; slice++)
    {
      int offset = slice * thickness;
      int iter = thickness; // number of columns to loop through
      if(offset + thickness > width) // If it's the last row,
      {
        iter = width - offset; // Don't go beyond the bounds
      }

      // Copy data into the temp.
      if(iter < 8) // we can't SIMD this nicely
      {
        for(int row = 0; row < height; row++)
        {
          for(int col = 0; col < iter; col++)
          {
            image_y(row, col) = image(row, col + offset);
          }
        }
      }
      else // we can simd this
      {
        for(int row = 0; row < height; row++)
        {
#ifdef _OPENMP
#pragma omp simd
#endif
          for(int col = 0; col < 8; col++)
          {
            image_y(row, col) = image(row, col + offset);
          }
        }
      }

// Set up the boundary.
#ifdef _OPENMP
#pragma omp simd
#endif
      for(int col = 0; col < 8; col++)
      {
        image_y(0, col) = coeff[0] * image_y(0, col);
        image_y(1, col) = coeff[0] * image_y(1, col)
                            + coeff[1] * image_y(0, col);
        image_y(2, col) = coeff[0] * image_y(2, col)
                            + coeff[1] * image_y(1, col)
                            + coeff[2] * image_y(0, col);
      }
      // Iterate over the main part of the image, except for the setup.
      for(int row = 3; row < height; row++)
      {
#ifdef _OPENMP
#pragma omp simd
#endif
        for(int col = 0; col < 8; col++)
        {
          image_y(row, col) = coeff[0] * image_y(row, col)
                                + coeff[1] * image_y(row - 1, col)
                                + coeff[2] * image_y(row - 2, col)
                                + coeff[3] * image_y(row - 3, col);
        }
      }
      // Iterate over the zeroed tail
      for(int row = height; row < paddedHeight; row++)
      {
#ifdef _OPENMP
#pragma omp simd
#endif
        for(int col = 0; col < 8; col++)
        {
          image_y(row, col) = coeff[1] * image_y(row - 1, col)
                                + coeff[2] * image_y(row - 2, col)
                                + coeff[3] * image_y(row - 3, col);
        }
      }
      // And go back
      for(int row = paddedHeight - 3 - 1; row >= 0; row--)
      {
#ifdef _OPENMP
#pragma omp simd
#endif
        for(int col = 0; col < 8; col++)
        {
          image_y(row, col) = coeff[0] * image_y(row, col)
                                + coeff[1] * image_y(row + 1, col)
                                + coeff[2] * image_y(row + 2, col)
                                + coeff[3] * image_y(row + 3, col);
        }
      }
      // And undo the attenuation, copying back from the temp.
      if(iter < 8) // we can't SIMD this nicely
      {
        for(int row = 0; row < height; row++)
        {
          for(int col = 0; col < iter; col++)
          {
            image(row, col + offset) = image_y(row, col) * attenuationY[row];
          }
        }
      }
      else
      {
        for(int row = 0; row < height; row++)
        {
#ifdef _OPENMP
#pragma omp simd
#endif
          for(int col = 0; col < 8; col++)
          {
            image(row, col + offset) = image_y(row, col) * attenuationY[row];
          }
        }
      }
    }
  }
}



#endif // MATRIX_H
extern "C" {
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "bauhaus/bauhaus.h"
#include "common/opencl.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/tiling.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>
#include <inttypes.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define BOX_ITERATIONS 8
#define NUM_BUCKETS 4 /* OpenCL bucket chain size for tmp buffers; minimum 2 */
#define BLOCKSIZE                                                                                            \
  2048 /* maximum blocksize. must be a power of 2 and will be automatically reduced if needed */

#define CLIP(x) ((x < 0) ? 0.0 : (x > 1.0) ? 1.0 : x)
#define LCLIP(x) ((x < 0) ? 0.0 : (x > 100.0) ? 100.0 : x)
DT_MODULE_INTROSPECTION(2, dt_iop_bloom2_params_t)

typedef struct dt_iop_bloom2_params_t
{
  float size_r;
  float size_g;
  float size_b;
  float strength_r;
  float strength_g;
  float strength_b;
} dt_iop_bloom2_params_t;

typedef struct dt_iop_bloom2_gui_data_t
{
  GtkBox *vbox;
//  GtkWidget *together;
  GtkWidget *size_r, *size_g, *size_b;
  GtkWidget *strength_r, *strength_g, *strength_b;
} dt_iop_bloom2_gui_data_t;

typedef struct dt_iop_bloom2_data_t
{
//  int together;
  float size_r, size_g, size_b;
  float strength_r, strength_g, strength_b;
} dt_iop_bloom2_data_t;

/*
typedef struct dt_iop_bloom2_global_data_t
{
  int kernel_bloom2_threshold;
  int kernel_bloom2_hblur;
  int kernel_bloom2_vblur;
  int kernel_bloom2_mix;
} dt_iop_bloom2_global_data_t;
*/


const char *name()
{
  return _("physical bloom");
}

const char *description()
{
  return _("physically correct bloom on individual RGB channels");
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

int groups()
{
  return IOP_GROUP_EFFECT;
}

/*
void init_key_accels(dt_iop_module_so_t *self)
{
  dt_accel_register_slider_iop(self, FALSE, NC_("accel", "size"));
  dt_accel_register_slider_iop(self, FALSE, NC_("accel", "strength"));
}

void connect_key_accels(dt_iop_module_t *self)
{
  const dt_iop_bloom2_gui_data_t *g = (dt_iop_bloom2_gui_data_t *)self->gui_data;
  dt_accel_connect_slider_iop(self, "size", GTK_WIDGET(g->size));
  dt_accel_connect_slider_iop(self, "strength", GTK_WIDGET(g->strength));
}
*/

void process(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, const void *const ivoid,
             void *const ovoid, const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  // Get params
  dt_iop_bloom2_data_t *data = (dt_iop_bloom2_data_t *)piece->data;

  // Set up the color transformation to RGBA
  dt_iop_color_intent_t intent = DT_INTENT_PERCEPTUAL;
  const cmsHPROFILE Lab
      = dt_colorspaces_get_profile(DT_COLORSPACE_LAB, "", DT_PROFILE_DIRECTION_ANY)->profile;
  const cmsHPROFILE Rec2020
      = dt_colorspaces_get_profile(DT_COLORSPACE_LIN_REC2020, "", DT_PROFILE_DIRECTION_ANY)->profile;
  cmsHTRANSFORM transform_lab_to_lin_rgba, transform_lin_rgba_to_lab;
  transform_lab_to_lin_rgba = cmsCreateTransform(Lab, TYPE_LabA_FLT, Rec2020, TYPE_RGBA_FLT, intent, 0);
  transform_lin_rgba_to_lab = cmsCreateTransform(Rec2020, TYPE_RGBA_FLT, Lab, TYPE_LabA_FLT, intent, 0);

  const int width = roi_in->width;
  const int height = roi_in->height;
  
  // Temp buffer for the whole image
  float *rgbbufin = (float *)calloc(width * height * 4, sizeof(float));
  float *rgbbufout = (float *)calloc(width * height * 4, sizeof(float));

  // Turn Lab into linear Rec2020
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(rgbbufin, transform_lab_to_lin_rgba)
#endif
  for(int y = 0; y < height; y++)
  {
    const float *in = (float *)ivoid + y * width * 4;
    float *out = rgbbufin + y * width * 4;
    cmsDoTransform(transform_lab_to_lin_rgba, in, out, width);
  }

  // Set up radius and strength, dividing by a hundred to get 0:1 values.
  // One for each channel.
  const float rad_r = 0.01f * data->size_r * sqrt(float(width*width + height*height));
  const float rad_g = 0.01f * data->size_g * sqrt(float(width*width + height*height));
  const float rad_b = 0.01f * data->size_b * sqrt(float(width*width + height*height));
  const float scale = roi_in->scale / piece->iscale;
  const float radius[4] = {scale * rad_r, scale * rad_g, scale * rad_b, 0.0f};
  const float stren_r = 0.01f * data->strength_r;
  const float stren_g = 0.01f * data->strength_g;
  const float stren_b = 0.01f * data->strength_b;
  const float strength[4] = {stren_r, stren_g, stren_b, 0.0f};

  matrix<float> temp_image;
  temp_image.set_size(height, width);

  // Iterate over each color.
  for(int color = 0; color < 4; color++)
  {
    if(0.0f >= radius[color] || 0.0f >= strength[color])//inactive on this channel
    {
      printf("color: %d\n", color);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(rgbbufin, rgbbufout, color)
#endif
      for(int i = 0; i < height; i++)
      {
        for(int j = 0; j < width; j++)
        {
          rgbbufout[(j + i * width) * 4 + color] = rgbbufin[(j + i * width) * 4 + color];
        }
      }
    }
    else
    {
      printf("radius: %f\n", radius[color]);
      printf("strength: %f\n", strength[color]);
      // copy into a temporary matrix
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(rgbbufin, temp_image, color)
#endif
      for(int i = 0; i < height; i++)
      {
        for(int j = 0; j < width; j++)
        {
          temp_image(i, j) = rgbbufin[(j + i * width) * 4 + color];
        }
      }

      // run the blur
      iir_gaussblur(temp_image, radius[color]);

      // how much to keep the original image
      const float complement = 1 - strength[color];

      // copy out, using the opacity.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(rgbbufin, temp_image, rgbbufout, strength, color)
#endif
      for(int i = 0; i < height; i++)
      {
        for(int j = 0; j < width; j++)
        {
          rgbbufout[(j + i * width) * 4 + color] = strength[color] * temp_image(i, j)
                  + complement * rgbbufin[(j + i * width) * 4 + color];
        }
      }
    }
  }

  free(rgbbufin);
  // Now go back to Lab.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(rgbbufout, transform_lin_rgba_to_lab)
#endif
  for(int y = 0; y < height; y++)
  {
    const float *in = rgbbufout + y * width * 4;
    float *out = (float *)ovoid + y * width * 4;
    cmsDoTransform(transform_lin_rgba_to_lab, in, out, width);
  }
  free(rgbbufout);
}

/*
#ifdef HAVE_OPENCL
static int bucket_next(unsigned int *state, unsigned int max)
{
  const unsigned int current = *state;
  const unsigned int next = (current >= max - 1 ? 0 : current + 1);

  *state = next;

  return next;
}

int process_cl(struct dt_iop_module_t *self, dt_dev_pixelpipe_iop_t *piece, cl_mem dev_in, cl_mem dev_out,
               const dt_iop_roi_t *const roi_in, const dt_iop_roi_t *const roi_out)
{
  const dt_iop_bloom2_data_t *d = (dt_iop_bloom2_data_t *)piece->data;
  const dt_iop_bloom2_global_data_t *gd = (dt_iop_bloom2_global_data_t *)self->data;

  cl_int err = -999;
  cl_mem dev_tmp[NUM_BUCKETS] = { NULL };
  cl_mem dev_tmp1;
  cl_mem dev_tmp2;
  unsigned int state = 0;

  const int devid = piece->pipe->devid;
  const int width = roi_in->width;
  const int height = roi_in->height;

  const float threshold = d->threshold;

  const int rad = 256.0f * (fmin(100.0f, d->size + 1.0f) / 100.0f);
  const float _r = ceilf(rad * roi_in->scale / piece->iscale);
  const int radius = MIN(256.0f, _r);
  const float scale = 1.0f / exp2f(-1.0f * (fmin(100.0f, d->strength + 1.0f) / 100.0f));

  size_t maxsizes[3] = { 0 };     // the maximum dimensions for a work group
  size_t workgroupsize = 0;       // the maximum number of items in a work group
  unsigned long localmemsize = 0; // the maximum amount of local memory we can use
  size_t kernelworkgroupsize = 0; // the maximum amount of items in work group for this kernel

  // make sure blocksize is not too large
  int blocksize = BLOCKSIZE;
  if(dt_opencl_get_work_group_limits(devid, maxsizes, &workgroupsize, &localmemsize) == CL_SUCCESS
     && dt_opencl_get_kernel_work_group_size(devid, gd->kernel_bloom2_hblur, &kernelworkgroupsize)
        == CL_SUCCESS)
  {
    // reduce blocksize step by step until it fits to limits
    while(blocksize > maxsizes[0] || blocksize > maxsizes[1] || blocksize > kernelworkgroupsize
          || blocksize > workgroupsize || (blocksize + 2 * radius) * sizeof(float) > localmemsize)
    {
      if(blocksize == 1) break;
      blocksize >>= 1;
    }
  }
  else
  {
    blocksize = 1; // slow but safe
  }

  const size_t bwidth = width % blocksize == 0 ? width : (width / blocksize + 1) * blocksize;
  const size_t bheight = height % blocksize == 0 ? height : (height / blocksize + 1) * blocksize;

  size_t sizes[3];
  size_t local[3];

  for(int i = 0; i < NUM_BUCKETS; i++)
  {
    dev_tmp[i] = dt_opencl_alloc_device(devid, width, height, sizeof(float));
    if(dev_tmp[i] == NULL) goto error;
  }

  // gather light by threshold
  sizes[0] = ROUNDUPWD(width);
  sizes[1] = ROUNDUPHT(height);
  sizes[2] = 1;
  dev_tmp1 = dev_tmp[bucket_next(&state, NUM_BUCKETS)];
  dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_threshold, 0, sizeof(cl_mem), (void *)&dev_in);
  dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_threshold, 1, sizeof(cl_mem), (void *)&dev_tmp1);
  dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_threshold, 2, sizeof(int), (void *)&width);
  dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_threshold, 3, sizeof(int), (void *)&height);
  dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_threshold, 4, sizeof(float), (void *)&scale);
  dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_threshold, 5, sizeof(float), (void *)&threshold);
  err = dt_opencl_enqueue_kernel_2d(devid, gd->kernel_bloom2_threshold, sizes);
  if(err != CL_SUCCESS) goto error;

  if(radius != 0)
    for(int i = 0; i < BOX_ITERATIONS; i++)
    {
      // horizontal blur
      sizes[0] = bwidth;
      sizes[1] = ROUNDUPHT(height);
      sizes[2] = 1;
      local[0] = blocksize;
      local[1] = 1;
      local[2] = 1;
      dev_tmp2 = dev_tmp[bucket_next(&state, NUM_BUCKETS)];
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_hblur, 0, sizeof(cl_mem), (void *)&dev_tmp1);
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_hblur, 1, sizeof(cl_mem), (void *)&dev_tmp2);
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_hblur, 2, sizeof(int), (void *)&radius);
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_hblur, 3, sizeof(int), (void *)&width);
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_hblur, 4, sizeof(int), (void *)&height);
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_hblur, 5, sizeof(int), (void *)&blocksize);
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_hblur, 6, (blocksize + 2 * radius) * sizeof(float),
                               NULL);
      err = dt_opencl_enqueue_kernel_2d_with_local(devid, gd->kernel_bloom2_hblur, sizes, local);
      if(err != CL_SUCCESS) goto error;


      // vertical blur
      sizes[0] = ROUNDUPWD(width);
      sizes[1] = bheight;
      sizes[2] = 1;
      local[0] = 1;
      local[1] = blocksize;
      local[2] = 1;
      dev_tmp1 = dev_tmp[bucket_next(&state, NUM_BUCKETS)];
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_vblur, 0, sizeof(cl_mem), (void *)&dev_tmp2);
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_vblur, 1, sizeof(cl_mem), (void *)&dev_tmp1);
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_vblur, 2, sizeof(int), (void *)&radius);
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_vblur, 3, sizeof(int), (void *)&width);
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_vblur, 4, sizeof(int), (void *)&height);
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_vblur, 5, sizeof(int), (void *)&blocksize);
      dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_vblur, 6, (blocksize + 2 * radius) * sizeof(float),
                               NULL);
      err = dt_opencl_enqueue_kernel_2d_with_local(devid, gd->kernel_bloom2_vblur, sizes, local);
      if(err != CL_SUCCESS) goto error;
    }

  // mixing out and in -> out
  sizes[0] = ROUNDUPWD(width);
  sizes[1] = ROUNDUPHT(height);
  sizes[2] = 1;
  dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_mix, 0, sizeof(cl_mem), (void *)&dev_in);
  dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_mix, 1, sizeof(cl_mem), (void *)&dev_tmp1);
  dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_mix, 2, sizeof(cl_mem), (void *)&dev_out);
  dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_mix, 3, sizeof(int), (void *)&width);
  dt_opencl_set_kernel_arg(devid, gd->kernel_bloom2_mix, 4, sizeof(int), (void *)&height);
  err = dt_opencl_enqueue_kernel_2d(devid, gd->kernel_bloom2_mix, sizes);
  if(err != CL_SUCCESS) goto error;

  for(int i = 0; i < NUM_BUCKETS; i++)
    if(dev_tmp[i] != NULL) dt_opencl_release_mem_object(dev_tmp[i]);
  return TRUE;

error:
  for(int i = 0; i < NUM_BUCKETS; i++)
    if(dev_tmp[i] != NULL) dt_opencl_release_mem_object(dev_tmp[i]);
  dt_print(DT_DEBUG_OPENCL, "[opencl_bloom2] couldn't enqueue kernel! %d\n", err);
  return FALSE;
}
#endif

void tiling_callback(struct dt_iop_module_t *self, struct dt_dev_pixelpipe_iop_t *piece,
                     const dt_iop_roi_t *roi_in, const dt_iop_roi_t *roi_out,
                     struct dt_develop_tiling_t *tiling)
{
  const dt_iop_bloom2_data_t *d = (dt_iop_bloom2_data_t *)piece->data;

  const int rad = 256.0f * (fmin(100.0f, d->size + 1.0f) / 100.0f);
  const float _r = ceilf(rad * roi_in->scale / piece->iscale);
  const int radius = MIN(256.0f, _r);

  tiling->factor = 2.0f + NUM_BUCKETS * 0.25f; // in + out + NUM_BUCKETS * 0.25 tmp
  tiling->maxbuf = 1.0f;
  tiling->overhead = 0;
  tiling->overlap = 5 * radius; // This is a guess. TODO: check if that's sufficiently large
  tiling->xalign = 1;
  tiling->yalign = 1;
  return;
}

void init_global(dt_iop_module_so_t *module)
{
  const int program = 12; // bloom2.cl, from programs.conf
  dt_iop_bloom2_global_data_t *gd = (dt_iop_bloom2_global_data_t *)malloc(sizeof(dt_iop_bloom2_global_data_t));
  module->data = gd;
  gd->kernel_bloom2_threshold = dt_opencl_create_kernel(program, "bloom2_threshold");
  gd->kernel_bloom2_hblur = dt_opencl_create_kernel(program, "bloom2_hblur");
  gd->kernel_bloom2_vblur = dt_opencl_create_kernel(program, "bloom2_vblur");
  gd->kernel_bloom2_mix = dt_opencl_create_kernel(program, "bloom2_mix");
}

void cleanup_global(dt_iop_module_so_t *module)
{
  const dt_iop_bloom2_global_data_t *gd = (dt_iop_bloom2_global_data_t *)module->data;
  dt_opencl_free_kernel(gd->kernel_bloom2_threshold);
  dt_opencl_free_kernel(gd->kernel_bloom2_hblur);
  dt_opencl_free_kernel(gd->kernel_bloom2_vblur);
  dt_opencl_free_kernel(gd->kernel_bloom2_mix);
  free(module->data);
  module->data = NULL;
}
*/

/*
static void together_callback(GtkWidget *w, dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_bloom2_params_t *p = (dt_iop_bloom2_params_t *)self->params;
  p->together = (dt_bauhaus_combobox_get(w) == 0) ? 1 : 0;
  dt_dev_add_history_item(darktable.develop, self, TRUE);
  if(p->together)
  {
    darktable.gui->reset = true;
    p->strength_g = p->strength_r;
    p->strength_b = p->strength_r;
    p->size_g = p->size_r;
    p->size_b = p->size_r;
    darktable.gui->reset = false;
  }
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}
*/

static void strength_r_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(darktable.gui->reset) return;
  dt_iop_bloom2_params_t *p = (dt_iop_bloom2_params_t *)self->params;
  p->strength_r = dt_bauhaus_slider_get(slider);
/*  if(p->together)
  {
    darktable.gui->reset = true;
    p->strength_g = p->strength_r;
    p->strength_b = p->strength_r;
    darktable.gui->reset = false;
  }*/
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void strength_g_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(darktable.gui->reset) return;
  dt_iop_bloom2_params_t *p = (dt_iop_bloom2_params_t *)self->params;
  p->strength_g = dt_bauhaus_slider_get(slider);
/*  if(p->together)
  {
    darktable.gui->reset = true;
    p->strength_r = p->strength_g;
    p->strength_b = p->strength_g;
    darktable.gui->reset = false;
  }*/
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void strength_b_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(darktable.gui->reset) return;
  dt_iop_bloom2_params_t *p = (dt_iop_bloom2_params_t *)self->params;
  p->strength_b = dt_bauhaus_slider_get(slider);
/*  if(p->together)
  {
    darktable.gui->reset = true;
    p->strength_r = p->strength_b;
    p->strength_g = p->strength_b;
    darktable.gui->reset = false;
  }*/
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

static void size_r_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(darktable.gui->reset) return;
  dt_iop_bloom2_params_t *p = (dt_iop_bloom2_params_t *)self->params;
  p->size_r = dt_bauhaus_slider_get(slider);
/*  if(p->together)
  {
    darktable.gui->reset = true;
    p->size_g = p->size_r;
    p->size_b = p->size_r;
    darktable.gui->reset = false;
  }*/
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}
static void size_g_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(darktable.gui->reset) return;
  dt_iop_bloom2_params_t *p = (dt_iop_bloom2_params_t *)self->params;
  p->size_g = dt_bauhaus_slider_get(slider);
/*  if(p->together)
  {
    darktable.gui->reset = true;
    p->size_r = p->size_g;
    p->size_b = p->size_g;
    darktable.gui->reset = false;
  }*/
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}
static void size_b_callback(GtkWidget *slider, gpointer user_data)
{
  dt_iop_module_t *self = (dt_iop_module_t *)user_data;
  if(darktable.gui->reset) return;
  dt_iop_bloom2_params_t *p = (dt_iop_bloom2_params_t *)self->params;
  p->size_b = dt_bauhaus_slider_get(slider);
/*  if(p->together)
  {
    darktable.gui->reset = true;
    p->size_r = p->size_b;
    p->size_g = p->size_b;
    darktable.gui->reset = false;
  }*/
  dt_dev_add_history_item(darktable.develop, self, TRUE);
}

void commit_params(struct dt_iop_module_t *self, dt_iop_params_t *p1, dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_bloom2_params_t *p = (dt_iop_bloom2_params_t *)p1;
  dt_iop_bloom2_data_t *d = (dt_iop_bloom2_data_t *)piece->data;

//  d->together = p->together;
  d->strength_r = p->strength_r;
  d->strength_g = p->strength_g;
  d->strength_b = p->strength_b;
  d->size_r = p->size_r;
  d->size_g = p->size_g;
  d->size_b = p->size_b;
}

void init_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_bloom2_data_t));
  self->commit_params(self, self->default_params, pipe, piece);
}

void cleanup_pipe(struct dt_iop_module_t *self, dt_dev_pixelpipe_t *pipe, dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_update(struct dt_iop_module_t *self)
{
  const dt_iop_module_t *module = (dt_iop_module_t *)self;
  const dt_iop_bloom2_params_t *p = (dt_iop_bloom2_params_t *)module->params;
  dt_iop_bloom2_gui_data_t *g = (dt_iop_bloom2_gui_data_t *)self->gui_data;
//  dt_bauhaus_combobox_set(g->together, p->together);
  dt_bauhaus_slider_set(g->size_r, p->size_r);
  dt_bauhaus_slider_set(g->size_g, p->size_g);
  dt_bauhaus_slider_set(g->size_b, p->size_b);
  dt_bauhaus_slider_set(g->strength_r, p->strength_r);
  dt_bauhaus_slider_set(g->strength_g, p->strength_g);
  dt_bauhaus_slider_set(g->strength_b, p->strength_b);
}

void init(dt_iop_module_t *module)
{
  module->params = calloc(1, sizeof(dt_iop_bloom2_params_t));
  module->default_params = calloc(1, sizeof(dt_iop_bloom2_params_t));
  module->default_enabled = 0;
  module->priority = 499; // module order created by iop_dependencies.py, do not edit!
  module->params_size = sizeof(dt_iop_bloom2_params_t);
  module->gui_data = NULL;
  //dt_iop_bloom2_params_t tmp = (dt_iop_bloom2_params_t){1, 5, 5, 5, 5, 5, 5};
  dt_iop_bloom2_params_t tmp = (dt_iop_bloom2_params_t){5, 5, 5, 5, 5, 5};
  memcpy(module->params, &tmp, sizeof(dt_iop_bloom2_params_t));
  memcpy(module->default_params, &tmp, sizeof(dt_iop_bloom2_params_t));
}

void cleanup(dt_iop_module_t *module)
{
  free(module->params);
  module->params = NULL;
}

void gui_init(struct dt_iop_module_t *self)
{
  self->gui_data = malloc(sizeof(dt_iop_bloom2_gui_data_t));
  dt_iop_bloom2_gui_data_t *g = (dt_iop_bloom2_gui_data_t *)self->gui_data;
  const dt_iop_bloom2_params_t *p = (dt_iop_bloom2_params_t *)self->params;

  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_BAUHAUS_SPACE);

  /* together
  g->together = dt_bauhaus_combobox_new(self);
  dt_bauhaus_widget_set_label(g->together, NULL, _("channel selector"));
  dt_bauhaus_combobox_add(g->together, _("independent"));
  dt_bauhaus_combobox_add(g->together, _("all channels together"));*/

  /* size */
  g->size_r = dt_bauhaus_slider_new_with_range(self, 0.0, 20.0, 0.01, p->size_r, 2);
  dt_bauhaus_slider_set_format(g->size_r, "%.00f%%");
  dt_bauhaus_widget_set_label(g->size_r, NULL, _("red size"));
  gtk_widget_set_tooltip_text(g->size_r, _("the size of the red channel bloom"));
  g->size_g = dt_bauhaus_slider_new_with_range(self, 0.0, 20.0, 0.01, p->size_g, 2);
  dt_bauhaus_slider_set_format(g->size_g, "%.00f%%");
  dt_bauhaus_widget_set_label(g->size_g, NULL, _("green size"));
  gtk_widget_set_tooltip_text(g->size_g, _("the size of the green channel bloom"));
  g->size_b = dt_bauhaus_slider_new_with_range(self, 0.0, 20.0, 0.01, p->size_b, 2);
  dt_bauhaus_slider_set_format(g->size_b, "%.00f%%");
  dt_bauhaus_widget_set_label(g->size_b, NULL, _("blue size"));
  gtk_widget_set_tooltip_text(g->size_b, _("the size of the blue channel bloom"));

  /* strength */
  g->strength_r = dt_bauhaus_slider_new_with_range(self, 0.0, 100.0, 0.01, p->strength_r, 2);
  dt_bauhaus_slider_set_format(g->strength_r, "%.00f%%");
  dt_bauhaus_widget_set_label(g->strength_r, NULL, _("red strength"));
  gtk_widget_set_tooltip_text(g->strength_r, _("how much of the red light gets blurred"));
  g->strength_g = dt_bauhaus_slider_new_with_range(self, 0.0, 100.0, 0.01, p->strength_g, 2);
  dt_bauhaus_slider_set_format(g->strength_g, "%.00f%%");
  dt_bauhaus_widget_set_label(g->strength_g, NULL, _("green strength"));
  gtk_widget_set_tooltip_text(g->strength_g, _("how much of the green light gets blurred"));
  g->strength_b = dt_bauhaus_slider_new_with_range(self, 0.0, 100.0, 0.01, p->strength_b, 2);
  dt_bauhaus_slider_set_format(g->strength_b, "%.00f%%");
  dt_bauhaus_widget_set_label(g->strength_b, NULL, _("blue strength"));
  gtk_widget_set_tooltip_text(g->strength_b, _("how much of the blue light gets blurred"));

//  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->together), TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->size_r), TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->size_g), TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->size_b), TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->strength_r), TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->strength_g), TRUE, TRUE, 0);
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(g->strength_b), TRUE, TRUE, 0);

//  g_signal_connect(G_OBJECT(g->together), "value-changed", G_CALLBACK(together_callback), self);
  g_signal_connect(G_OBJECT(g->size_r), "value-changed", G_CALLBACK(size_r_callback), self);
  g_signal_connect(G_OBJECT(g->size_g), "value-changed", G_CALLBACK(size_g_callback), self);
  g_signal_connect(G_OBJECT(g->size_b), "value-changed", G_CALLBACK(size_b_callback), self);
  g_signal_connect(G_OBJECT(g->strength_r), "value-changed", G_CALLBACK(strength_r_callback), self);
  g_signal_connect(G_OBJECT(g->strength_g), "value-changed", G_CALLBACK(strength_g_callback), self);
  g_signal_connect(G_OBJECT(g->strength_b), "value-changed", G_CALLBACK(strength_b_callback), self);
}

void gui_cleanup(struct dt_iop_module_t *self)
{
  free(self->gui_data);
  self->gui_data = NULL;
}
}
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.sh
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
