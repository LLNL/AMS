/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#ifdef __ENABLE_MPI__
#include <mpi.h>
#endif

#include "binomialOptions.h"
#include "kernel.hpp"
#include "realtype.h"

#define DOUBLE 0
#define FLOAT 1
#define INT 2
#define LONG 3

#define THREADBLOCK_SIZE 256
#define ELEMS_PER_THREAD (NUM_STEPS / THREADBLOCK_SIZE)

#define NUM_TRIALS 2


static double CND(double d)
{
  const double A1 = 0.31938153;
  const double A2 = -0.356563782;
  const double A3 = 1.781477937;
  const double A4 = -1.821255978;
  const double A5 = 1.330274429;
  const double RSQRT2PI = 0.39894228040143267793994605993438;

  double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

  double cnd = RSQRT2PI * exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0) cnd = 1.0 - cnd;

  return cnd;
}


void BlackScholesCall(real &callResult,
                      const real S,
                      const real X,
                      const real T,
                      const real R,
                      const real V)
{
  double sqrtT = sqrt(T);

  double d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
  double d2 = d1 - V * sqrtT;

  double CNDD1 = CND(d1);
  double CNDD2 = CND(d2);

  //Calculate Call and Put simultaneously
  double expRT = exp(-R * T);

  callResult = (real)(S * CNDD1 - X * expRT * CNDD2);
}


void writeQualityFile(const char *fileName,
                      void *ptr,
                      int type,
                      size_t numElements)
{
  FILE *fd = fopen(fileName, "wb");
  assert(fd && "Could Not Open File\n");
  fwrite(&numElements, sizeof(size_t), 1, fd);
  fwrite(&type, sizeof(int), 1, fd);
  if (type == DOUBLE)
    fwrite(ptr, sizeof(double), numElements, fd);
  else if (type == FLOAT)
    fwrite(ptr, sizeof(float), numElements, fd);
  else if (type == INT)
    fwrite(ptr, sizeof(int), numElements, fd);
  else
    assert(0 && "Not supported data type to write\n");
  fclose(fd);
}

void readData(FILE *fd, double **data, size_t *numElements)
{
  assert(fd && "File pointer is not valid\n");
  fread(numElements, sizeof(size_t), 1, fd);
  size_t elements = *numElements;
  double *ptr = (double *)malloc(sizeof(double) * elements);
  assert(ptr && "Could Not allocate pointer\n");
  *data = ptr;
  size_t i;
  int type;
  fread(&type, sizeof(int), 1, fd);
  if (type == DOUBLE) {
    fread(ptr, sizeof(double), elements, fd);
  } else if (type == FLOAT) {
    float *tmp = (float *)malloc(sizeof(float) * elements);
    fread(tmp, sizeof(float), elements, fd);
    for (i = 0; i < elements; i++) {
      ptr[i] = (double)tmp[i];
    }
    free(tmp);
  } else if (type == INT) {
    int *tmp = (int *)malloc(sizeof(int) * elements);
    fread(tmp, sizeof(int), elements, fd);
    for (i = 0; i < elements; i++) {
      ptr[i] = (double)tmp[i];
    }
    free(tmp);
  }
  return;
}

void readData(FILE *fd, float **data, size_t *numElements)
{
  assert(fd && "File pointer is not valid\n");
  fread(numElements, sizeof(size_t), 1, fd);
  size_t elements = *numElements;

  float *ptr = (float *)malloc(sizeof(float) * elements);
  assert(ptr && "Could Not allocate pointer\n");
  *data = ptr;

  size_t i;
  int type;
  fread(&type, sizeof(int), 1, fd);
  if (type == FLOAT) {
    fread(ptr, sizeof(float), elements, fd);
  } else if (type == DOUBLE) {
    double *tmp = (double *)malloc(sizeof(double) * elements);
    fread(tmp, sizeof(double), elements, fd);
    for (i = 0; i < elements; i++) {
      ptr[i] = (float)tmp[i];
    }
    free(tmp);
  } else if (type == INT) {
    int *tmp = (int *)malloc(sizeof(int) * elements);
    fread(tmp, sizeof(int), elements, fd);
    for (i = 0; i < elements; i++) {
      ptr[i] = (float)tmp[i];
    }
    free(tmp);
  }
  return;
}

void readData(FILE *fd, int **data, size_t *numElements)
{
  assert(fd && "File pointer is not valid\n");
  fread(numElements, sizeof(size_t), 1, fd);
  size_t elements = *numElements;

  int *ptr = (int *)malloc(sizeof(int) * elements);
  assert(ptr && "Could Not allocate pointer\n");
  *data = ptr;

  size_t i;
  int type;
  fread(&type, sizeof(int), 1, fd);
  if (type == INT) {
    fread(ptr, sizeof(int), elements, fd);
  } else if (type == DOUBLE) {
    double *tmp = (double *)malloc(sizeof(double) * elements);
    fread(tmp, sizeof(double), elements, fd);
    for (i = 0; i < elements; i++) {
      ptr[i] = (int)tmp[i];
    }
    free(tmp);
  } else if (type == FLOAT) {
    float *tmp = (float *)malloc(sizeof(float) * elements);
    fread(tmp, sizeof(float), elements, fd);
    for (i = 0; i < elements; i++) {
      ptr[i] = (int)tmp[i];
    }
    free(tmp);
  }
  return;
}

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////

real randData(real low, real high)
{
  real t = (real)rand() / (real)RAND_MAX;
  return ((real)1.0 - t) * low + t * high;
}

// Process an array of OptN options on GPU
extern "C" void binomialOptionsEntry(real *callValue,
                                     real *_S,
                                     real *_X,
                                     real *_R,
                                     real *_V,
                                     real *_T,
                                     size_t optN);


int main(int argc, char **argv)
{
  FILE *file;
  int size = 1;
  int rank = 0;

#ifdef __ENABLE_MPI__
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  if (argc != 3) {
    std::cout << "USAGE: " << argv[0] << " num-options batch_size";
    return EXIT_FAILURE;
  }


  size_t numOptions = std::atoi(argv[1]);
  size_t batch_size = std::atoi(argv[2]);

  bool write_output = false;


  int *otype;

  real sumDelta, sumRef, gpuTime, errorVal;

#define PAD 256
#define LINESIZE 64
  //  readData(file, &otype, &numOptions);
  //  readData(file, &S, &numOptions);
  //  readData(file, &X, &numOptions);
  //  readData(file, &R, &numOptions);
  //  readData(file, &V, &numOptions);
  //  readData(file, &T, &numOptions);

  batch_size = std::min(numOptions, batch_size);
  real *S = new real[batch_size];
  real *X = new real[batch_size];
  real *T = new real[batch_size];
  real *R = new real[batch_size];
  real *V = new real[batch_size];
  real *callValue = new real[batch_size];
  real *callValueBS = new real[batch_size];
  // Adding rank to allow ranks to compute different random data, yet remain deterministic
  srand(123 + rank);


  printf("Running GPU binomial tree...\n");

  auto start = std::chrono::high_resolution_clock::now();
  sumDelta = 0;
  sumRef = 0;

  BinomialOptions BO(batch_size, rank, size);
  std::vector<real> gpuTiming;

#ifdef __ENABLE_MPI__
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  auto t_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < numOptions; i += batch_size) {
    for (int j = 0; j < std::min(numOptions - i * batch_size, batch_size);
         j++) {
      S[j] = randData(5.0f, 30.0f);
      X[j] = randData(1.0f, 100.0f);
      T[j] = randData(0.25f, 10.0f);
      R[j] = 0.06f;
      V[j] = 0.10f;
      BlackScholesCall(callValueBS[j], S[j], X[j], T[j], R[j], V[j]);
    }

    auto start = std::chrono::high_resolution_clock::now();
    BO.run(callValue,
           S,
           X,
           R,
           V,
           T,
           std::min(numOptions - i * batch_size, batch_size));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<real> elapsed_seconds = end - start;
    gpuTime = (real)elapsed_seconds.count();
    double gDuration = gpuTime;


    for (size_t j = 0; j < std::min(numOptions - i * batch_size, batch_size);
         j++) {
      sumDelta += fabs(callValueBS[j] - callValue[j]);
      sumRef += fabs(callValueBS[j]);
    }

    errorVal = sumDelta / sumRef;
    double gErrorVal = errorVal;
#ifdef __ENABLE_MPI__
    MPI_Reduce(
        &errorVal, &gErrorVal, 1, BO_MPI_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(
        &gpuTime, &gDuration, 1, BO_MPI_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
#endif

    if (rank == 0) {
      real optionsPerSecond = std::min(numOptions - i * batch_size, batch_size);
      optionsPerSecond = (optionsPerSecond * size) / gpuTime;

      gErrorVal /= size;
      gpuTime = gDuration / size;
      gpuTiming.push_back(gpuTime);
      printf(
          "Error val: %g Throughput: %g (options/sec) Average duration: %g\n",
          gErrorVal,
          optionsPerSecond,
          gpuTime);
    }
  }
#ifdef __ENABLE_MPI__
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  if (rank == 0) {
    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<real> elapsed_seconds = t_end - t_start;
    std::cout << "Total Duration was " << elapsed_seconds.count() << " (s) \n";
    std::cout << "Total number of options was " << size * numOptions
              << " (s) \n";
    real sum_of_elems = std::accumulate(gpuTiming.begin(),
                                        gpuTiming.end(),
                                        decltype(gpuTiming)::value_type(0));
    std::cout << "Total GPU Time was " << sum_of_elems << " (s) \n";
    std::cout << "Average Throughput " << (size * numOptions) / sum_of_elems
              << " (options/sec) \n";
    std::cout << "Average Throughput (including host) "
              << numOptions / elapsed_seconds.count() << " (options/sec) \n";
  }

#ifdef __ENABLE_MPI__
  MPI_Finalize();
#endif

  if (errorVal > 5e-4) {
    printf("Test failed! %f\n", errorVal);
    exit(EXIT_FAILURE);
  }


  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
