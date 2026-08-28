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

#include "kernel.hpp"

#include <cuda_runtime.h>
#include <stdio.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "binomialOptions.h"
#include "realtype.h"

#ifdef USE_AMS
#include <AMS.h>
#endif

using namespace ams;


// Overloaded shortcut functions for different precision modes
#ifndef DOUBLE_PRECISION
__device__ inline float expiryCallValue(float S, float X, float vDt, int i)
{
  float d = S * __expf(vDt * (2.0f * i - NUM_STEPS)) - X;
  return (d > 0.0F) ? d : 0.0F;
}
#else
__device__ inline double expiryCallValue(double S, double X, double vDt, int i)
{
  double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
  return (d > 0.0) ? d : 0.0;
}
#endif


// GPU kernel
#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (NUM_STEPS / THREADBLOCK_SIZE)
#if NUM_STEPS % THREADBLOCK_SIZE
#error Bad constants
#endif

__global__ void static binomialOptionsKernel(const real* _S,
                                             const real* _X,
                                             const real* _vDt,
                                             const real* _puByDf,
                                             const real* _pdByDf,
                                             real* callValue)
{
  __shared__ real call_exchange[THREADBLOCK_SIZE + 1];

  const int tid = threadIdx.x;
  const real S = _S[blockIdx.x];
  const real X = _X[blockIdx.x];
  const real vDt = _vDt[blockIdx.x];
  const real puByDf = _puByDf[blockIdx.x];
  const real pdByDf = _pdByDf[blockIdx.x];

  real call[ELEMS_PER_THREAD + 1];
#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; ++i)
    call[i] = expiryCallValue(S, X, vDt, tid * ELEMS_PER_THREAD + i);

  if (tid == 0)
    call_exchange[THREADBLOCK_SIZE] = expiryCallValue(S, X, vDt, NUM_STEPS);

  int final_it = max(0, tid * ELEMS_PER_THREAD - 1);

#pragma unroll 16
  for (int i = NUM_STEPS; i > 0; --i) {
    call_exchange[tid] = call[0];
    __syncthreads();
    call[ELEMS_PER_THREAD] = call_exchange[tid + 1];
    __syncthreads();

    if (i > final_it) {
#pragma unroll
      for (int j = 0; j < ELEMS_PER_THREAD; ++j)
        call[j] = puByDf * call[j + 1] + pdByDf * call[j];
    }
  }

  if (tid == 0) {
    callValue[blockIdx.x] = call[0];
  }
}

__global__ static void preProcessKernel(const real* d_T,
                                        const real* d_R,
                                        const real* d_V,
                                        real* d_puByDf,
                                        real* d_pdByDf,
                                        real* d_vDt,
                                        size_t optN)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < optN) {
    const real T = d_T[i];
    const real R = d_R[i];
    const real V = d_V[i];

    const real dt = T / (real)NUM_STEPS;
    const real vDt = V * sqrt(dt);
    const real rDt = R * dt;
    const real If = exp(rDt);
    const real Df = exp(-rDt);
    const real u = exp(vDt);
    const real d = exp(-vDt);
    const real pu = (If - d) / (u - d);
    const real pd = (real)1.0 - pu;
    const real puByDf = pu * Df;
    const real pdByDf = pd * Df;

    d_puByDf[i] = puByDf;
    d_pdByDf[i] = pdByDf;
    d_vDt[i] = vDt;
  }
}


// Host-side interface to GPU binomialOptions
static void binomialOptionsGPU(real* d_CallValue,
                               const real* d_S,
                               const real* d_X,
                               const real* d_R,
                               const real* d_V,
                               const real* d_T,
                               real* d_puByDf,
                               real* d_pdByDf,
                               real* d_vDt,
                               size_t optN)
{
  int blockSize = 256;
  int numBlocks = (optN + blockSize - 1) / blockSize;

  preProcessKernel<<<numBlocks, blockSize>>>(
      d_T, d_R, d_V, d_puByDf, d_pdByDf, d_vDt, optN);


  binomialOptionsKernel<<<optN, THREADBLOCK_SIZE>>>(
      d_S, d_X, d_vDt, d_puByDf, d_pdByDf, d_CallValue);
}


BinomialOptions::BinomialOptions(unsigned int batchSize,
                                 int rank,
                                 int worldSize)
    : batchSize(batchSize), rank(rank), worldSize(worldSize)
{
  cudaMalloc((void**)&d_CallValue, sizeof(real) * batchSize);
  cudaMalloc((void**)&d_S, sizeof(real) * batchSize);
  cudaMalloc((void**)&d_X, sizeof(real) * batchSize);
  cudaMalloc((void**)&d_R, sizeof(real) * batchSize);
  cudaMalloc((void**)&d_V, sizeof(real) * batchSize);
  cudaMalloc((void**)&d_T, sizeof(real) * batchSize);
  cudaMalloc((void**)&d_puByDf, sizeof(real) * batchSize);
  cudaMalloc((void**)&d_pdByDf, sizeof(real) * batchSize);
  cudaMalloc((void**)&d_vDt, sizeof(real) * batchSize);

#ifdef USE_AMS
  const char* model_name = std::getenv("BO_MODEL_NAME");
  std::cout << "Model name is " << model_name << "\n";
  if (model_name) {
    model = AMSQueryModel(model_name);
  } else {
    model = AMSQueryModel("binomialOptions");
  }

  wf = AMSCreateExecutor(model, rank, worldSize);
#endif
}

#ifdef USE_AMS
void BinomialOptions::AMSRun(void* cls,
                             long numOptions,
                             void** inputs,
                             void** outputs)
{
  BinomialOptions* BO = reinterpret_cast<BinomialOptions*>(cls);
  binomialOptionsGPU((real*)outputs[0],
                     (const real*)inputs[0],
                     (const real*)inputs[1],
                     (const real*)inputs[2],
                     (const real*)inputs[3],
                     (const real*)inputs[4],
                     BO->d_puByDf,
                     BO->d_pdByDf,
                     BO->d_vDt,
                     numOptions);
}
#endif

void BinomialOptions::run(real* callValue,
                          real* _S,
                          real* _X,
                          real* _R,
                          real* _V,
                          real* _T,
                          size_t optN)
{
  cudaMemcpy(d_R, _R, sizeof(real) * optN, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, _V, sizeof(real) * optN, cudaMemcpyHostToDevice);
  cudaMemcpy(d_T, _T, sizeof(real) * optN, cudaMemcpyHostToDevice);
  cudaMemcpy(d_S, _S, sizeof(real) * optN, cudaMemcpyHostToDevice);
  cudaMemcpy(d_X, _X, sizeof(real) * optN, cudaMemcpyHostToDevice);

#ifdef USE_AMS

  SmallVector<AMSTensor> inputs;
  SmallVector<AMSTensor> inout;
  SmallVector<AMSTensor> outputs;
  inputs.push_back(std::move(AMSTensor::view(d_S,
                                             {static_cast<long>(optN), 1L},
                                             {1, 1},
                                             AMSResourceType::AMS_DEVICE)));
  inputs.push_back(std::move(AMSTensor::view(d_X,
                                             {static_cast<long>(optN), 1L},
                                             {1, 1},
                                             AMSResourceType::AMS_DEVICE)));
  inputs.push_back(std::move(AMSTensor::view(d_R,
                                             {static_cast<long>(optN), 1L},
                                             {1, 1},
                                             AMSResourceType::AMS_DEVICE)));
  inputs.push_back(std::move(AMSTensor::view(d_V,
                                             {static_cast<long>(optN), 1L},
                                             {1, 1},
                                             AMSResourceType::AMS_DEVICE)));
  inputs.push_back(std::move(AMSTensor::view(d_T,
                                             {static_cast<long>(optN), 1L},
                                             {1, 1},
                                             AMSResourceType::AMS_DEVICE)));


  outputs.push_back(std::move(AMSTensor::view(d_CallValue,
                                              {static_cast<long>(optN), 1},
                                              {1, 1},
                                              AMSResourceType::AMS_DEVICE)));

  DomainLambda OrigComputation = [&,
                                  this](const SmallVector<AMSTensor>& ams_ins,
                                        SmallVector<AMSTensor>& ams_inouts,
                                        SmallVector<AMSTensor>& ams_outs) {
    binomialOptionsGPU(ams_outs[0].data<real>(),
                       ams_ins[0].data<real>(),
                       ams_ins[1].data<real>(),
                       ams_ins[2].data<real>(),
                       ams_ins[3].data<real>(),
                       ams_ins[4].data<real>(),
                       d_puByDf,
                       d_pdByDf,
                       d_vDt,
                       ams_outs[0].shape()[0]);
  };


  AMSExecute(wf, OrigComputation, inputs, inout, outputs);
#else
  binomialOptionsGPU(
      d_CallValue, d_S, d_X, d_R, d_V, d_T, d_puByDf, d_pdByDf, d_vDt, optN);
#endif

  cudaMemcpy(callValue,
             d_CallValue,
             optN * sizeof(real),
             cudaMemcpyDeviceToHost);
}

BinomialOptions::~BinomialOptions()
{
  cudaFree(d_CallValue);
  cudaFree(d_S);
  cudaFree(d_X);
  cudaFree(d_vDt);
  cudaFree(d_puByDf);
  cudaFree(d_pdByDf);
  cudaFree(d_T);
  cudaFree(d_R);
  cudaFree(d_V);
}
