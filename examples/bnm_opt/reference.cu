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

#include <stdio.h>
#include <math.h>
#include "binomialOptions.h"
#include "realtype.h"

// Polynomial approximation of cumulative normal distribution function
static real CND(real d)
{
  const real       A1 = 0.31938153;
  const real       A2 = -0.356563782;
  const real       A3 = 1.781477937;
  const real       A4 = -1.821255978;
  const real       A5 = 1.330274429;
  const real RSQRT2PI = 0.39894228040143267793994605993438;

  real
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));

  real
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
    (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0)
    cnd = 1.0 - cnd;

  return cnd;
}

extern "C" void BlackScholesCall(
    real &callResult,
    TOptionData optionData
    )
{
  real S = optionData.S;
  real X = optionData.X;
  real T = optionData.T;
  real R = optionData.R;
  real V = optionData.V;

  real sqrtT = sqrt(T);
  real    d1 = (log(S / X) + (R + (real)0.5 * V * V) * T) / (V * sqrtT);
  real    d2 = d1 - V * sqrtT;
  real CNDD1 = CND(d1);
  real CNDD2 = CND(d2);

  //Calculate Call and Put simultaneously
  real expRT = exp(- R * T);
  callResult   = (real)(S * CNDD1 - X * expRT * CNDD2);
}