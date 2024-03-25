/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS__
#define __AMS__

#include "AMS-config.h"

#ifdef __AMS_ENABLE_CALIPER__
#include <caliper/cali-manager.h>
#include <caliper/cali.h>
#define CALIPER(stmt) stmt
#else
#define CALIPER(stmt)
#endif

#ifdef __AMS_ENABLE_MPI__
#include <mpi.h>
#define MPI_CALL(stmt)                                                         \
  if (stmt != MPI_SUCCESS) {                                                   \
    fprintf(stderr, "Error in MPI-Call (File: %s, %d)\n", __FILE__, __LINE__); \
  }
#else
typedef void *MPI_Comm;
#define MPI_CALL(stm)
#endif

#ifdef __AMS_ENABLE_PERFFLOWASPECT__
#define PERFFASPECT() __attribute__((annotate("@critical_path()")))
#else
#define PERFFASPECT()
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*AMSPhysicFn)(void *, long, const void *const *, void *const *);

typedef void *AMSExecutor;

typedef enum { Single = 0, Double } AMSDType;

typedef enum {
  UNKNOWN = -1,
  HOST = 0,
  DEVICE = 1,
  PINNED = 2,
  RSEND
} AMSResourceType;

typedef enum { UBALANCED = 0, BALANCED } AMSExecPolicy;

typedef enum { DBNone = 0, DBCSV, DBREDIS, DBHDF5, DBRMQ } AMSDBType;

// TODO: create a cleaner interface that separates UQ type (FAISS, DeltaUQ) with policy (max, mean).
enum struct AMSUQPolicy {
  AMSUQPolicy_BEGIN = 0,
  FAISS_Mean,
  FAISS_Max,
  DeltaUQ_Mean,
  DeltaUQ_Max,
  RandomUQ,
  AMSUQPolicy_END
};

typedef struct ams_conf {
  const AMSExecPolicy ePolicy;
  const AMSDType dType;
  const AMSResourceType device;
  const AMSDBType dbType;
  AMSPhysicFn cBack;
  char *SPath;
  char *UQPath;
  char *DBPath;
  double threshold;
  const AMSUQPolicy uqPolicy;
  const int nClusters;
  int pId;
  int wSize;
} AMSConfig;

AMSExecutor AMSCreateExecutor(const AMSConfig config);

#ifdef __AMS_ENABLE_MPI__
void AMSDistributedExecute(AMSExecutor executor,
                           MPI_Comm Comm,
                           void *probDescr,
                           const int numElements,
                           const void **input_data,
                           void **output_data,
                           int inputDim,
                           int outputDim);
#endif
void AMSExecute(AMSExecutor executor,
                void *probDescr,
                const int numElements,
                const void **input_data,
                void **output_data,
                int inputDim,
                int outputDim);

void AMSDestroyExecutor(AMSExecutor executor);

#ifdef __AMS_ENABLE_MPI__
int AMSSetCommunicator(MPI_Comm Comm);
#endif

void AMSSetAllocator(AMSResourceType resource, const char *alloc_name);
const char *AMSGetAllocatorName(AMSResourceType device);

#ifdef __cplusplus
}
#endif

#endif
