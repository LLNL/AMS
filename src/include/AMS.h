// Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

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

typedef enum { SinglePass = 0, Partition, Predicate } AMSExecPolicy;

typedef enum { None = 0, CSV, REDIS, HDF5 } AMSDBType;

typedef enum {
  FAISSMean =0,
  FAISSMax,
  DeltaUQ // Not supported
} AMSUQPolicy;

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

const char *AMSGetAllocatorName(AMSResourceType device);
void AMSSetupAllocator(const AMSResourceType device);
void AMSSetDefaultAllocator(const AMSResourceType device);
void AMSResourceInfo();
int AMSGetLocationId(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
