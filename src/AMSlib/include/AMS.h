/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS__
#define __AMS__

#include <cstdint>

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

typedef long AMSExecutor;
typedef int AMSCAbstrModel;

typedef enum { AMS_SINGLE = 0, AMS_DOUBLE } AMSDType;

typedef enum {
  AMS_UNKNOWN = -1,
  AMS_HOST = 0,
  AMS_DEVICE = 1,
  AMS_PINNED = 2,
  AMS_RSEND
} AMSResourceType;

typedef enum { AMS_UBALANCED = 0, AMS_BALANCED } AMSExecPolicy;

typedef enum { AMS_NONE = 0, AMS_CSV, AMS_REDIS, AMS_HDF5, AMS_RMQ } AMSDBType;

typedef enum {
  AMS_UQ_BEGIN = 0,
  AMS_FAISS_MEAN,
  AMS_FAISS_MAX,
  AMS_DELTAUQ_MEAN,
  AMS_DELTAUQ_MAX,
  AMS_RANDOM,
  AMS_UQ_END
} AMSUQPolicy;


#warning "I need to remove the exec policy when creating the executor"
AMSExecutor AMSCreateExecutor(AMSCAbstrModel model,
                              AMSExecPolicy exec_policy,
                              AMSDType data_type,
                              AMSResourceType resource_type,
                              AMSPhysicFn call_back,
                              int process_id,
                              int world_size);

AMSCAbstrModel AMSRegisterAbstractModel(const char *domain_name,
                                        AMSUQPolicy uq_policy,
                                        double threshold,
                                        const char *surrogate_path,
                                        const char *uq_path,
                                        const char *db_label,
                                        int num_clusters);

AMSCAbstrModel AMSQueryModel(const char *domain_model);

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
void configure_ams_fs_database(AMSDBType db_type, const char *db_path);

#ifdef __cplusplus
}
#endif

#endif
