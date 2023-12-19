/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef REDIST_LOAD_HPP
#define REDIST_LOAD_HPP

#include <mpi.h>

#include <cassert>
#include <cstdio>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "AMS.h"
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"


namespace ams
{

/**
 * @brief A class that balances workload across a communicator.
 *
 * @details This is a RAII class modeling a load balancing transaction in 3 phases.
 *
 * Assuming a code that performs the following pseudo-code:
 *
 * Type inputs[M][elements], outputs[N][elements];
 * expensiveCalc(inputs, outputs);
 *
 * The use can use the AMSLoadBalancer and transform the code as follows:
 *
 * Type inputs[M][elements], outputs[N][elements];
 * AMSLoadBalancer obj;                           // <- Phase 1
 * Type **lbInputs = obj.scatterInputs(inputs);   // <- Phase 2
 * expensiveCalc(lbInputs, obj.outputs());
 * gatherOutputs(outputs)                         // Phase 3
 *
 * Phase 1: Upon object creation all ranks collaboratively identify the total workload
 * across all ranks in the communicator. The root rank stores information regarding the
 * precise number of load of every rank.
 *
 * Phase 2: We balance the 2-D vector across all ranks. In the AMSLoadBalancer.inputs() the object
 * stores elements of other ranks.
 *
 * Phase 3: We reverse the load balancing elements to their initial potions.
 *
 * Upon object deletion heap allocated memory will be freed.
 */
template <typename FPTypeValue>
class AMSLoadBalancer
{
  /** @brief The rank id of the current process in the Provided communicator */
  int rId;

  /** @brief The total number of processes in the world */
  int worldSize;

  /** @brief The initial number of elements this process (before load balancing) */
  int localLoad;

  /** @brief The total number of elements across all processes in the communicator */
  int globalLoad;

  /** @brief The balanced number of elements this process */
  int balancedLoad;

  /** @brief The current communicator to load balance. */
  MPI_Comm Comm;

  /** @brief Integer array specifying the displacement from which to take the outgoing data to process for un-balanced data. */
  int *displs;

  /** @brief An integer array (of length group size) specifying the
   * number of elements to send to each processor before load balancing( Significant only at root )
   **/
  int *dataElements;

  /** @brief Integer array specifying the displacement from which to take the outgoing data to process for balanced data. */
  int *balancedDispls;

  /** @brief An integer array (of length group size) specifying the
   * number of elements to send to each processor to perform load balancing (Significant only at root)
   **/
  int *balancedElements;

  /** @brief Vector storing the load balanced inputs of this rank.*/
  std::vector<FPTypeValue *> distInputs;

  /** @brief Vector storing the load balanced outputs of this rank.*/
  std::vector<FPTypeValue *> distOutputs;

  /** @brief The id of the root rank (In this rank all data will be gahtered and scattered */
  const int root = 0;

  /** @brief The memory location of the data (GPU (DEVICE), CPU (HOST) ) */
  AMSResourceType resource;

private:
  /** @brief Computes the number of balanced elements each process will gather and initializes
   * memory structures.
   * @param[in] numIn The number of input dimensions
   * @param[in] numOut The number of input dimensions
   * @param[in] resource The resource type to allocate data in.

   * @details The function computes the total number of elements every rank will need to balance.
   * It initializes the 'dataElements', 'displs' on the root node and the localLoad, balancedLoad
   * across all ranks.
  */
PERFFASPECT()
  void init(int numIn, int numOut, AMSResourceType resource)
  {
    // We need to store information
    if (rId == root) {
      dataElements =
          ams::ResourceManager::allocate<int>(worldSize, AMSResourceType::HOST);
      displs = ams::ResourceManager::allocate<int>(worldSize + 1,
                                                   AMSResourceType::HOST);
    }

    // Gather the the number of items from each rank
    int rc = MPI_Gather(reinterpret_cast<const void *>(&localLoad),
                        1,
                        MPI_INT,
                        reinterpret_cast<void *>(dataElements),
                        1,
                        MPI_INT,
                        root,
                        Comm);
    CFATAL(LoadBalance, rc != MPI_SUCCESS, "Cannot gather per rank sizes")

    // Populate displacement array
    if (rId == 0) {
      globalLoad = 0;
      displs[0] = static_cast<int>(0);
      for (size_t i = 0ul; i < worldSize; ++i) {
        displs[i + 1] = dataElements[i] + displs[i];
      }
      globalLoad = displs[worldSize];
    }

    balancedLoad = computeBalanceLoad();

    if (rId == root) {
      balancedElements =
          ResourceManager::allocate<int>(worldSize, AMSResourceType::HOST);
      balancedDispls =
          ResourceManager::allocate<int>(worldSize, AMSResourceType::HOST);
      for (int i = 0; i < worldSize; i++) {
        balancedElements[i] = (globalLoad / worldSize) +
                              static_cast<int>(i < (globalLoad % worldSize));
        if (i != 0)
          balancedDispls[i] = balancedElements[i] + balancedDispls[i - 1];
        else
          balancedDispls[i] = 0;
      }
    }

    for (int i = 0; i < numIn; i++) {
      distInputs.push_back(
          ams::ResourceManager::allocate<FPTypeValue>(balancedLoad, resource));
    }

    for (int i = 0; i < numOut; i++) {
      distOutputs.push_back(
          ams::ResourceManager::allocate<FPTypeValue>(balancedLoad, resource));
    }
  }

  /** @brief Computes the number of elements every rank will receive after balancing.
   *  @returns the number of elements computed by this rank.
   **/
PERFFASPECT()
  int computeBalanceLoad()
  {
    int rc = MPI_Bcast(&globalLoad, 1, MPI_INT, root, Comm);
    CFATAL(LoadBalance, rc != MPI_SUCCESS, "Cannot broadcast global load")

    int load = (globalLoad / worldSize) +
               static_cast<int>(rId < (globalLoad % worldSize));
    return load;
  }

  /**
 *  @brief Take a vector of FPTypeValue from each rank and
 *  gather them at root and scater them back to the ranks (balanced or unbalanced) based on
 *  the values stored in 'sNElems'\.
 *
 *  @param[in] src Data array of items residing on the caller rank.
 *  \param[out] dest Data array of items after re-distributing them into their ranks.
 *  \param[out] buffer Output vetor that contains all the data gathered from
 *                       all the ranks. (significant only at root)
 *  \param[in] gNElems Lengths of the src arrays gathered from all the
 *                         ranks. (significant only at root)
 *  \param[in] gDElems Displacement in respect to the buffer vector. Data from Rank 'R' will be
 *                  stored under location buffer[gDElems[R]]. (significant only at root)
 *  \param[in] dType MPI data type of the elements to be send. Must much the respective FPTypeValue.
 *  \param[in] sNElems Lengths of the dest arrays scattered from root to all ranks
 *  \param[in] sDElems Displacement in respect to the buffer that will be scattered across all ranks. Root rank will send
 *                  to Rank 'R' the data under location buffer[gDElems[R]]. (significant only at root)
 *  \param[in] gElems number of elements this rank will send to root rank.
 *  \param[in] sElems number of elements this rank will receive from root rank.
 *
 *  \return void.
 */
PERFFASPECT()
  void distribute(FPTypeValue *src,
                  FPTypeValue *dest,
                  FPTypeValue *buffer,
                  int *gNElems,
                  int *gDElems,
                  int *sNElems,
                  int *sDElems,
                  MPI_Datatype dType,
                  int gElems,
                  int sElems)
  {
    int rc = MPI_Gatherv(
        src, gElems, dType, buffer, gNElems, gDElems, dType, root, Comm);

    CFATAL(LoadBalance, rc != MPI_SUCCESS, "Cannot GatherV data to root")

    rc = MPI_Scatterv(
        buffer, sNElems, sDElems, dType, dest, sElems, dType, root, Comm);

    CFATAL(LoadBalance, rc != MPI_SUCCESS, "Cannot ScatterV data from root")
  }

  /**
 *  @brief Take a vector of vectors of FPTypeValue from each rank and
 *  gather them at root and scater them back to the ranks (balanced or unbalanced) based on
 *  the values stored in 'sNElems'\.
 *
 *  @param[in] src Data array of items residing on the caller rank.
 *  \param[out] dest Data array of items after re-distributing them into their ranks.
 *  \param[out] buffer Output vetor that contains all the data gathered from
 *                       all the ranks. (significant only at root)
 *  \param[in] gNElems Lengths of the src arrays gathered from all the
 *                         ranks. (significant only at root)
 *  \param[in] gDElems Displacement in respect to the buffer vector. Data from Rank 'R' will be
 *                  stored under location buffer[gDElems[R]]. (significant only at root)
 *  \param[in] dType MPI data type of the elements to be send. Must much the respective FPTypeValue.
 *  \param[in] sNElems Lengths of the dest arrays scattered from root to all ranks
 *  \param[in] sDElems Displacement in respect to the buffer that will be scattered across all ranks. Root rank will send
 *                  to Rank 'R' the data under location buffer[gDElems[R]]. (significant only at root)
 *  \param[in] gElems number of elements this rank will send to root rank.
 *  \param[in] sElems number of elements this rank will receive from root rank.
 *  \param[in] resource Location to allocate temp buffers (CPU, GPU).
 *
 *  \return void.
 */
PERFFASPECT()
  void distributeV(std::vector<FPTypeValue *> &src,
                   std::vector<FPTypeValue *> &dest,
                   int *gNElems,
                   int *gDElems,
                   int *sNElems,
                   int *sDElems,
                   MPI_Datatype dType,
                   int gElems,
                   int sElems,
                   AMSResourceType resource)
  {
    FPTypeValue *temp_data;

    if (rId == root) {
      temp_data = ResourceManager::allocate<FPTypeValue>(globalLoad, resource);
    }

    for (int i = 0; i < src.size(); i++) {
      distribute(src[i],
                 dest[i],
                 temp_data,
                 gNElems,
                 gDElems,
                 sNElems,
                 sDElems,
                 dType,
                 gElems,
                 sElems);
    }

    if (rId == root) {
      ResourceManager::deallocate<FPTypeValue>(temp_data, resource);
    }

    return;
  }

public:
  static void *operator new(size_t) = delete;
  static void *operator new[](size_t) = delete;
  static void operator delete(void *) = delete;
  static void operator delete[](void *) = delete;

  /**
   * @brief instantiates a RAII object to be used for a load balance transaction.
   * @param[in] rId The rank id of the current rank in respect to the Comm communicator.
   * @param[in] worldSize The total number of ranks in respect to the Comm communicator.
   * @param[in] localLoad The number of elements this rank has to compute originally (before load balance).
   * @param[in] Comm The MPI communicator.
   * @param[in] numIn The number of input vectors to be balanced.
   * @param[in] numOut The number of output vectors to be balanced.
   * @param[in] resource The location of data allocations (CPU|GPU).
   */
  AMSLoadBalancer(int rId,
                  int worldSize,
                  int localLoad,
                  MPI_Comm comm,
                  int numIn,
                  int numOut,
                  AMSResourceType resource)
      : rId(rId),
        worldSize(worldSize),
        localLoad(localLoad),
        Comm(comm),
        globalLoad(-1),
        displs(nullptr),
        dataElements(nullptr),
        balancedElements(nullptr),
        balancedDispls(nullptr),
        resource(resource)
  {
    init(numIn, numOut, resource);
  }

  /** @brief deallocates all objects of this load balancing transcation */
  ~AMSLoadBalancer()
  {
    CINFO(LoadBalance, root==rId, "Total data %d Data per rank %d", globalLoad, balancedLoad);
    if (displs) ams::ResourceManager::deallocate(displs, AMSResourceType::HOST);
    if (dataElements)
      ams::ResourceManager::deallocate(dataElements, AMSResourceType::HOST);
    if (balancedElements)
      ams::ResourceManager::deallocate(balancedElements, AMSResourceType::HOST);
    if (balancedDispls)
      ams::ResourceManager::deallocate(balancedDispls, AMSResourceType::HOST);

    for (int i = 0; i < distOutputs.size(); i++)
      ams::ResourceManager::deallocate(distOutputs[i], resource);

    for (int i = 0; i < distInputs.size(); i++) {
      ams::ResourceManager::deallocate(distInputs[i], resource);
    }
  };

  /**
   * @brief Reverse load balance in respect to the output vectors.
   * @param[out] outputs The vector to store all the output values gathered from their compute (remote) ranks.
   * @param[in] resource The location of the data (CPU|GPU)
   */
PERFFASPECT()
  void gatherOutputs(std::vector<FPTypeValue *> &outputs,
                     AMSResourceType resource)
  {

    MPI_Datatype dType;

    if (isDouble<FPTypeValue>::default_value())
      dType = MPI_DOUBLE;
    else
      dType = MPI_FLOAT;

    distributeV(distOutputs,
                outputs,
                balancedElements,
                balancedDispls,  // Distribute the balanced load
                dataElements,    // Expecting to gather these
                displs,
                dType,
                balancedLoad,
                localLoad,
                resource);
    return;
  }

  /**
   * @brief Load balance the input vectors.
   * @param[out] inputs The vector to load balance across all compute (remote) ranks.
   * @param[in] resource The location of the data (CPU|GPU)
   */
PERFFASPECT()
  void scatterInputs(std::vector<FPTypeValue *> &inputs,
                     AMSResourceType resource)
  {
    MPI_Datatype dType;

    if (isDouble<FPTypeValue>::default_value())
      dType = MPI_DOUBLE;
    else
      dType = MPI_FLOAT;

    distributeV(inputs,
                distInputs,
                dataElements,
                displs,
                balancedElements,
                balancedDispls,  // Distribute the balanced load
                dType,
                localLoad,
                balancedLoad,
                resource);
  }

  /**
   * @brief Get access to load balanced inputs.
   * \returns   A pointer pointing to the balanced input elements.
   **/
  FPTypeValue **inputs() { return distInputs.data(); }

  /**
   * @brief Get access to load balanced outputs.
   * \returns A pointer pointing to the balanced output elements.
   **/
  FPTypeValue **outputs() { return distOutputs.data(); }

  /**
   * @brief Get the number of elements in the balanced vector.
   * \returns The number of elements in each balanced vector.
   **/
  int getBalancedSize() { return balancedLoad; }
};
}  // namespace ams

#endif  //REDIST_LOAD_HPP
