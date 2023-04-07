// Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mfem.hpp>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "app/eos.hpp"
#include "app/eos_constant_on_host.hpp"
#include "app/eos_idealgas.hpp"
#include "app/utils_mfem.hpp"

// this macro completely bypasses all AMS functionality
// this allows us to check how easy is it to test ams
#define USE_AMS

#ifdef USE_AMS
#include "AMS.h"
#endif

using TypeValue = double;
using mfem::ForallWrap;

int computeNumElements(int globalNumElements, int id, int numRanks)
{
  int lElements = (globalNumElements + numRanks - 1) / numRanks;
  lElements = std::min(lElements, globalNumElements - id * lElements);
  return lElements;
}

const std::unordered_set<std::string> eos_options{"ideal_gas", "constant_host"};

double unitrand() { return (double)rand() / RAND_MAX; }

// TODO: we could to this on the device but need something more than `rand'
template <typename T>
void random_init(mfem::Array<T> &arr)
{
  T *h_arr = arr.HostWrite();
  for (int i = 0; i < arr.Size(); ++i) {
    h_arr[i] = unitrand();
  }
}

PERFFASPECT()
int main(int argc, char **argv)
{
  // -------------------------------------------------------------------------
  // declare runtime options and default values
  // -------------------------------------------------------------------------

  // Number of ranks in this run
  int wS = 1;
  // My Local Id
  int rId = 0;
  // Level of Threading provided by MPI
  int provided = 0;
  MPI_CALL(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided));
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &wS));
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rId));
  // FIXME: Create a logger class to write
  // depending on rank id and severity.
  if (rId != 0) {
    std::cout.setstate(std::ios::failbit);
  }

  const char *device_name = "cpu";
  const char *eos_name = "ideal_gas";
  const char *model_path = "";
  const char *hdcache_path = "";
  const char *db_config = "";
  const char *db_type = "";
  const char *rmq_config   = "";

  const char *uq_policy_opt = "mean";
  int k_nearest = 5;

  int seed = 0;
  TypeValue empty_element_ratio = -1;

  int stop_cycle = 1;

  int num_mats = 5;
  int num_elems = 10000;
  int num_qpts = 64;
  bool pack_sparse_mats = true;

  bool imbalance = false;
  bool lbalance = false;
  TypeValue threshold = 0.5;
  TypeValue avg = 0.5;
  TypeValue stdDev = 0.2;
  bool reqDB = false;

#ifdef __ENABLE_DB__
  reqDB = true;
#endif

  bool verbose = false;

  // -------------------------------------------------------------------------
  // setup command line parser
  // -------------------------------------------------------------------------
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&device_name, "-d", "--device", "Device config string");

  // surrogate model
  args.AddOption(&model_path, "-S", "--surrogate", "Path to surrogate model");
  args.AddOption(&hdcache_path, "-H", "--hdcache", "Path to hdcache index");

  // eos model and length of simulation
  args.AddOption(&eos_name, "-z", "--eos", "EOS model type");
  args.AddOption(&stop_cycle, "-c", "--stop-cycle", "Stop cycle");

  // data parameters
  args.AddOption(&num_mats, "-m", "--num-mats", "Number of materials");
  args.AddOption(&num_elems, "-e", "--num-elems", "Number of elements");
  args.AddOption(&num_qpts,
                 "-q",
                 "--num-qpts",
                 "Number of quadrature points per element");
  args.AddOption(&empty_element_ratio,
                 "-r",
                 "--empty-element-ratio",
                 "Fraction of elements that are empty "
                 "for each material. If -1 use a random value for each. ");

  // random speed and packing
  args.AddOption(&seed, "-s", "--seed", "Seed for rand");
  args.AddOption(&pack_sparse_mats,
                 "-p",
                 "--pack-sparse",
                 "-np",
                 "--do-not-pack-sparse",
                 "pack sparse material data before evals (cpu only)");

  args.AddOption(&imbalance,
                 "-i",
                 "--with-imbalance",
                 "-ni",
                 "--without-imbalance",
                 "Create artificial load imbalance across ranks");

  args.AddOption(&avg,
                 "-avg",
                 "--average",
                 "Average value of random number generator of imbalance "
                 "threshold");

  args.AddOption(&stdDev,
                 "-std",
                 "--stdev",
                 "Standard deviation of random number generator of imbalance ");

  args.AddOption(&lbalance,
                 "-lb",
                 "--with-load-balance",
                 "-nlb",
                 "--without-load-balance",
                 "Enable Load balance module in AMS");

  args.AddOption(&threshold,
                 "-t",
                 "--threshold",
                 "Threshold value used to control selection of surrogate "
                 "vs physics execution");

  args.AddOption(&db_config,
                 "-db",
                 "--dbconfig",
                 "Path to directory where applications will store their data",
                 reqDB);

  args.AddOption(&db_type,
                 "-dt",
                 "--dbtype",
                 "Configuration option of the different DB types:\n"
                 "\t 'csv' Use csv as back end\n"
                 "\t 'hdf5': use hdf5 as a back end\n");

  args.AddOption(&k_nearest, "-knn", "--k-nearest-neighbors", "Number of closest neightbors we should look at");

  args.AddOption(&uq_policy_opt,
                 "-uq",
                 "--uqtype",
                 "Types of UQ to select from: \n"
                 "\t 'mean' Uncertainty is computed in comparison against the mean distance of k-nearest neighbors\n"
                 "\t 'max': Uncertainty is computed in comparison with the k'st cluster \n"
                 "\t 'deltauq': Uncertainty through DUQ (not supported)\n");


  args.AddOption(
      &verbose, "-v", "--verbose", "-qu", "--quiet", "Print extra stuff");
  args.AddOption(&rmq_config, "-rmq", "--rabbitmq", "Path to Data Broker configuration (e.g., RabbitMQ)");

  // -------------------------------------------------------------------------
  // parse arguments
  // -------------------------------------------------------------------------
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(std::cout);
    return -1;
  }

  if (rId == 0) {
    args.PrintOptions(std::cout);
    std::cout << std::endl;
  }

  // -------------------------------------------------------------------------
  // additional argument validation
  // -------------------------------------------------------------------------
  if (eos_options.find(eos_name) != eos_options.end()) {
    std::cout << "Using eos = '" << eos_name << "'" << std::endl;
  } else {
    std::cerr << "Unsupported eos '" << eos_name << "'" << std::endl
              << "Available options: " << std::endl;

    for (const auto &s : eos_options) {
      std::cerr << " - " << s << std::endl;
    }
    return -1;
  }

  // small validation
  assert(stop_cycle > 0);
  assert(num_mats > 0);
  assert(num_elems > 0);
  assert(num_qpts > 0);

#ifdef __ENABLE_TORCH__
  if (model_path == nullptr) {
    std::cerr << "Compiled with Py-Torch enabled. It is mandatory to provide a "
                 "surrogate model"
              << std::endl;
    exit(-1);
  }
#endif
  assert((empty_element_ratio >= 0 && empty_element_ratio < 1) ||
         empty_element_ratio == -1);

  std::cout << "Total computed elements across all ranks: " << wS * num_elems
            << "(Weak Scaling)\n";

  // -------------------------------------------------------------------------
  // setup
  // -------------------------------------------------------------------------
  CALIPER(cali::ConfigManager mgr;)
  CALIPER(mgr.start();)
  CALIPER(CALI_MARK_BEGIN("Setup");)

  const bool use_device = std::strcmp(device_name, "cpu") != 0;
  AMSDBType dbType =
      (std::strcmp(db_type, "csv") == 0) ? AMSDBType::CSV : AMSDBType::None;
  if ( dbType != AMSDBType::CSV )
    dbType = ((std::strcmp(db_type, "hdf5") == 0)) ? AMSDBType::HDF5 : AMSDBType::None;

  AMSUQPolicy uq_policy =
      (std::strcmp(uq_policy_opt, "max") == 0) ? AMSUQPolicy::FAISSMax: AMSUQPolicy::FAISSMean;

  if ( uq_policy != AMSUQPolicy::FAISSMax )
    uq_policy = ((std::strcmp(uq_policy_opt, "deltauq") == 0))
      ? AMSUQPolicy::DeltaUQ : AMSUQPolicy::FAISSMean;


  // set up a randomization seed
  srand(seed + rId);

  if (imbalance) {
    // I need to select a threshold for my specific rank
    // Out of this distribution
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(avg, stdDev);
    threshold = distribution(generator);
#ifdef __ENABLE_MPI__
    if (wS > 1) {
      if (rId == 0) {
        for (int i = 1; i < wS; i++) {
          double tmp = distribution(generator);
          MPI_Send(&tmp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
      } else {
        double tmp;
        MPI_Recv(&tmp, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        threshold = static_cast<TypeValue>(tmp);
      }
    }
#endif
  }

  std::cerr << "Rank:" << rId << " Threshold " << threshold << "\n";

  // -------------------------------------------------------------------------
  // setup data allocators
  // -------------------------------------------------------------------------
  AMSSetupAllocator(AMSResourceType::HOST);
  if (use_device) {
    AMSSetupAllocator(AMSResourceType::DEVICE);
    AMSSetupAllocator(AMSResourceType::PINNED);
    AMSSetDefaultAllocator(AMSResourceType::DEVICE);
  } else {
    AMSSetDefaultAllocator(AMSResourceType::HOST);
  }

  // -------------------------------------------------------------------------
  // setup mfem memory manager
  // -------------------------------------------------------------------------
  // hardcoded names!
  const std::string &alloc_name_host(
      AMSGetAllocatorName(AMSResourceType::HOST));
  const std::string &alloc_name_device(
      AMSGetAllocatorName(AMSResourceType::DEVICE));

  mfem::MemoryManager::SetUmpireHostAllocatorName(alloc_name_host.c_str());
  if (use_device) {
    mfem::MemoryManager::SetUmpireDeviceAllocatorName(
        alloc_name_device.c_str());
  }

  mfem::Device::SetMemoryTypes(mfem::MemoryType::HOST_UMPIRE,
                               mfem::MemoryType::DEVICE_UMPIRE);

  mfem::Device device(device_name);
  std::cout << std::endl;
  device.Print();
  std::cout << std::endl;

  //AMSResourceInfo();

  // -------------------------------------------------------------------------
  // setup indicators
  //  to represent which combinations of materials and elements exist
  // -------------------------------------------------------------------------
  std::cout << "Setting up indicators" << std::endl;
  bool indicators[num_mats * num_elems];

  for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
    // min ratio if empty_element_ratio is -1
    const TypeValue min_ratio = 0.2;
    const TypeValue ratio = empty_element_ratio == -1
                                ? unitrand() * (1 - min_ratio) + min_ratio
                                : 1 - empty_element_ratio;
    const int num_nonzero_elems = ratio * num_elems;
    std::cout << "  using " << num_nonzero_elems << "/" << num_elems
              << " for material " << mat_idx << std::endl;

    int nz = 0;
    for (int elem_idx = 0; elem_idx < num_elems; ++elem_idx) {
      const int me = elem_idx + mat_idx * num_elems;
      indicators[me] = false;

      if (nz < num_nonzero_elems) {
        if (((num_nonzero_elems - nz) == (num_elems - elem_idx)) ||
            unitrand() <= ratio) {
          indicators[me] = true;
          nz++;
        }
      }
    }
  }

  if (verbose) {
    print_tensor_array("indicators", indicators, {1, num_mats, num_elems});
  }

  // -------------------------------------------------------------------------
  // setup AMS
  // -------------------------------------------------------------------------

#ifdef USE_AMS
  // We need a AMSExecutor for each material.
  // This implicitly implies that we are going to
  // have a differnt ml model for each type of material.
  // If we have a single model for all materials then we
  // need to make a single model here.
  AMSExecutor *workflow = new AMSExecutor[num_mats]();
#endif

  // ---------------------------------------------------------------------
  // setup EOS models
  // ---------------------------------------------------------------------
  std::vector<EOS *> eoses(num_mats, nullptr);
  for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
    if (eos_name == std::string("ideal_gas")) {
      eoses[mat_idx] = new IdealGas(1.6, 1.4);
    } else if (eos_name == std::string("constant_host")) {
      eoses[mat_idx] = new ConstantEOSOnHost(alloc_name_host.c_str(), 1.0);
    } else {
      std::cerr << "unknown eos `" << eos_name << "'" << std::endl;
      return 1;
    }
  }

  // ---------------------------------------------------------------------
  // setup AMS workflow (surrogate and cache)
  // ---------------------------------------------------------------------
#ifdef USE_AMS
  const char *uq_path = nullptr;
  const char *surrogate_path = nullptr;
  const char *db_path = nullptr;
  const char *rmq_path       = nullptr;

#ifdef __ENABLE_FAISS__
  uq_path = (strlen(hdcache_path) > 0) ? hdcache_path : nullptr;
#endif

  std::cout << "surrogate Path is : " << model_path << "\n";
#ifdef __ENABLE_TORCH__
  surrogate_path = (strlen(model_path) > 0) ? model_path : nullptr;
#endif

  AMSBrokerType ams_broker_type = AMSBrokerType::NoBroker;
#ifdef __ENABLE_RMQ__
  std::cout << "RabbitMQ configuration path : " << rmq_config << "\n";
  rmq_path = (strlen(rmq_config) > 0) ? rmq_config : nullptr;
  ams_broker_type = AMSBrokerType::RMQ;
#endif

#ifdef __ENABLE_DB__
  db_path = "miniapp.txt";
#ifdef __ENABLE_REDIS__
  /*
  * A JSON that contains all Redis info (port, host, password, SSL certificate path)
  * See README to generate the certificate (.crt file).
  * {
  *      "database-password": "mypassword",
  *      "service-port": 32273,
  *      "host": "cz-username-testredis1.apps.czapps.llnl.gov",
  *      "cert": "redis_certificate.crt"
  * }
  */
  //db_path = "test-config-redis.json";

  db_path = (strlen(db_config) > 0) ? db_config : nullptr;

  AMSResourceType ams_device = AMSResourceType::HOST;
  if (use_device) ams_device = AMSResourceType::DEVICE;
  AMSExecPolicy ams_loadBalance = AMSExecPolicy::UBALANCED;
  if ( lbalance ) ams_loadBalance = AMSExecPolicy::BALANCED;

  AMSConfig amsConf = {ams_loadBalance,
                       AMSDType::Double,
                       ams_device,
                       dbType,
                       callBack,
                       (char *)surrogate_path,
                       (char *)uq_path,
                       (char *)db_path,
                       threshold,
                       uq_policy,
                       k_nearest,
                       rId,
                       wS };
  AMSExecutor wf = AMSCreateExecutor(amsConf);

  for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
    workflow[mat_idx] = wf;
  }
#endif

  // ---------------------------------------------------------------------
  // setup sparse element info
  // ---------------------------------------------------------------------
  // need a way to store indices of active elements for each material
  // will use a linearized array for this
  // first num_mat elements will store the total count of all elements thus far
  //                                                    (actually, + num_mats)
  // after that, store the indices of those elements in order
  mfem::Array<int> sparse_elem_indices;
  sparse_elem_indices.SetSize(num_mats);
  sparse_elem_indices.Reserve(num_mats * num_elems);

  for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
    for (int elem_idx = 0; elem_idx < num_elems; ++elem_idx) {
      if (indicators[elem_idx + num_elems * mat_idx]) {
        sparse_elem_indices.Append(elem_idx);
      }
    }
    sparse_elem_indices[mat_idx] = sparse_elem_indices.Size();
  }
  CALIPER(CALI_MARK_END("Setup");)

  // -------------------------------------------------------------------------
  // run through the cycles (time-steps)
  // -------------------------------------------------------------------------
  CALIPER(CALI_MARK_BEGIN("TimeStepLoop");)
  // inputs
  mfem::Array<TypeValue> density(num_mats * num_elems * num_qpts);
  mfem::Array<TypeValue> energy(num_mats * num_elems * num_qpts);

  // outputs
  mfem::Array<TypeValue> pressure(num_qpts * num_elems * num_mats);
  mfem::Array<TypeValue> soundspeed2(num_qpts * num_elems * num_mats);
  mfem::Array<TypeValue> bulkmod(num_qpts * num_elems * num_mats);
  mfem::Array<TypeValue> temperature(num_qpts * num_elems * num_mats);

  // spasity, currently static after setup
  const int *h_sparse_elem_indices = sparse_elem_indices.HostRead();
  const int *d_sparse_elem_indices = sparse_elem_indices.Read();

  for (int c = 0; c <= stop_cycle; ++c) {
    std::cout << std::endl << "--> cycle: " << c << std::endl;

    CALIPER(CALI_MARK_BEGIN("Randomize Inputs");)
    random_init(density);
    random_init(energy);
    CALIPER(CALI_MARK_END("Randomize Inputs");)

    if (verbose) {
      print_tensor_array("density",
                         density.HostRead(),
                         {num_mats, num_elems, num_qpts});
      print_tensor_array("energy",
                         energy.HostRead(),
                         {num_mats, num_elems, num_qpts});
    }

    CALIPER(CALI_MARK_BEGIN("Cycle");)
    {
      // move/allocate data on the device.
      // if the data is already on the device this is basically a noop

      const auto d_density =
          mfemReshapeArray3(density, Read, num_qpts, num_elems, num_mats);
      const auto d_energy =
          mfemReshapeArray3(energy, Read, num_qpts, num_elems, num_mats);

      auto d_pressure =
          mfemReshapeArray3(pressure, Write, num_qpts, num_elems, num_mats);
      auto d_soundspeed2 =
          mfemReshapeArray3(soundspeed2, Write, num_qpts, num_elems, num_mats);
      auto d_bulkmod =
          mfemReshapeArray3(bulkmod, Write, num_qpts, num_elems, num_mats);
      auto d_temperature =
          mfemReshapeArray3(temperature, Write, num_qpts, num_elems, num_mats);

      // ---------------------------------------------------------------------
      // for each material
      for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
        const int offset_curr =
            mat_idx == 0 ? num_mats : h_sparse_elem_indices[mat_idx - 1];
        const int offset_next = h_sparse_elem_indices[mat_idx];

        const int num_elems_for_mat = offset_next - offset_curr;
        if (num_elems_for_mat == 0) {
          continue;
        }

        // -----------------------------------------------------------------
        // NOTE: we've found it's faster to do sparse lookups on GPUs but on
        // CPUs the dense packing->looked->unpacking is better if we're using
        // expensive eoses. in the future we may just use dense representations
        // everywhere but for now we use sparse ones.
        if (pack_sparse_mats && num_elems_for_mat < num_elems) {
          std::cout << " material " << mat_idx << ": using sparse packing for "
                    << num_elems_for_mat << " elems" << std::endl;

          // -------------------------------------------------------------
          // TODO: I think Tom mentiond we can allocate these outside the loop
          // check again
          // NOTE: In blast we allocate these in a temporary memory pool
          mfem::Array<TypeValue> dense_density(num_elems_for_mat * num_qpts);
          mfem::Array<TypeValue> dense_energy(num_elems_for_mat * num_qpts);
          mfem::Array<TypeValue> dense_pressure(num_elems_for_mat * num_qpts);
          mfem::Array<TypeValue> dense_soundspeed2(num_elems_for_mat *
                                                   num_qpts);
          mfem::Array<TypeValue> dense_bulkmod(num_elems_for_mat * num_qpts);
          mfem::Array<TypeValue> dense_temperature(num_elems_for_mat *
                                                   num_qpts);

          // these are device tensors!
          auto d_dense_density = mfemReshapeArray2(dense_density,
                                                   Write,
                                                   num_qpts,
                                                   num_elems_for_mat);
          auto d_dense_energy = mfemReshapeArray2(dense_energy,
                                                  Write,
                                                  num_qpts,
                                                  num_elems_for_mat);
          auto d_dense_pressure = mfemReshapeArray2(dense_pressure,
                                                    Write,
                                                    num_qpts,
                                                    num_elems_for_mat);
          auto d_dense_soundspeed2 = mfemReshapeArray2(dense_soundspeed2,
                                                       Write,
                                                       num_qpts,
                                                       num_elems_for_mat);
          auto d_dense_bulkmod = mfemReshapeArray2(dense_bulkmod,
                                                   Write,
                                                   num_qpts,
                                                   num_elems_for_mat);
          auto d_dense_temperature = mfemReshapeArray2(dense_temperature,
                                                       Write,
                                                       num_qpts,
                                                       num_elems_for_mat);

          // -------------------------------------------------------------
          // sparse -> dense
          CALIPER(CALI_MARK_BEGIN("SPARSE_TO_DENSE");)
          pack_ij(mat_idx,
                  num_qpts,
                  num_elems_for_mat,
                  offset_curr,
                  d_sparse_elem_indices,
                  d_density,
                  d_dense_density,
                  d_energy,
                  d_dense_energy);
          CALIPER(CALI_MARK_END("SPARSE_TO_DENSE");)
          // -------------------------------------------------------------
          std::vector<const double *> inputs = {&d_dense_density(0, 0),
                                                &d_dense_energy(0, 0)};
          std::vector<double *> outputs = {&d_dense_pressure(0, 0),
                                           &d_dense_soundspeed2(0, 0),
                                           &d_dense_bulkmod(0, 0),
                                           &d_dense_temperature(0, 0)};

#ifdef USE_AMS
#ifdef __ENABLE_MPI__
          AMSDistributedExecute(workflow[mat_idx],
                                MPI_COMM_WORLD,
                                static_cast<void *>(eoses[mat_idx]),
                                num_elems_for_mat * num_qpts,
                                reinterpret_cast<const void **>(inputs.data()),
                                reinterpret_cast<void **>(outputs.data()),
                                inputs.size(),
                                outputs.size());
#else
          AMSExecute(workflow[mat_idx],
                     static_cast<void *>(eoses[mat_idx]),
                     num_elems_for_mat * num_qpts,
                     reinterpret_cast<const void **>(inputs.data()),
                     reinterpret_cast<void **>(outputs.data()),
                     inputs.size(),
                     outputs.size());
#endif
#else
          eoses[mat_idx]->Eval(num_elems_for_mat * num_qpts,
                               &d_dense_density(0, 0),
                               &d_dense_energy(0, 0),
                               &d_dense_pressure(0, 0),
                               &d_dense_soundspeed2(0, 0),
                               &d_dense_bulkmod(0, 0),
                               &d_dense_temperature(0, 0));
#endif
          // -------------------------------------------------------------
          // dense -> sparse
          CALIPER(CALI_MARK_BEGIN("DENSE_TO_SPARSE");)
          unpack_ij(mat_idx,
                    num_qpts,
                    num_elems_for_mat,
                    offset_curr,
                    d_sparse_elem_indices,
                    d_dense_pressure,
                    d_pressure,
                    d_dense_soundspeed2,
                    d_soundspeed2,
                    d_dense_bulkmod,
                    d_bulkmod,
                    d_dense_temperature,
                    d_temperature);
          CALIPER(CALI_MARK_END("DENSE_TO_SPARSE");)
          // -------------------------------------------------------------
        } else {
#ifdef USE_AMS
          std::cout << " material " << mat_idx << ": using dense packing for "
                    << num_elems << " elems" << std::endl;

          std::vector<const double *> inputs = {&d_density(0, 0, mat_idx),
                                                &d_energy(0, 0, mat_idx)};
          std::vector<double *> outputs = {&d_pressure(0, 0, mat_idx),
                                           &d_soundspeed2(0, 0, mat_idx),
                                           &d_bulkmod(0, 0, mat_idx),
                                           &d_temperature(0, 0, mat_idx)};
#ifdef __ENABLE_MPI__
          AMSDistributedExecute(workflow[mat_idx],
                                MPI_COMM_WORLD,
                                static_cast<void *>(eoses[mat_idx]),
                                num_elems_for_mat * num_qpts,
                                reinterpret_cast<const void **>(inputs.data()),
                                reinterpret_cast<void **>(outputs.data()),
                                inputs.size(),
                                outputs.size());
#else
          AMSExecute(workflow[mat_idx],
                     static_cast<void *>(eoses[mat_idx]),
                     num_elems * num_qpts,
                     reinterpret_cast<const void **>(inputs.data()),
                     reinterpret_cast<void **>(outputs.data()),
                     inputs.size(),
                     outputs.size());
#endif
#else
          eoses[mat_idx]->Eval(num_elems * num_qpts,
                               &d_density(0, 0, mat_idx),
                               &d_energy(0, 0, mat_idx),
                               &d_pressure(0, 0, mat_idx),
                               &d_soundspeed2(0, 0, mat_idx),
                               &d_bulkmod(0, 0, mat_idx),
                               &d_temperature(0, 0, mat_idx));
#endif
        }
      }
    }
    CALIPER(CALI_MARK_END("Cycle");)
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
  }
  CALIPER(CALI_MARK_END("TimeStepLoop"););
  MPI_CALL(MPI_Finalize());
  return 0;
}
