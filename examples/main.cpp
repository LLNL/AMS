/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifdef __AMS_ENABLE_ADIAK__
#include <adiak.hpp>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mfem.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <umpire/strategy/QuickPool.hpp>
#include <unordered_set>
#include <vector>

// clang-format off
#include "app/eos.hpp"
#include "app/eos_constant_on_host.hpp"
#include "app/eos_idealgas.hpp"
#include "app/utils_mfem.hpp"
#include "app/eos_ams.hpp"
// clang-format on

// this macro completely bypasses all AMS functionality
// this allows us to check how easy is it to test ams

#include "AMS.h"

void printMemory(std::unordered_set<std::string> &allocators)
{
  auto &rm = umpire::ResourceManager::getInstance();
  for (auto AN : allocators) {
    auto alloc = rm.getAllocator(AN);
    size_t wm = alloc.getHighWatermark();
    size_t cs = alloc.getCurrentSize();
    size_t as = alloc.getActualSize();
    std::cout << "Allocator '" << AN << "' High WaterMark:" << wm
              << " Current Size:" << cs << " Actual Size:" << as << "\n";
  }
}


void createUmpirePool(std::string parent_name, std::string pool_name)
{
  std::cout << "Pool Name " << pool_name << "Parent Allocation " << parent_name
            << "\n";
  auto &rm = umpire::ResourceManager::getInstance();
  auto alloc_resource = rm.makeAllocator<umpire::strategy::QuickPool, true>(
      pool_name, rm.getAllocator(parent_name));
}

std::unordered_set<std::string> createMemoryAllocators(
    std::string pool,
    std::string &physics_host_alloc,
    std::string &physics_device_alloc,
    std::string &physics_pinned_alloc,
    std::string &ams_host_alloc,
    std::string &ams_device_alloc,
    std::string &ams_pinned_alloc)
{
  std::unordered_set<std::string> allocator_names;
  if (pool == "default") {
    physics_host_alloc = ams_host_alloc = "HOST";
    allocator_names.insert(ams_host_alloc);
#ifdef __ENABLE_CUDA__
    physics_device_alloc = ams_device_alloc = "DEVICE";
    allocator_names.insert(ams_device_alloc);
    physics_pinned_alloc = ams_pinned_alloc = "PINNED";
    allocator_names.insert(ams_pinned_alloc);
#endif
  } else if (pool == "split") {
    physics_host_alloc = "phys-host";
    createUmpirePool("HOST", "phys-host");
    allocator_names.insert(physics_host_alloc);

    ams_host_alloc = "ams-host";
    createUmpirePool("HOST", ams_host_alloc);
    allocator_names.insert(ams_host_alloc);

#ifdef __ENABLE_CUDA__
    physics_device_alloc = "phys-device";
    createUmpirePool("DEVICE", physics_device_alloc);
    allocator_names.insert(physics_device_alloc);

    physics_pinned_alloc = "phys-pinned";
    createUmpirePool("PINNED", physics_pinned_alloc);
    allocator_names.insert(physics_pinned_alloc);

    ams_device_alloc = "ams-device";
    createUmpirePool("DEVICE", ams_device_alloc);
    allocator_names.insert(ams_device_alloc);

    ams_pinned_alloc = "ams-pinned";
    createUmpirePool("PINNED", ams_pinned_alloc);
    allocator_names.insert(ams_pinned_alloc);
#endif
  } else if (pool == "same") {
    physics_host_alloc = ams_host_alloc = "common-host";
    createUmpirePool("HOST", "common-host");
    allocator_names.insert(physics_host_alloc);
#ifdef __ENABLE_CUDA__
    physics_device_alloc = ams_device_alloc = "common-device";
    createUmpirePool("DEVICE", "common-device");
    allocator_names.insert(ams_device_alloc);
    physics_pinned_alloc = ams_pinned_alloc = "common-pinned";
    createUmpirePool("PINNED", "common-pinned");
    allocator_names.insert(ams_pinned_alloc);
#endif
  } else {
    std::cout << "Stategy is " << pool << "\n";
    throw std::runtime_error("Pool strategy does not exist\n");
  }
  return std::move(allocator_names);
}

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

// Runs AMS using typed value.
// TODO: Gather arguments in a struct.
// TODO: Move parsing of arguments in main.
template <typename TypeValue>
int run(const char *device_name,
        const char *db_type,
        const char *uq_policy_opt,
        AMSDType precision,
        int seed,
        int rId,
        int imbalance,
        int wS,
        double avg,
        double stdDev,
        double threshold,
        const char *pool,
        int num_mats,
        int num_elems,
        int num_qpts,
        double empty_element_ratio,
        bool verbose,
        const char *eos_name,
        int stop_cycle,
        bool pack_sparse_mats,
        const char *hdcache_path,
        const char *model_path,
        const char *db_config,
        bool lbalance,
        int k_nearest)
{
  // -------------------------------------------------------------------------
  // setup
  // -------------------------------------------------------------------------
  CALIPER(cali::ConfigManager mgr;)
  CALIPER(mgr.start();)
  CALIPER(CALI_MARK_BEGIN("Setup");)

  const bool use_device = std::strcmp(device_name, "cpu") != 0;
  AMSDBType dbType = AMSDBType::None;
  if (std::strcmp(db_type, "csv") == 0) {
    dbType = AMSDBType::CSV;
  } else if (std::strcmp(db_type, "hdf5") == 0) {
    dbType = AMSDBType::HDF5;
  } else if (std::strcmp(db_type, "rmq") == 0) {
    dbType = AMSDBType::RMQ;
  }

  AMSUQPolicy uq_policy;

  if (strcmp(uq_policy_opt, "faiss-max") == 0)
    uq_policy = AMSUQPolicy::FAISS_Max;
  else if (strcmp(uq_policy_opt, "faiss-mean") == 0)
    uq_policy = AMSUQPolicy::FAISS_Mean;
  else if (strcmp(uq_policy_opt, "deltauq-max") == 0)
    uq_policy = AMSUQPolicy::DeltaUQ_Max;
  else if (strcmp(uq_policy_opt, "deltauq-mean") == 0)
    uq_policy = AMSUQPolicy::DeltaUQ_Mean;
  else if (strcmp(uq_policy_opt, "random") == 0)
    uq_policy = AMSUQPolicy::Random;
  else
    throw std::runtime_error("Invalid UQ policy");

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
  // setup mfem memory manager
  // -------------------------------------------------------------------------
  // hardcoded names!

  std::string physics_host_alloc;
  std::string physics_device_alloc;
  std::string physics_pinned_alloc;

  std::string ams_host_alloc;
  std::string ams_device_alloc;
  std::string ams_pinned_alloc;

  auto allocator_names = createMemoryAllocators(std::string(pool),
                                                physics_host_alloc,
                                                physics_device_alloc,
                                                physics_pinned_alloc,
                                                ams_host_alloc,
                                                ams_device_alloc,
                                                ams_pinned_alloc);


  mfem::MemoryManager::SetUmpireHostAllocatorName(physics_host_alloc.c_str());
  if (use_device) {
    mfem::MemoryManager::SetUmpireDeviceAllocatorName(
        physics_device_alloc.c_str());
  }


  // When we are not allocating from parent/root umpire allocator
  // we need to inform AMS about the pool allocators.
  if (strcmp(pool, "default") != 0) {
    AMSSetAllocator(AMSResourceType::HOST, ams_host_alloc.c_str());

    if (use_device) {
      AMSSetAllocator(AMSResourceType::DEVICE, ams_device_alloc.c_str());
      AMSSetAllocator(AMSResourceType::PINNED, ams_pinned_alloc.c_str());
    }
  }

  mfem::Device::SetMemoryTypes(mfem::MemoryType::HOST_UMPIRE,
                               mfem::MemoryType::DEVICE_UMPIRE);

  mfem::Device device(device_name);
  std::cout << std::endl;
  device.Print();
  std::cout << std::endl;


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

  // ---------------------------------------------------------------------
  // setup AMS options
  // ---------------------------------------------------------------------
#ifdef USE_AMS
  constexpr bool use_ams = true;
  const char *uq_path = nullptr;
  const char *surrogate_path = nullptr;
  const char *db_path = nullptr;

#ifdef __ENABLE_FAISS__
  uq_path = (strlen(hdcache_path) > 0) ? hdcache_path : nullptr;
#endif

  std::cout << "surrogate Path is : " << model_path << "\n";
#ifdef __ENABLE_TORCH__
  surrogate_path = (strlen(model_path) > 0) ? model_path : nullptr;
#endif

  db_path = (strlen(db_config) > 0) ? db_config : nullptr;

  AMSResourceType ams_device = AMSResourceType::HOST;
  if (use_device) ams_device = AMSResourceType::DEVICE;
  AMSExecPolicy ams_loadBalance = AMSExecPolicy::UBALANCED;
  if (lbalance) ams_loadBalance = AMSExecPolicy::BALANCED;
#else
  constexpr bool use_ams = false;
#endif

  // ---------------------------------------------------------------------
  // setup EOS models
  // ---------------------------------------------------------------------
  std::vector<EOS<TypeValue> *> eoses(num_mats, nullptr);
  for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
    EOS<TypeValue> *base;
    if (eos_name == std::string("ideal_gas")) {
      base = new IdealGas<TypeValue>(1.6, 1.4);
    } else if (eos_name == std::string("constant_host")) {
      base = new ConstantEOSOnHost<TypeValue>(physics_host_alloc.c_str(), 1.0);
    } else {
      std::cerr << "unknown eos `" << eos_name << "'" << std::endl;
      return 1;
    }
#ifdef USE_AMS
    if (use_ams) {
      eoses[mat_idx] = new AMSEOS<TypeValue>(base,
                                             dbType,
                                             precision,
                                             ams_loadBalance,
                                             ams_device,
                                             uq_policy,
                                             k_nearest,
                                             rId,
                                             wS,
                                             threshold,
                                             surrogate_path,
                                             uq_path,
                                             db_path);

    } else
#endif
    {
      eoses[mat_idx] = base;
    }
  }

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
          eoses[mat_idx]->Eval(num_elems_for_mat * num_qpts,
                               &d_dense_density(0, 0),
                               &d_dense_energy(0, 0),
                               &d_dense_pressure(0, 0),
                               &d_dense_soundspeed2(0, 0),
                               &d_dense_bulkmod(0, 0),
                               &d_dense_temperature(0, 0));

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
          std::cout << " material " << mat_idx << ": using dense packing for "
                    << num_elems << " elems" << std::endl;

          eoses[mat_idx]->Eval(num_elems * num_qpts,
                               &d_density(0, 0, mat_idx),
                               &d_energy(0, 0, mat_idx),
                               &d_pressure(0, 0, mat_idx),
                               &d_soundspeed2(0, 0, mat_idx),
                               &d_bulkmod(0, 0, mat_idx),
                               &d_temperature(0, 0, mat_idx));
        }
      }
    }
    CALIPER(CALI_MARK_END("Cycle");)
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    printMemory(allocator_names);
  }

  // TODO: Add smart-pointers
  for (int mat_idx = 0; mat_idx < num_mats; ++mat_idx) {
    delete eoses[mat_idx];
    eoses[mat_idx] = nullptr;
  }

  CALIPER(CALI_MARK_END("TimeStepLoop"););

  return 0;
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

  const char *precision_opt = "double";
  AMSDType precision = AMSDType::Double;

  const char *uq_policy_opt = "";
  int k_nearest = 5;

  int seed = 0;
  double empty_element_ratio = -1;

  int stop_cycle = 1;

  int num_mats = 5;
  int num_elems = 10000;
  int num_qpts = 64;
  bool pack_sparse_mats = true;

  bool imbalance = false;
  bool lbalance = false;
  double threshold = 0.5;
  double avg = 0.5;
  double stdDev = 0.2;
  bool reqDB = false;
  const char *pool = "default";

#ifdef __ENABLE_DB__
  reqDB = true;
#endif

  bool verbose = false;

#ifdef __AMS_ENABLE_ADIAK__
  // add adiak init here
  adiak::init(NULL);

  // replace with adiak::collect_all(); once adiak v0.4.0
  adiak::uid();
  adiak::launchdate();
  adiak::launchday();
  adiak::executable();
  adiak::executablepath();
  adiak::workdir();
  adiak::libraries();
  adiak::cmdline();
  adiak::hostname();
  adiak::clustername();
  adiak::walltime();
  adiak::systime();
  adiak::cputime();
  adiak::jobsize();
  adiak::hostlist();
  adiak::numhosts();
  adiak::value("compiler", std::string("@RAJAPERF_COMPILER@"));
#endif

  // -------------------------------------------------------------------------
  // setup command line parser
  // -------------------------------------------------------------------------
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&device_name, "-d", "--device", "Device config string");

  // set precision
  args.AddOption(&precision_opt,
                 "-pr",
                 "--precision",
                 "Set precision (single or double)");

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
                 "Path to directory where applications will store their data "
                 "(or Path to JSON configuration if RabbitMQ is chosen)",
                 reqDB);

  args.AddOption(&db_type,
                 "-dt",
                 "--dbtype",
                 "Configuration option of the different DB types:\n"
                 "\t 'csv' Use csv as back end\n"
                 "\t 'hdf5': use hdf5 as a back end\n"
                 "\t 'rmq': use RabbitMQ as a back end\n");

  args.AddOption(&k_nearest,
                 "-knn",
                 "--k-nearest-neighbors",
                 "Number of closest neightbors we should look at");

  args.AddOption(&uq_policy_opt,
                 "-uq",
                 "--uqtype",
                 "Types of UQ to select from: \n"
                 "\t 'faiss-mean' Uncertainty is computed in comparison "
                 "against the "
                 "mean distance of k-nearest neighbors\n"
                 "\t 'faiss-max': Uncertainty is computed in comparison with "
                 "the "
                 "k'st cluster \n"
                 "\t 'deltauq-mean': Uncertainty through DUQ using mean\n"
                 "\t 'deltauq-max': Uncertainty through DUQ using max\n"
                 "\t 'random': Uncertainty throug a random model\n");

  args.AddOption(
      &verbose, "-v", "--verbose", "-qu", "--quiet", "Print extra stuff");

  args.AddOption(&pool,
                 "-ptype",
                 "--pool-type",
                 "How to assign memory pools to AMSlib:\n"
                 "\t 'default' Use the default Umpire pool\n"
                 "\t 'split' provide a separate pool to AMSlib\n"
                 "\t 'same': assign the same with physics to AMS\n");

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

  if (strcmp(precision_opt, "single") == 0)
    precision = AMSDType::Single;
  else if (strcmp(precision_opt, "double") == 0)
    precision = AMSDType::Double;
  else {
    std::cerr << "Invalid precision " << precision_opt << "\n";
    return -1;
  }

  int ret = 0;
  if (precision == AMSDType::Single)
    ret = run<float>(device_name,
                     db_type,
                     uq_policy_opt,
                     precision,
                     seed,
                     rId,
                     imbalance,
                     wS,
                     avg,
                     stdDev,
                     threshold,
                     pool,
                     num_mats,
                     num_elems,
                     num_qpts,
                     empty_element_ratio,
                     verbose,
                     eos_name,
                     stop_cycle,
                     pack_sparse_mats,
                     hdcache_path,
                     model_path,
                     db_config,
                     lbalance,
                     k_nearest);
  else if (precision == AMSDType::Double)
    ret = run<double>(device_name,
                      db_type,
                      uq_policy_opt,
                      precision,
                      seed,
                      rId,
                      imbalance,
                      wS,
                      avg,
                      stdDev,
                      threshold,
                      pool,
                      num_mats,
                      num_elems,
                      num_qpts,
                      empty_element_ratio,
                      verbose,
                      eos_name,
                      stop_cycle,
                      pack_sparse_mats,
                      hdcache_path,
                      model_path,
                      db_config,
                      lbalance,
                      k_nearest);
  else {
    std::cerr << "Invalid precision " << precision_opt << "\n";
    return -1;
  }

  // ---------------------------------------------------------------------------
#ifdef __AMS_ENABLE_ADIAK__
  // adiak finalize
  adiak::fini();
#endif

  MPI_CALL(MPI_Finalize());
  return ret;
}
