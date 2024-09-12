#ifdef __AMS_ENABLE_MPI__
#include <mpi.h>
#endif
#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>

#include <mfem.hpp>

#include "AMS.h"
// #include <ml/uq.hpp>
// #include <wf/basedb.hpp>
// #include <wf/resource_manager.hpp>
// #include "wf/debug.h"

// #include <nlohmann/json.hpp>



void createUmpirePool(std::string parent_name, std::string pool_name)
{
  auto &rm = umpire::ResourceManager::getInstance();
  auto alloc_resource = rm.makeAllocator<umpire::strategy::QuickPool, true>(
      pool_name, rm.getAllocator(parent_name));
}


AMSDType getDataType(char *d_type)
{
  AMSDType dType = AMSDType::AMS_DOUBLE;
  if (std::strcmp(d_type, "float") == 0) {
    dType = AMSDType::AMS_SINGLE;
  } else if (std::strcmp(d_type, "double") == 0) {
    dType = AMSDType::AMS_DOUBLE;
  } else {
    assert(false && "Unknown data type");
  }
  return dType;
}

template <typename DType>
struct Problem {
  int num_inputs;
  int num_outputs;
  Problem(int ni, int no) : num_inputs(ni), num_outputs(no) {}

  void run(long num_elements, DType **inputs, DType **outputs)
  {
    for (int i = 0; i < num_elements; i++) {
      DType sum = 0;
      for (int j = 0; j < num_inputs; j++) {
        sum += inputs[j][i];
      }

      for (int j = 0; j < num_outputs; j++) {
        outputs[j][i] = sum;
      }
    }
  }


  const DType *initialize_inputs(DType *inputs, long length)
  {
    for (int i = 0; i < length; i++) {
      inputs[i] = static_cast<DType>(i);
    }
    return inputs;
  }

  void ams_run(AMSExecutor &wf,
               AMSResourceType resource,
               int iterations,
               int num_elements)
  {
    auto &rm = umpire::ResourceManager::getInstance();

    for (int i = 0; i < iterations; i++) {
      int elements = num_elements;  // * ((DType)(rand()) / RAND_MAX) + 1;
      std::vector<const DType *> inputs;
      std::vector<DType *> outputs;

      // Allocate Input memory
      for (int j = 0; j < num_inputs; j++) {
        DType *data = new DType[elements];
        inputs.push_back(initialize_inputs(data, elements));
      }

      // Allocate Output memory
      for (int j = 0; j < num_outputs; j++) {
        outputs.push_back(new DType[elements]);
      }

      AMSExecute(wf,
                 (void *)this,
                 elements,
                 reinterpret_cast<const void **>(inputs.data()),
                 reinterpret_cast<void **>(outputs.data()),
                 inputs.size(),
                 outputs.size());

      for (int i = 0; i < num_outputs; i++) {
        delete[] outputs[i];
        outputs[i] = nullptr;
      }


      for (int i = 0; i < num_inputs; i++) {
        delete[] inputs[i];
        inputs[i] = nullptr;
      }
    }
  }
};

void callBackDouble(void *cls, long elements, void **inputs, void **outputs)
{
  std::cout << "Called the double precision model\n";
  static_cast<Problem<double> *>(cls)->run(elements,
                                           (double **)(inputs),
                                           (double **)(outputs));
}


void callBackSingle(void *cls, long elements, void **inputs, void **outputs)
{
  std::cout << "Called the single precision model\n";
  static_cast<Problem<float> *>(cls)->run(elements,
                                          (float **)(inputs),
                                          (float **)(outputs));
}


int main(int argc, char **argv)
{

  if (argc != 7) {
    std::cout << "Wrong cli\n";
    std::cout << argv[0]
              << " use_device(0|1) num_inputs num_outputs "
                 "data_type(float|double) "
                 "num_iterations num_elements" << std::endl;
    // return -1;
  }

  // // -------------------------------------------------------------------------
  // // setup command line parser
  // // -------------------------------------------------------------------------
  // mfem::OptionsParser args(argc, argv);
  // args.AddOption(&device_name, "-d", "--device", "Device config string");

  // // data parameters
  // args.AddOption(&num_elems, "-e", "--num-elems", "Number of elements");

  // // random speed and packing
  // args.AddOption(&seed, "-s", "--seed", "Seed for rand");

  // args.AddOption(&db_config,
  //                "-db",
  //                "--dbconfig",
  //                "Path to directory where applications will store their data "
  //                "(or Path to JSON configuration if RabbitMQ is chosen)",
  //                reqDB);

  // args.AddOption(&db_type,
  //                "-dt",
  //                "--dbtype",
  //                "Configuration option of the different DB types:\n"
  //                "\t 'csv' Use csv as back end\n"
  //                "\t 'hdf5': use hdf5 as a back end\n"
  //                "\t 'rmq': use RabbitMQ as a back end\n");

  // args.AddOption(
  //     &verbose, "-v", "--verbose", "-qu", "--quiet", "Print extra stuff");

  // // -------------------------------------------------------------------------
  // // parse arguments
  // // -------------------------------------------------------------------------
  // args.Parse();
  // if (!args.Good()) {
  //   args.PrintUsage(std::cout);
  //   return -1;
  // }

  // if (rId == 0) {
  //   args.PrintOptions(std::cout);
  //   std::cout << std::endl;
  // }



  // int use_device = std::atoi(argv[1]);
  // int num_inputs = std::atoi(argv[2]);
  // int num_outputs = std::atoi(argv[3]);
  // AMSDType data_type = getDataType(argv[4]);
  // int num_iterations = std::atoi(argv[5]);
  // int avg_elements = std::atoi(argv[6]);
  AMSResourceType resource = AMSResourceType::AMS_HOST;
  srand(time(NULL));

  int num_inputs = 2;
  int num_outputs = 4;
  AMSDType data_type = getDataType("double");
  int num_iterations = 1;
  int avg_elements = 10;

  // Configure DB
  auto device_name = "cpu";
  auto db_type = "rmq";

  const bool use_device = std::strcmp(device_name, "cpu") != 0;
  AMSDBType dbType = AMSDBType::AMS_NONE;
  if (std::strcmp(db_type, "csv") == 0) {
    dbType = AMSDBType::AMS_CSV;
  } else if (std::strcmp(db_type, "hdf5") == 0) {
    dbType = AMSDBType::AMS_HDF5;
  } else if (std::strcmp(db_type, "rmq") == 0) {
    dbType = AMSDBType::AMS_RMQ;
  }


  char* db_config = "/g/g92/pottier1/ams/AMS/build_lassen_nompi/rmq-short.json";
  // AMSConfigureRMQDatabase(db_config);

  if (db_config == nullptr) dbType = AMSDBType::AMS_NONE;
  if (dbType != AMSDBType::AMS_RMQ) {
    AMSConfigureFSDatabase(dbType, db_config);
  }

  // const char *db_path = (strlen(db_config) > 0) ? db_config : nullptr;
  // std::ifstream json_file(db_config);
  // nlohmann::json data =  nlohmann::json::parse(json_file);
  // parseDatabase(data);


  createUmpirePool("HOST", "TEST_HOST");
  AMSSetAllocator(AMSResourceType::AMS_HOST, "TEST_HOST");

  AMSCAbstrModel ams_model = AMSQueryModel("no_model");

  // AMSCAbstrModel ams_model = AMSRegisterAbstractModel("bench_db_no_model",
  //                                                       AMSUQPolicy::AMS_RANDOM,
  //                                                       0.5,
  //                                                       "",
  //                                                       "",
  //                                                       "bench_db_no_model",
  //                                                       1);

  if (data_type == AMSDType::AMS_SINGLE) {
    Problem<float> prob(num_inputs, num_outputs);
    AMSExecutor wf = AMSCreateExecutor(ams_model,
                                        AMSDType::AMS_SINGLE,
                                        resource,
                                        (AMSPhysicFn)callBackSingle,
                                        0,
                                        1);

    prob.ams_run(wf, resource, num_iterations, avg_elements);
  } else {
    Problem<double> prob(num_inputs, num_outputs);
    AMSExecutor wf = AMSCreateExecutor(ams_model,
                                        AMSDType::AMS_DOUBLE,
                                        resource,
                                        (AMSPhysicFn)callBackDouble,
                                        0,
                                        1);
    prob.ams_run(wf, resource, num_iterations, avg_elements);
  }

  return 0;
}
