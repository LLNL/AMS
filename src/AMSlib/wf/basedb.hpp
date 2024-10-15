/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_BASE_DB__
#define __AMS_BASE_DB__


#include <H5Ipublic.h>

#include <cstdint>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "AMS.h"
#include "debug.h"
#include "wf/debug.h"
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"

namespace fs = std::experimental::filesystem;

#ifdef __ENABLE_REDIS__
#include <sw/redis++/redis++.h>

#include <iomanip>
#warning Redis is currently not supported/tested
#endif

#ifdef __ENABLE_HDF5__
#include <hdf5.h>
#define HDF5_ERROR(Eid)                                             \
  if (Eid < 0) {                                                    \
    std::cerr << "[Error] Happened in " << __FILE__ << ":"          \
              << __PRETTY_FUNCTION__ << " ( " << __LINE__ << ")\n"; \
    exit(-1);                                                       \
  }
#endif


#ifdef __ENABLE_RMQ__
#include <amqpcpp.h>
#include <amqpcpp/libevent.h>
#include <amqpcpp/linux_tcp.h>
#include <amqpcpp/throttle.h>
#include <event2/event-config.h>
#include <event2/event.h>
#include <event2/thread.h>
#include <openssl/err.h>
#include <openssl/opensslv.h>
#include <openssl/ssl.h>

#include <algorithm>
#include <chrono>
#include <deque>
#include <future>
#include <random>
#include <thread>
#include <tuple>

#endif  // __ENABLE_RMQ__

namespace ams
{
namespace db
{

AMSDBType getDBType(std::string type);
std::string getDBTypeAsStr(AMSDBType type);

/**
 * @brief A simple pure virtual interface to store data in some
 * persistent storage device
 */
class BaseDB
{
  /** @brief unique id of the process running this simulation */
  uint64_t id;
  /** @brief True if surrogate model update is allowed */
  bool allowUpdate;

public:
  BaseDB(const BaseDB&) = delete;
  BaseDB& operator=(const BaseDB&) = delete;

  BaseDB(uint64_t id) : id(id), allowUpdate(false) {}

  BaseDB(uint64_t id, bool allowUpdate) : id(id), allowUpdate(allowUpdate) {}

  virtual void close() {}

  virtual ~BaseDB() {}

  /**
   * @brief Define the type of the DB (File, Redis etc)
   */
  virtual std::string type() = 0;

  virtual AMSDBType dbType() = 0;

  /**
   * @brief Takes an input and an output vector each holding 1-D vectors data, and
   * store. them in persistent data storage.
   * @param[in] num_elements Number of elements of each 1-D vector
   * @param[in] inputs Vector of 1-D vectors containing the inputs to be stored
   * @param[in] inputs Vector of 1-D vectors, each 1-D vectors contains
   * 'num_elements'  values to be stored
   * @param[in] outputs Vector of 1-D vectors, each 1-D vectors contains
   * 'num_elements'  values to be stored
   */

  virtual void store(size_t num_elements,
                     std::vector<double*>& inputs,
                     std::vector<double*>& outputs,
                     bool* predicate = nullptr) = 0;


  virtual void store(size_t num_elements,
                     std::vector<float*>& inputs,
                     std::vector<float*>& outputs,
                     bool* predicate = nullptr) = 0;

  uint64_t getId() const { return id; }

  bool allowModelUpdate() { return allowUpdate; }

  virtual bool updateModel() { return false; }

  virtual std::string getLatestModel() { return {}; }

  virtual bool storePredicate() const { return false; }
};

/**
 * @brief A pure virtual interface for data bases storing data using
 * some file format (filesystem DB).
 */
class FileDB : public BaseDB
{
protected:
  /** @brief Path to file to write data to */
  std::string fn;
  /** @brief absolute path to directory storing the data */
  std::string fp;

  /**
   *  @brief check error code, if it exists print message and exit application
   *  @param[in] ec error code
   */
  void checkError(std::error_code& ec)
  {
    if (ec) {
      std::cerr << "Error in is_regular_file: " << ec.message();
      exit(-1);
    }
  }

public:
  /**
   * @brief Takes an input and an output vector each holding 1-D vectors data, and
   * store. them in persistent data storage.
   * @param[in] path Path to an existing directory where to store our data
   * @param[in] suffix The suffix of the file to write to
   * @param[in] rId a unique Id for each process taking part in a distributed
   * execution (rank-id)
   * */
  FileDB(std::string path,
         std::string fn,
         const std::string suffix,
         uint64_t rId)
      : BaseDB(rId)
  {
    fs::path Path(path);
    std::error_code ec;

    if (!fs::exists(Path, ec)) {
      std::cerr << "[ERROR]: Path:'" << path << "' does not exist\n";
      exit(-1);
    }

    checkError(ec);

    if (!fs::is_directory(Path, ec)) {
      std::cerr << "[ERROR]: Path:'" << path << "' is a file NOT a directory\n";
      exit(-1);
    }

    Path = fs::absolute(Path);
    fp = Path.string();

    // We can now create the filename
    std::string dbfn(fn + "_");
    dbfn += std::to_string(rId) + suffix;
    Path /= fs::path(dbfn);
    this->fn = fs::absolute(Path).string();
    DBG(DB, "File System DB writes to file %s", this->fn.c_str())
  }
};


class csvDB final : public FileDB
{
private:
  /** @brief file descriptor */
  bool writeHeader;
  std::fstream fd;

  PERFFASPECT()
  template <typename TypeValue>
  void _store(size_t num_elements,
              std::vector<TypeValue*>& inputs,
              std::vector<TypeValue*>& outputs)
  {
    DBG(DB,
        "DB of type %s stores %ld elements of input/output dimensions (%lu, "
        "%lu)",
        type().c_str(),
        num_elements,
        inputs.size(),
        outputs.size())

    const size_t num_in = inputs.size();
    const size_t num_out = outputs.size();

    if (writeHeader) {
      for (size_t i = 0; i < num_in; i++)
        fd << "input_" << i << ":";
      for (size_t i = 0; i < num_out - 1; i++)
        fd << "output_" << i << ":";
      fd << "output_" << num_out - 1 << "\n";
      writeHeader = false;
    }

    for (size_t i = 0; i < num_elements; i++) {
      for (size_t j = 0; j < num_in; j++) {
        fd << inputs[j][i] << ":";
      }

      for (size_t j = 0; j < num_out - 1; j++) {
        fd << outputs[j][i] << ":";
      }
      fd << outputs[num_out - 1][i] << "\n";
    }
  }


public:
  csvDB(const csvDB&) = delete;
  csvDB& operator=(const csvDB&) = delete;

  /**
   * @brief constructs the class and opens the file to write to
   * @param[in] fn Name of the file to store data to
   * @param[in] rId a unique Id for each process taking part in a distributed
   * execution (rank-id)
   */
  csvDB(std::string path, std::string fn, uint64_t rId)
      : FileDB(path, fn, ".csv", rId)
  {
    writeHeader = !fs::exists(this->fn);
    fd.open(this->fn, std::ios_base::app | std::ios_base::out);
    if (!fd.is_open()) {
      std::cerr << "Cannot open db file: " << this->fn << std::endl;
    }
    DBG(DB, "DB Type: %s", type().c_str())
  }

  /**
   * @brief deconstructs the class and closes the file
   */
  ~csvDB()
  {
    DBG(DB, "Closing File: %s %s", type().c_str(), this->fn.c_str())
    fd.close();
  }

  virtual void store(size_t num_elements,
                     std::vector<float*>& inputs,
                     std::vector<float*>& outputs,
                     bool* predicate = nullptr) override
  {
    CFATAL(CSV,
           predicate != nullptr,
           "CSV database does not support storing uq-predicates")

    _store(num_elements, inputs, outputs);
  }

  virtual void store(size_t num_elements,
                     std::vector<double*>& inputs,
                     std::vector<double*>& outputs,
                     bool* predicate = nullptr) override
  {

    CFATAL(CSV,
           predicate != nullptr,
           "CSV database does not support storing uq-predicates")

    _store(num_elements, inputs, outputs);
  }


  /**
   * @brief Define the type of the DB (File, Redis etc)
   */
  std::string type() override { return "csv"; }

  /**
   * @brief Return the DB enumerationt type (File, Redis etc)
   */
  AMSDBType dbType() override { return AMSDBType::AMS_CSV; };

  /**
   * @brief Takes an input and an output vector each holding 1-D vectors data, and
   * store them into a csv file delimited by ':'. This should never be used for
   * large scale simulations as txt/csv format will be extremely slow.
   * @param[in] num_elements Number of elements of each 1-D vector
   * @param[in] inputs Vector of 1-D vectors containing the inputs to bestored
   * @param[in] inputs Vector of 1-D vectors, each 1-D vectors contains
   * 'num_elements'  values to be stored
   * @param[in] outputs Vector of 1-D vectors, each 1-D vectors contains
   * 'num_elements'  values to be stored
   */
};


#ifdef __ENABLE_HDF5__
class hdf5DB final : public FileDB
{
private:
  /** @brief file descriptor */
  hid_t HFile;
  /** @brief vector holding the hdf5 dataset descriptor.
   * We currently store every input on a separate dataset
   */
  std::vector<hid_t> HDIsets;

  /** @brief vector holding the hdf5 dataset descriptor.
   * We currently store every output on a separate dataset
   */
  std::vector<hid_t> HDOsets;

  /** @brief Total number of elements we have in our file   */
  hsize_t totalElements;

  hid_t HDType;

  /** @brief the dataset descriptor of the predicates */
  hid_t pSet;

  const bool predicateStore;

  /** @brief create or get existing hdf5 dataset with the provided name
   * storing data as Ckunked pieces. The Chunk value controls the chunking
   * performed by HDF5 and thus controls the write performance
   * @param[in] group in which we will store data under
   * @param[in] dName name of the data set
   * @param[in] dataType dataType to be stored for this dataset
   * @param[in] Chunk chunk size of dataset used by HDF5.
   * @reval dataset HDF5 key value
   */
  hid_t getDataSet(hid_t group,
                   std::string dName,
                   hid_t dataType,
                   const size_t Chunk = 1024L);


  /**
   * @brief Create the HDF5 datasets and store their descriptors in the in/out
   * vectors
   * @param[in] num_elements of every vector
   * @param[in] numIn number of input 1-D vectors
   * @param[in] numOut number of output 1-D vectors
   */
  void createDataSets(size_t numElements,
                      const size_t numIn,
                      const size_t numOut);

  /**
   * @brief Write all the data in the vectors in the respective datasets.
   * @param[in] dsets Vector containing the hdf5-dataset descriptor for every
   * vector to be written
   * @param[in] data vectors containing 1-D vectors of numElements values each
   * to be written in the db.
   * @param[in] numElements The number of elements each vector has
   */
  template <typename TypeValue>
  void writeDataToDataset(std::vector<hid_t>& dsets,
                          std::vector<TypeValue*>& data,
                          size_t numElements);

  /** @brief Writes a single 1-D vector to the dataset
   * @param[in] dSet the dataset to write the data to
   * @param[in] data the data we need to write
   * @param[in] elements the number of data elements we have
   * @param[in] datatype of elements we will write
   */
  void writeVecToDataset(hid_t dSet, void* data, size_t elements, hid_t DType);

  PERFFASPECT()
  template <typename TypeValue>
  void _store(size_t num_elements,
              std::vector<TypeValue*>& inputs,
              std::vector<TypeValue*>& outputs,
              bool* predicate = nullptr);

public:
  // Delete copy constructors. We do not want to copy the DB around
  hdf5DB(const hdf5DB&) = delete;
  hdf5DB& operator=(const hdf5DB&) = delete;

  /**
   * @brief constructs the class and opens the hdf5 file to write to
   * @param[in] path path to directory to open/create the file 
   * @param[in] fn Name of the file to store the data to
   * @param[in] rId a unique Id for each process taking part in a distributed
   * execution (rank-id)
   */
  hdf5DB(std::string path,
         std::string domain_name,
         std::string fn,
         uint64_t rId,
         bool predicate = false);

  /**
   * @brief deconstructs the class and closes the file
   */
  ~hdf5DB();

  /**
   * @brief Define the type of the DB
   */
  std::string type() override { return "hdf5"; }

  /**
   * @brief Return the DB enumerationt type (File, Redis etc)
   */
  AMSDBType dbType() override { return AMSDBType::AMS_HDF5; };


  /**
   * @brief Takes an input and an output vector each holding 1-D vectors data,
   * and store them into a hdf5 file delimited by ':'. This should never be used
   * for large scale simulations as txt/hdf5 format will be extremely slow.
   * @param[in] num_elements Number of elements of each 1-D vector
   * @param[in] inputs Vector of 1-D vectors containing the inputs to bestored
   * @param[in] inputs Vector of 1-D vectors, each 1-D vectors contains
   * 'num_elements'  values to be stored
   * @param[in] outputs Vector of 1-D vectors, each 1-D vectors contains
   * 'num_elements'  values to be stored
   */
  void store(size_t num_elements,
             std::vector<float*>& inputs,
             std::vector<float*>& outputs,
             bool* predicate = nullptr) override;


  /**
   * @brief Takes an input and an output vector each holding 1-D vectors data,
   * and store them into a hdf5 file delimited by ':'. This should never be used
   * for large scale simulations as txt/hdf5 format will be extremely slow.
   * @param[in] num_elements Number of elements of each 1-D vector
   * @param[in] inputs Vector of 1-D vectors containing the inputs to bestored
   * @param[in] inputs Vector of 1-D vectors, each 1-D vectors contains
   * 'num_elements'  values to be stored
   * @param[in] outputs Vector of 1-D vectors, each 1-D vectors contains
   * 'num_elements'  values to be stored
   */
  void store(size_t num_elements,
             std::vector<double*>& inputs,
             std::vector<double*>& outputs,
             bool* predicate = nullptr) override;

  /**
   * @brief Returns whether the DB can also store predicate information for debug
   * purposes
   */
  bool storePredicate() const override { return predicateStore; }
};
#endif


#ifdef __ENABLE_REDIS__
template <typename TypeValue>
class RedisDB : public BaseDB<TypeValue>
{
  const std::string _fn;  // path to the file storing the DB access config
  uint64_t _dbid;
  sw::redis::Redis* _redis;
  uint64_t keyId;

public:
  RedisDB(const RedisDB&) = delete;
  RedisDB& operator=(const RedisDB&) = delete;

  /**
   * @brief constructs the class and opens the file to write to
   * @param[in] fn Name of the file to store data to
   * @param[in] rId a unique Id for each process taking part in a distributed
   * execution (rank-id)
   */
  RedisDB(std::string fn, uint64_t rId)
      : BaseDB<TypeValue>(rId), _fn(fn), _redis(nullptr), keyId(0)
  {
    _dbid = reinterpret_cast<uint64_t>(this);
    auto connection_info = read_json(fn);

    sw::redis::ConnectionOptions connection_options;
    connection_options.type = sw::redis::ConnectionType::TCP;
    connection_options.host = connection_info["host"];
    connection_options.port = std::stoi(connection_info["service-port"]);
    connection_options.password = connection_info["database-password"];
    connection_options.db = 0;  // Optionnal, 0 is the default
    connection_options.tls.enabled =
        true;  // Required to connect to PDS within LC
    connection_options.tls.cacert = connection_info["cert"];

    sw::redis::ConnectionPoolOptions pool_options;
    pool_options.size = 100;  // Pool size, i.e. max number of connections.

    _redis = new sw::redis::Redis(connection_options, pool_options);
  }

  ~RedisDB()
  {
    std::cerr << "Deleting RedisDB object\n";
    delete _redis;
  }

  inline std::string type() override { return "RedisDB"; }

  /**
   * @brief Return the DB enumerationt type (File, Redis etc)
   */
  AMSDBType dbType() { return AMSDBType::REDIS; };


  inline std::string info() { return _redis->info(); }

  // Return the number of keys in the DB
  inline long long dbsize() { return _redis->dbsize(); }

  /* !
   * ! WARNING: Flush the entire Redis, accross all DBs!
   * !
   */
  inline void flushall() { _redis->flushall(); }

  /*
   * ! WARNING: Flush the entire current DB!
   * !
   */
  inline void flushdb() { _redis->flushdb(); }

  std::unordered_map<std::string, std::string> read_json(std::string fn)
  {
    std::ifstream config;
    std::unordered_map<std::string, std::string> connection_info = {
        {"database-password", ""},
        {"host", ""},
        {"service-port", ""},
        {"cert", ""},
    };

    config.open(fn, std::ifstream::in);
    if (config.is_open()) {
      std::string line;
      // Quite inefficient parsing (to say the least..) but the file to parse is
      // small (4 lines)
      // TODO: maybe use Boost or another JSON library
      while (std::getline(config, line)) {
        if (line.find("{") != std::string::npos ||
            line.find("}") != std::string::npos) {
          continue;
        }
        line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
        line.erase(std::remove(line.begin(), line.end(), ','), line.end());
        line.erase(std::remove(line.begin(), line.end(), '"'), line.end());

        std::string key = line.substr(0, line.find(':'));
        line.erase(0, line.find(":") + 1);
        connection_info[key] = line;
        // std::cerr << "key=" << key << " and value=" << line << std::endl;
      }
      config.close();
    } else {
      std::cerr << "Config located at: " << fn << std::endl;
      throw std::runtime_error("Could not open Redis config file");
    }
    return connection_info;
  }

  void store(size_t num_elements,
             std::vector<TypeValue*>& inputs,
             std::vector<TypeValue*>& outputs,
             bool predicate = nullptr) override
  {

    CFATAL(REDIS,
           predicate != nullptr,
           "REDIS database does not support storing uq-predicates")

    const size_t num_in = inputs.size();
    const size_t num_out = outputs.size();

    // TODO:
    //      Make insertion more efficient.
    //      Right now it's pretty naive and expensive
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_elements; i++) {
      std::string key = std::to_string(_dbid) + ":" + std::to_string(keyId) +
                        ":" +
                        std::to_string(i);  // In Redis a key must be a string
      std::ostringstream fd;
      for (size_t j = 0; j < num_in; j++) {
        fd << inputs[j][i] << ":";
      }
      for (size_t j = 0; j < num_out - 1; j++) {
        fd << outputs[j][i] << ":";
      }
      fd << outputs[num_out - 1][i];
      std::string val(fd.str());
      _redis->set(key, val);
    }

    keyId += 1;

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    auto nb_keys = this->dbsize();

    std::cout << std::setprecision(2) << "Inserted " << num_elements
              << " keys [Total keys = " << nb_keys << "]  into RedisDB [Total "
              << duration.count() << "ms, "
              << static_cast<double>(num_elements) / duration.count()
              << " key/ms]" << std::endl;
  }
};

#endif  // __ENABLE_REDIS__

#ifdef __ENABLE_RMQ__

enum RMQConnectionStatus { FAILED, CONNECTED, CLOSED, ERROR };

/**
  * @brief AMS represents the header as follows:
  * The header is 16 bytes long:
  *   - 1 byte is the size of the header (here 16). Limit max: 255
  *   - 1 byte is the precision (4 for float, 8 for double). Limit max: 255
  *   - 2 bytes are the MPI rank (0 if AMS is not running with MPI). Limit max: 65535
  *   - 2 bytes to store the size of the MSG domain name. Limit max: 65535
  *   - 4 bytes are the number of elements in the message. Limit max: 2^32 - 1
  *   - 2 bytes are the input dimension. Limit max: 65535
  *   - 2 bytes are the output dimension. Limit max: 65535
  *   - 2 bytes for padding. Limit max: 2^16 - 1
  *
  * |_Header_|_Datatype_|___Rank___|__DomainSize__|__#elems__|___InDim____|___OutDim___|_Pad_|.real data.|
  * ^        ^          ^          ^              ^          ^            ^            ^     ^           ^
  * | Byte 1 |  Byte 2  | Byte 3-4 |  Byte 5-6    |Byte 6-10 | Byte 10-12 | Byte 12-14 |-----| Byte 16-k |
  *
  * where X = datatype * num_element * (InDim + OutDim). Total message size is 16+k. 
  *
  * The data starts at byte 16, ends at byte k.
  * The data is structured as pairs of input/outputs. Let K be the total number of 
  * elements, then we have K pairs of inputs/outputs (either float or double):
  *
  *  |__Header_(16B)__|__Input 1__|__Output 1__|...|__Input_K__|__Output_K__|
  */
struct AMSMsgHeader {
  /** @brief Header size (bytes) */
  uint8_t hsize;
  /** @brief Data type size (bytes) */
  uint8_t dtype;
  /** @brief MPI rank */
  uint16_t mpi_rank;
  /** @brief Domain Name Size */
  uint16_t domain_size;
  /** @brief Number of elements */
  uint32_t num_elem;
  /** @brief Inputs dimension */
  uint16_t in_dim;
  /** @brief Outputs dimension */
  uint16_t out_dim;

  /**
   * @brief Constructor for AMSMsgHeader
   * @param[in]  mpi_rank     MPI rank
   * @param[in]  num_elem     Number of elements (input/outputs)
   * @param[in]  in_dim       Inputs dimension
   * @param[in]  out_dim      Outputs dimension
   */
  AMSMsgHeader(size_t mpi_rank,
               size_t domain_size,
               size_t num_elem,
               size_t in_dim,
               size_t out_dim,
               size_t type_size);

  /**
   * @brief Constructor for AMSMsgHeader
   * @param[in]  mpi_rank     MPI rank
   * @param[in]  num_elem     Number of elements (input/outputs)
   * @param[in]  in_dim       Inputs dimension
   * @param[in]  out_dim      Outputs dimension
   */
  AMSMsgHeader(uint16_t mpi_rank,
               uint16_t domain_size,
               uint32_t num_elem,
               uint16_t in_dim,
               uint16_t out_dim,
               uint8_t type_size);

  /**
   * @brief Return the size of a header in the AMS protocol.
   * @return The size of a message header in AMS (in byte)
   */
  static size_t constexpr size()
  {
    return ((sizeof(hsize) + sizeof(dtype) + sizeof(mpi_rank) +
             sizeof(domain_size) + sizeof(num_elem) + sizeof(in_dim) +
             sizeof(out_dim) + sizeof(double) - 1) /
            sizeof(double)) *
           sizeof(double);
  }

  /**
   * @brief Fill an empty buffer with a valid header.
   * @param[in] data_blob The buffer to fill
   * @return The number of bytes in the header or 0 if error
   */
  size_t encode(uint8_t* data_blob);

  /**
   * @brief Return a valid header based on a pre-existing data buffer
   * @param[in] data_blob The buffer to fill
   * @return An AMSMsgHeader with the correct attributes
   */
  static AMSMsgHeader decode(uint8_t* data_blob);
};


/**
 * @brief Class representing a message for the AMSLib
 */
class AMSMessage
{
public:
  /** @brief message ID */
  int _id;
  /** @brief The MPI rank (0 if MPI is not used) */
  uint64_t _rank;
  /** @brief The data represented as a binary blob */
  uint8_t* _data;
  /** @brief The total size of the binary blob in bytes */
  size_t _total_size;
  /** @brief The number of input/output pairs */
  size_t _num_elements;
  /** @brief The dimensions of inputs */
  size_t _input_dim;
  /** @brief The dimensions of outputs */
  size_t _output_dim;

  /**
   * @brief Empty constructor
   */
  AMSMessage()
      : _id(0),
        _rank(0),
        _num_elements(0),
        _input_dim(0),
        _output_dim(0),
        _data(nullptr),
        _total_size(0)
  {
  }

  /**
   * @brief Constructor
   * @param[in]  id                  ID of the message
   * @param[in]  rId                 MPI Rank of the messages (0 default)
   * @param[in]  num_elements        Number of elements
   * @param[in]  inputs              Inputs
   * @param[in]  outputs             Outputs
   */
  template <typename TypeValue>
  AMSMessage(int id,
             uint64_t rId,
             std::string& domain_name,
             size_t num_elements,
             const std::vector<TypeValue*>& inputs,
             const std::vector<TypeValue*>& outputs)
      : _id(id),
        _rank(rId),
        _num_elements(num_elements),
        _input_dim(inputs.size()),
        _output_dim(outputs.size()),
        _data(nullptr),
        _total_size(0)
  {
    AMSMsgHeader header(_rank,
                        domain_name.size(),
                        _num_elements,
                        _input_dim,
                        _output_dim,
                        sizeof(TypeValue));

    _total_size = AMSMsgHeader::size() + domain_name.size() +
                  getTotalElements() * sizeof(TypeValue);
    auto& rm = ams::ResourceManager::getInstance();
    _data = rm.allocate<uint8_t>(_total_size, AMSResourceType::AMS_HOST);

    size_t current_offset = header.encode(_data);
    std::memcpy(&_data[current_offset],
                domain_name.c_str(),
                domain_name.size());
    current_offset += domain_name.size();
    current_offset +=
        encode_data(reinterpret_cast<TypeValue*>(_data + current_offset),
                    inputs,
                    outputs);
    DBG(AMSMessage, "Allocated message %d: %p", _id, _data);
  }

  /**
   * @brief Constructor
   * @param[in]  id                  ID of the message
   * @param[in]  rId                 MPI rank of the message
   * @param[in]  data                Pointer containing data
   */
  AMSMessage(int id, uint64_t rId, uint8_t* data);

  AMSMessage(const AMSMessage& other)
  {
    DBG(AMSMessage, "Copy AMSMessage : %p -- %d", other._data, other._id);
    swap(other);
  };

  /**
   * @brief Internal Method swapping for AMSMessage
   * @param[in]  other         Message to swap
   */
  void swap(const AMSMessage& other);

  AMSMessage& operator=(const AMSMessage&) = delete;

  AMSMessage(AMSMessage&& other) noexcept { *this = std::move(other); }

  AMSMessage& operator=(AMSMessage&& other) noexcept
  {
    // DBG(AMSMessage, "Move AMSMessage : %p -- %d", other._data, other._id);
    if (this != &other) {
      swap(other);
      other._data = nullptr;
    }
    return *this;
  }

  /**
   * @brief Fill a buffer with a data section starting at a given position.
   * @param[in]  data_blob          The buffer to fill
   * @param[in]  offset     Position where to start writing in the buffer
   * @param[in]  inputs             Inputs
   * @param[in]  outputs            Outputs
   * @return The number of bytes in the message or 0 if error
   */
  template <typename TypeValue>
  size_t encode_data(TypeValue* data_blob,
                     const std::vector<TypeValue*>& inputs,
                     const std::vector<TypeValue*>& outputs)
  {
    size_t x_dim = _input_dim + _output_dim;
    if (!data_blob) return 0;
    // Creating the body part of the messages
    for (size_t i = 0; i < _num_elements; i++) {
      for (size_t j = 0; j < _input_dim; j++) {
        data_blob[i * x_dim + j] = inputs[j][i];
      }
    }

    for (size_t i = 0; i < _num_elements; i++) {
      for (size_t j = 0; j < _output_dim; j++) {
        data_blob[i * x_dim + _input_dim + j] = outputs[j][i];
      }
    }

    return (x_dim * _num_elements) * sizeof(TypeValue);
  }

  /**
   *  @brief Return the total number of elements in this message
   *  @return  Size in bytes of the data portion
   */
  size_t getTotalElements() const
  {
    return (_num_elements * (_input_dim + _output_dim));
  }

  /**
   * @brief Return the underlying data pointer
   * @return Data pointer (binary blob)
   */
  uint8_t* data() const { return _data; }

  /**
   * @brief Return message ID
   * @return message ID
   */
  int id() const { return _id; }

  /**
   * @brief Return MPI rank
   * @return MPI rank
   */
  int rank() const { return _rank; }

  /**
   * @brief Return the size in bytes of the underlying binary blob
   * @return Byte size of data pointer
   */
  size_t size() const { return _total_size; }

  ~AMSMessage()
  {
    DBG(AMSMessage,
        "Destroying message  %d: %p (underlying memory NOT freed)",
        _id,
        _data)
  }
};  // class AMSMessage


/**
 * @brief Structure that represents incoming RabbitMQ messages.
 */
class AMSMessageInbound
{
public:
  /** @brief Delivery tag (ID of the message) */
  uint64_t id;
  /** @brief MPI rank */
  uint64_t rId;
  /** @brief message content (body) */
  std::string body;
  /** @brief RabbitMQ exchange from which the message has been received */
  std::string exchange;
  /** @brief routing key */
  std::string routing_key;
  /** @brief True if messages has been redelivered */
  bool redelivered;

  AMSMessageInbound() = default;

  AMSMessageInbound(AMSMessageInbound&) = default;
  AMSMessageInbound& operator=(AMSMessageInbound&) = default;

  AMSMessageInbound(AMSMessageInbound&&) = default;
  AMSMessageInbound& operator=(AMSMessageInbound&&) = default;

  AMSMessageInbound(uint64_t id,
                    uint64_t rId,
                    std::string body,
                    std::string exchange,
                    std::string routing_key,
                    bool redelivered);

  /**
   * @brief Check if a message is empty.
   * @return True if message is empty
   */
  bool empty();

  /**
   * @brief Check if a message is empty.
   * @return True if message is empty.
   */
  bool isTraining();

  /**
   * @brief Get the model path from the message.
   * @return Return model path or empty string if no model available.
   */
  std::string getModelPath();

private:
  /**
   * @brief Check if a message is empty.
   * @return True if message is empty
   */
  std::vector<std::string> splitString(std::string str, std::string delimiter);

};  // class AMSMessageInbound


/**
 * @brief Specific handler for RabbitMQ connections based on libevent.
 */
class RMQHandler : public AMQP::LibEventHandler
{
protected:
  /** @brief Path to TLS certificate (if empty, no TLS certificate)*/
  std::string _cacert;
  /** @brief MPI rank (0 if no MPI support) */
  uint64_t _rId;
  /** @brief LibEvent I/O loop */
  std::shared_ptr<struct event_base> _loop;

  std::promise<RMQConnectionStatus> establish_connection;
  std::future<RMQConnectionStatus> established;

  std::promise<RMQConnectionStatus> close_connection;
  std::future<RMQConnectionStatus> closed;

  std::promise<RMQConnectionStatus> error_connection;
  std::future<RMQConnectionStatus> ftr_error;

public:
  /**
   *  @brief Constructor
   *  @param[in]  loop         Event Loop
   *  @param[in]  rId          MPI rank
   *  @param[in]  cacert       SSL Cacert
   */
  RMQHandler(uint64_t rId,
             std::shared_ptr<struct event_base> loop,
             std::string cacert = "");

  ~RMQHandler() = default;

  /**
   *  @brief  Wait (blocking call) until connection has been established or that ms * repeat is over.
   *  @param[in]  ms            Number of milliseconds the function will wait on the future
   *  @param[in]  repeat        Number of times the function will wait
   *  @return     True if connection has been established
   */
  bool waitToEstablish(unsigned ms, int repeat = 1);

  /**
   *  @brief  Wait (blocking call) until connection has been closed or that ms * repeat is over.
   *  @param[in]  ms            Number of milliseconds the function will wait on the future
   *  @param[in]  repeat        Number of times the function will wait
   *  @return     True if connection has been closed
   */
  bool waitToClose(unsigned ms, int repeat = 1);

  /**
   *  @brief  Check if the connection can be used to send messages.
   *  @return     True if connection is valid (i.e., can send messages)
   */
  bool connectionValid();

private:
  /**
   *  @brief Method that is called after a TCP connection has been set up, and
   * right before the SSL handshake is going to be performed to secure the
   * connection (only for amqps:// connections). This method can be overridden
   * in user space to load client side certificates.
   *  @param[in]  connection      The connection for which TLS was just started
   *  @param[in]  ssl             Pointer to the SSL structure that can be
   * modified
   *  @return     bool            True to proceed / accept the connection, false
   * to break up
   */
  virtual bool onSecuring(AMQP::TcpConnection* connection, SSL* ssl) override;

  /**
   *  @brief Method that is called when the secure TLS connection has been
   * established. This is only called for amqps:// connections. It allows you to
   * inspect whether the connection is secure enough for your liking (you can
   *  for example check the server certificate). The AMQP protocol still has
   *  to be started.
   *  @param[in]  connection      The connection that has been secured
   *  @param[in]  ssl             SSL structure from openssl library
   *  @return     bool            True if connection can be used
   */
  virtual bool onSecured(AMQP::TcpConnection* connection,
                         const SSL* ssl) override;

  /**
    *  Method that is called when the AMQP protocol is ended. This is the
    *  counter-part of a call to connection.close() to graceful shutdown
    *  the connection. Note that the TCP connection is at this time still 
    *  active, and you will also receive calls to onLost() and onDetached()
    *  @param  connection      The connection over which the AMQP protocol ended
    */
  virtual void onClosed(AMQP::TcpConnection* connection) override;

  /**
   *  @brief Method that is called by the AMQP library when a fatal error occurs
   *  on the connection, for example because data received from RabbitMQ
   *  could not be recognized, or the underlying connection is lost. This
   *  call is normally followed by a call to onLost() (if the error occurred
   *  after the TCP connection was established) and onDetached().
   *  @param[in]  connection      The connection on which the error occurred
   *  @param[in]  message         A human readable error message
   */
  virtual void onError(AMQP::TcpConnection* connection,
                       const char* message) override;

  /**
    *  @brief Final method that is called. This signals that no further calls to your
    *  handler will be made about the connection.
    *  @param  connection      The connection that can be destructed
    */
  virtual void onDetached(AMQP::TcpConnection* connection) override;

  bool waitFuture(std::future<RMQConnectionStatus>& future,
                  unsigned ms,
                  int repeat);
};  // class RMQHandler

/**
 * @brief Specific handler for RabbitMQ connections based on libevent.
 * 
 * Each MPI rank has its RMQConsumerHandler managing its own RabbitMQ queue.
 * RabbitMQ will generate random queue name, this queue will be bound
 * to the exchange provided.
 * 
 * Important, if the exchange already exist for a given ExchangeType (from
 * a previous run for example), then trying to create an exchange with the
 * same name but with a different ExchangeType will lead to a crash. In that
 * case, either you remove the exchange manually on the RabbitMQ server or
 * you use an exchange name that does not exist (different name).
 * 
 * Note that, if messages are sent to that exchange before a queue is bound,
 * these messages are lost. RabbitMQ can notify the sender that these messages
 * never arrived if the sender uses publication confirmation.
 */
class RMQConsumerHandler final : public RMQHandler
{
private:
  /** @brief main channel used to send data to the broker */
  std::shared_ptr<AMQP::TcpChannel> _channel;
  /** @brief RabbitMQ queue (internal use only) */
  std::string _queue;
  /** @brief RabbitMQ exchange */
  std::string _exchange;
  /** @brief RabbitMQ routing key */
  std::string _routing_key;
  /** @brief Type of the exchange used (AMQP::topic, AMQP::fanout, AMQP::direct) */
  AMQP::ExchangeType _extype;
  /** @brief Queue that contains all the messages received on receiver queue */
  std::shared_ptr<std::vector<AMSMessageInbound>> _messages;

public:
  /**
   *  @brief Constructor
   *  @param[in]  loop         Event Loop
   *  @param[in]  cacert       SSL Cacert
   *  @param[in]  routing_key  Routing key
   *  @param[in]  exchange     Exchange
   */
  RMQConsumerHandler(uint64_t rId,
                     std::shared_ptr<struct event_base> loop,
                     std::string cacert,
                     std::string exchange,
                     std::string routing_key,
                     AMQP::ExchangeType extype = AMQP::fanout);

  /**
   *  @brief Delete the message with given ID
   *  @param[in] delivery_tag Delivery tag that will be deleted (if found)
   */
  void delMessage(uint64_t delivery_tag) { getMessages(delivery_tag, true); }

  /**
   *  @brief Check if messages received contains new model paths
   *  @return Return a tuple with the ID and path of the latest model available or ID=0 and empty string if no model available
   */
  std::tuple<uint64_t, std::string> getLatestModel();

  /**
   *  @brief Return the most recent messages and delete it
   *  @return A structure AMSMessageInbound which is a std::tuple (see typedef)
   */
  AMSMessageInbound popMessages();

  /**
   *  @brief Return the message corresponding to the delivery tag. Do not delete the
   *  message.
   *  @param[in] delivery_tag Delivery tag that will be returned (if found)
   *  @param[in] erase if True, the element will also be deleted from underyling structure
   *  @return A structure AMSMessageInbound which is a std::tuple (see typedef)
   */
  AMSMessageInbound getMessages(uint64_t delivery_tag, bool erase);

  ~RMQConsumerHandler() = default;

private:
  /**
   *  @brief Method that is called by the AMQP library when the login attempt
   *  succeeded. After this the connection is ready to use.
   *  @param[in]  connection      The connection that can now be used
   */
  virtual void onReady(AMQP::TcpConnection* connection) override;
};  // class RMQConsumerHandler


/**
 * @brief Class that manages a RabbitMQ broker and handles connection, event
 * loop and set up various handlers.
 */
class RMQConsumer
{
private:
  /** @brief Connection to the broker */
  AMQP::TcpConnection* _connection;
  /** @brief name of the exchange */
  std::string _exchange;
  /** @brief name of the routing binded to exchange */
  std::string _routing_key;
  /** @brief TLS certificate file */
  std::string _cacert;
  /** @brief MPI rank (if MPI is used, otherwise 0) */
  uint64_t _rId;
  /** @brief The event loop for sender (usually the default one in libevent) */
  std::shared_ptr<struct event_base> _loop;
  /** @brief The handler which contains various callbacks for the sender */
  std::shared_ptr<RMQConsumerHandler> _handler;
  /** @brief Queue that contains all the messages received on receiver queue (messages can be popped in) */
  std::vector<AMSMessageInbound> _messages;

public:
  RMQConsumer(const RMQConsumer&) = delete;
  RMQConsumer& operator=(const RMQConsumer&) = delete;

  RMQConsumer(uint64_t rId,
              const AMQP::Address& address,
              std::string cacert,
              std::string routing_key,
              std::string exchange);

  /**
   *  @brief Start the underlying I/O loop (blocking call)
   */
  void start();

  /**
   *  @brief Stop the underlying I/O loop
   */
  void stop();

  /**
   *  @brief Check if the underlying RabbitMQ connection is ready and usable
   *  @return True if the publisher is ready to publish
   */
  bool ready();

  /**
   *  @brief Wait that the connection is ready (blocking call)
   *  @param[in] ms Number of milliseconds to wait between each tentative
   *  @param[in] repeat Number of tentatives
   *  @return True if the publisher is ready to publish
   */
  bool waitToEstablish(unsigned ms, int repeat = 1);

  /**
   *  @brief Return the most recent messages and delete it
   *  @return A structure AMSMessageInbound which is a std::tuple (see typedef)
   */
  AMSMessageInbound popMessages();

  /**
   *  @brief Delete the message with given ID
   *  @param[in] delivery_tag Delivery tag that will be deleted (if found)
   */
  void delMessage(uint64_t delivery_tag);

  /**
   *  @brief Return the message corresponding to the delivery tag. Do not delete the
   *  message.
   *  @param[in] delivery_tag Delivery tag that will be returned (if found)
   *  @param[in] erase if True, the element will also be deleted from underyling structure
   *  @return A structure AMSMessageInbound which is a std::tuple (see typedef)
   */
  AMSMessageInbound getMessages(uint64_t delivery_tag, bool erase = false);

  /**
   *  @brief Return the path of latest ML model available
   *  @return Tuple with ID of new model and ML model path or empty string if no model available
   */
  std::tuple<uint64_t, std::string> getLatestModel();

  /**
   *  @brief    Close the unerlying connection
   *  @param[in] ms Number of milliseconds to wait between each tentative
   *  @param[in] repeat Number of tentatives
   *  @return  True if connection was closed properly
   */
  bool close(unsigned ms, int repeat = 1);

  ~RMQConsumer();
};  // class RMQConsumer

/**
 * @brief Specific handler for RabbitMQ connections based on libevent.
 */
class RMQPublisherHandler final : public RMQHandler
{
private:
  std::shared_ptr<AMQP::TcpChannel> _channel;
  /** @brief AMQP reliable channel (wrapper of classic channel with added functionalities) */
  std::shared_ptr<AMQP::Reliable<AMQP::Tagger>> _rchannel;
  /** @brief RabbitMQ queue */
  std::string _queue;
  /** @brief Total number of messages sent */
  int _nb_msg;
  /** @brief Number of messages successfully acknowledged */
  int _nb_msg_ack;
  /** @brief Mutex to protect multithread accesses to _messages */
  std::mutex _mutex;
  /** @brief Messages that have not been successfully acknowledged */
  std::vector<AMSMessage> _messages;

public:
  /**
   *  @brief Constructor
   *  @param[in]  loop         Event Loop
   *  @param[in]  cacert       SSL Cacert
   *  @param[in]  rank         MPI rank
   */
  RMQPublisherHandler(uint64_t rId,
                      std::shared_ptr<struct event_base> loop,
                      std::string cacert,
                      std::string queue);

  ~RMQPublisherHandler() = default;

  /**
   *  @brief  Publish data on RMQ queue.
   *  @param[in]  msg            The AMSMessage to publish
   */
  void publish(AMSMessage&& msg);

  /**
   *  @brief  Return the messages that have NOT been acknowledged by the RabbitMQ server. 
   *  @return     A vector of AMSMessage
   */
  std::vector<AMSMessage>& msgBuffer();

  /**
   *  @brief    Free AMSMessages held by the handler
   */
  void cleanup();

  /**
   *  @brief    Total number of messages sent
   *  @return   Number of messages
   */
  int msgSent() const;

  /**
   *  @brief    Total number of messages successfully acknowledged
   *  @return   Number of messages
   */
  int msgAcknowledged() const;

  /**
   *  @brief    Total number of messages unacknowledged
   *  @return   Number of messages unacknowledged
   */
  unsigned unacknowledged() const;

  /**
   *  @brief    Flush the handler by waiting for all unacknowledged mesages.
   *            it will wait for a given amount of time until timeout.
   */
  void flush();

private:
  /**
   *  @brief Method that is called by the AMQP library when the login attempt
   *  succeeded. After this the connection is ready to use.
   *  @param[in]  connection      The connection that can now be used
   */
  virtual void onReady(AMQP::TcpConnection* connection) override;

  /**
   *  @brief  Free the data pointed pointer in a vector and update vector.
   *  @param[in]  addr            Address of memory to free.
   *  @param[in]  buffer          The vector containing memory buffers
   */
  void freeMessage(int msg_id, std::vector<AMSMessage>& buf);

  /**
   *  @brief  Free the data pointed by each pointer in a vector.
   *  @param[in]  buffer            The vector containing memory buffers
   */
  void freeAllMessages(std::vector<AMSMessage>& buffer);

};  // class RMQPublisherHandler


/**
 * @brief Class that manages a RabbitMQ broker and handles connection, event
 * loop and set up various handlers.
 */
class RMQPublisher
{
private:
  /** @brief Connection to the broker */
  AMQP::TcpConnection* _connection;
  /** @brief MPI rank (0 if no MPI support) */
  uint64_t _rId;
  /** @brief name of the queue to send data */
  std::string _queue;
  /** @brief TLS certificate file */
  std::string _cacert;
  /** @brief MPI rank (if MPI is used, otherwise 0) */
  int _rank;
  /** @brief The event loop for sender (usually the default one in libevent) */
  std::shared_ptr<struct event_base> _loop;
  /** @brief The handler which contains various callbacks for the sender */
  std::shared_ptr<RMQPublisherHandler> _handler;
  /** @brief Buffer holding unacknowledged messages in case of crash */
  std::vector<AMSMessage> _buffer_msg;

public:
  RMQPublisher(const RMQPublisher&) = delete;
  RMQPublisher& operator=(const RMQPublisher&) = delete;

  RMQPublisher(
      uint64_t rId,
      const AMQP::Address& address,
      std::string cacert,
      std::string queue,
      std::vector<AMSMessage>&& msgs_to_send = std::vector<AMSMessage>());

  /**
   * @brief Check if the underlying RabbitMQ connection is ready and usable
   * @return True if the publisher is ready to publish
   */
  bool ready_publish();

  /**
   * @brief Wait that the connection is ready (blocking call)
   * @return True if the publisher is ready to publish
   */
  bool waitToEstablish(unsigned ms, int repeat = 1);

  /**
   * @brief Return the number of unacknowledged messages
   * @return Number of unacknowledged messages
   */
  unsigned unacknowledged() const;

  /**
   * @brief Start the underlying I/O loop (blocking call)
   */
  void start();

  /**
   * @brief Stop the underlying I/O loop
   */
  void stop();

  /**
   * @brief Check if the underlying connection has no errors
   * @return True if no errors
   */
  bool connectionValid();

  /**
   * @brief Return the messages that have not been acknowledged.
   * It does not mean they have not been delivered but the
   * acknowledgements have not arrived yet.
   * @return A vector of AMSMessage
   */
  std::vector<AMSMessage>& getMsgBuffer();

  /**
   *  @brief    Total number of messages successfully acknowledged
   *  @return   Number of messages
   */
  void cleanup();

  void publish(AMSMessage&& message);

  /**
   *  @brief    Total number of messages sent
   *  @return   Number of messages
   */
  int msgSent() const;

  /**
   *  @brief    Total number of messages successfully acknowledged
   *  @return   Number of messages
   */
  int msgAcknowledged() const;

  /**
   *  @brief    Total number of messages successfully acknowledged
   *  @return   Number of messages
   */
  bool close(unsigned ms, int repeat = 1);

  ~RMQPublisher() = default;

};  // class RMQPublisher

/**
 * @brief Class that manages a RabbitMQ broker and handles connection, event
 * loop and set up various handlers.
 * @details This class handles a specific type of database backend in AMSLib.
 * Instead of writing inputs/outputs directly to files (CSV or HDF5), we
 * send these elements (a collection of inputs and their corresponding outputs)
 * to a service called RabbitMQ which is listening on a given IP and port.
 * 
 * This class requires a RabbitMQ server to be running somewhere,
 * the credentials of that server should be formatted as a JSON file as follows:
 *
 *  {
 *    "rabbitmq-name": "testamsrabbitmq",
 *    "rabbitmq-password": "XXX",
 *    "rabbitmq-user": "pottier1",
 *    "rabbitmq-vhost": "ams",
 *    "service-port": 31495,
 *    "service-host": "url.czapps.llnl.gov",
 *    "rabbitmq-cert": "tls-cert.crt",
 *    "rabbitmq-outbound-queue": "test3",
 *    "rabbitmq-exchange": "ams-fanout",
 *    "rabbitmq-routing-key": "training"
 *  }
 *
 * The TLS certificate must be generated by the user and the absolute paths are preferred.
 * A TLS certificate can be generated with the following command:
 *
 *    openssl s_client \ 
 *        -connect $REMOTE_HOST:$REMOTE_PORT -showcerts < /dev/null \
 *        2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' > tls.crt
 * 
 * RabbitMQDB creates two RabbitMQ connections per MPI rank, one for publishing data to RMQ and one for consuming data.
 * Each connection has its own I/O loop (based on Libevent) running in a dedicated thread because I/O loop are blocking.
 * Therefore, we have two threads per MPI rank.
 *
 * Here, RMQInterface::publish() has access to internal RabbitMQ channels and can publish the message 
 * on the outbound queue (rabbitmq-outbound-queue in the JSON configuration).
 * Note that storing data like that is much faster than with writing files as a call to RabbitMQDB::store()
 * is virtually free, the actual data sending part is taking place in a thread and does not slow down
 * the main simulation (MPI).
 *
 * 2. Consuming data: The inbound queue (rabbitmq-inbound-queue in the JSON configuration) is the queue for incoming data. The
 * RMQConsumer is listening on that queue for messages. In the AMSLib approach, that queue is used to communicate 
 * updates to rank regarding the ML surrrogate model. RMQConsumer will automatically populate a std::vector with all
 * messages received since the execution of AMS started.
 *
 * Global note: Most calls dealing with RabbitMQ (to establish a RMQ connection, opening a channel, publish data etc)
 * are asynchronous callbacks (similar to asyncio in Python or future in C++).
 * So, the simulation can have already started and the RMQ connection might not be valid which is why most part
 * of the code that deals with RMQ are wrapped into callbacks that will get run only in case of success.
 * For example, we create a channel only if the underlying connection has been succesfuly initiated
 * (see RMQPublisherHandler::onReady()).
 */
class RMQInterface
{
private:
  /** @brief Path of the config file (JSON) */
  std::string _config;
  /** @brief MPI rank (0 if no MPI support) */
  uint64_t _rId;
  /** @brief name of the queue to send data */
  std::string _queue_sender;
  /** @brief name of the exchange to receive data */
  std::string _exchange;
  /** @brief name of the routing key to receive data */
  std::string _routing_key;
  /** @brief Address of the RabbitMQ server */
  std::shared_ptr<AMQP::Address> _address;
  /** @brief TLS certificate path */
  std::string _cacert;
  /** @brief Represent the ID of the last message sent */
  int _msg_tag;
  /** @brief Publisher sending messages to RMQ server */
  std::shared_ptr<RMQPublisher> _publisher;
  /** @brief Thread in charge of the publisher */
  std::thread _publisher_thread;
  /** @brief Consumer listening to RMQ and consuming messages */
  std::shared_ptr<RMQConsumer> _consumer;
  /** @brief Thread in charge of the consumer */
  std::thread _consumer_thread;
  /** @brief True if connected to RabbitMQ */
  bool connected;

public:
  RMQInterface() : connected(false), _rId(0) {}

  /**
   * @brief Connect to a RabbitMQ server
   * @param[in] rmq_name The name of the RabbitMQ server
   * @param[in] rmq_name The name of the RabbitMQ server
   * @param[in] rmq_password The password
   * @param[in] rmq_user Username
   * @param[in] rmq_vhost Virtual host (by default RabbitMQ vhost = '/')
   * @param[in] service_port The port number
   * @param[in] service_host URL of RabbitMQ server
   * @param[in] rmq_cert Path to TLS certificate
   * @param[in] outbound_queue Name of the queue on which AMSlib publishes (send) messages
   * @param[in] exchange Exchange for incoming messages
   * @param[in] routing_key Routing key for incoming messages (must match what the AMS Python side is using)
   * @return True if connection succeeded
   */
  bool connect(std::string rmq_name,
               std::string rmq_password,
               std::string rmq_user,
               std::string rmq_vhost,
               int service_port,
               std::string service_host,
               std::string rmq_cert,
               std::string outbound_queue,
               std::string exchange,
               std::string routing_key);

  /**
   * @brief Check if the RabbitMQ connection is connected.
   * @return True if connected
   */
  bool isConnected() const { return connected; }

  /**
   * @brief Set the internal ID of the interface (usually MPI rank).
   * @param[in] id The ID
   */
  void setId(uint64_t id) { _rId = id; }

  /**
   * @brief Try to restart the RabbitMQ publisher (restart the thread managing messages publishing)
   */
  void restartPublisher();

  /**
   * @brief Return the latest model and, by default, delete the corresponding message from the Consumer
   * @param[in] domain_name The name of the domain
   * @param[in] num_elements The number of elements for inputs/outputs
   * @param[in] inputs A vector containing arrays of inputs, each array has num_elements elements
   * @param[in] outputs A vector containing arrays of outputs, each array has num_elements elements
   */
  template <typename TypeValue>
  void publish(std::string& domain_name,
               size_t num_elements,
               std::vector<TypeValue*>& inputs,
               std::vector<TypeValue*>& outputs)
  {
    DBG(RMQInterface,
        "[tag=%d] stores %ld elements of input/output "
        "dimensions (%ld, %ld)",
        _msg_tag,
        num_elements,
        inputs.size(),
        outputs.size())

    AMSMessage msg(_msg_tag, _rId, domain_name, num_elements, inputs, outputs);

    if (!_publisher->connectionValid()) {
      connected = false;
      restartPublisher();
      bool status = _publisher->waitToEstablish(100, 10);
      if (!status) {
        _publisher->stop();
        _publisher_thread.join();
        FATAL(RMQInterface,
              "Could not establish publisher RabbitMQ connection");
      }
      connected = true;
    }
    _publisher->publish(std::move(msg));
    _msg_tag++;
  }

  /**
   * @brief Close the underlying connection
   */
  void close();

  /**
   * @brief Check if a new ML model is available
   * @return True if there is a valid ML model
   */
  bool updateModel()
  {
    // NOTE: The architecture here is not great for now, we have redundant call to getLatestModel
    // Solution: when switching to C++ use std::variant to return an std::optional
    // the std::optional would be a string if a model is available otherwise it's a bool false
    auto data = _consumer->getLatestModel();
    return !std::get<1>(data).empty();
  }

  /**
   * @brief Return the latest model and, by default, delete the corresponding message from the Consumer
   * @param[in] remove_msg if True, delete the message corresponding to the model
   * @return The Path of the new model
   */
  std::string getLatestModel(bool remove_msg = true)
  {
    auto res = _consumer->getLatestModel();
    bool empty = std::get<1>(res).empty();
    if (remove_msg && !empty) {
      auto id = std::get<0>(res);
      _consumer->delMessage(id);
    }
    return std::get<1>(res);
  }

  ~RMQInterface()
  {
    if (connected) close();
  }
};

/* A class that provides a BaseDB interface to AMS workflow.
 * When storing data it pushes the data to the RMQ server asynchronously
*/

class RabbitMQDB final : public BaseDB
{
private:
  /** @brief the application domain that stores the data */
  std::string appDomain;
  /** @brief An interface to RMQ to push the data to */
  RMQInterface& interface;

public:
  RabbitMQDB(const RabbitMQDB&) = delete;
  RabbitMQDB& operator=(const RabbitMQDB&) = delete;

  RabbitMQDB(RMQInterface& interface,
             std::string& domain,
             uint64_t id,
             bool allowModelUpdate)
      : BaseDB(id, allowModelUpdate), appDomain(domain), interface(interface)
  {
    /* We set manually the MPI rank here because when
    * RMQInterface was statically initialized, MPI was not
    * necessarily initialized and ready. So we provide the
    * option of setting the distributed ID afterward.
    * 
    * Note: this ID is encoded into AMSMessage but for
    * logging we use a randomly generated ID to stay
    * consistent over time (some logging could happen
    * before setId is called).
    */
    interface.setId(id);
  }

  /**
   * @brief Takes an input and an output vector each holding 1-D vectors data, and push
   * it onto the libevent buffer. If the underlying connection is not valid anymore, a
   new connection will be set up and unacknowledged messages will be (re) sent.
   * @param[in] num_elements Number of elements of each 1-D vector
   * @param[in] inputs Vector of 1-D vectors containing the inputs to be sent
   * @param[in] outputs Vector of 1-D vectors, each 1-D vectors contains
   * 'num_elements' values to be sent
   * @param[in] predicate (NOT SUPPORTED YET) Series of predicate
   */
  PERFFASPECT()
  void store(size_t num_elements,
             std::vector<double*>& inputs,
             std::vector<double*>& outputs,
             bool* predicate = nullptr) override
  {
    CFATAL(RMQDB,
           predicate != nullptr,
           "RMQ database does not support storing uq-predicates")
    interface.publish(appDomain, num_elements, inputs, outputs);
  }

  void store(size_t num_elements,
             std::vector<float*>& inputs,
             std::vector<float*>& outputs,
             bool* predicate = nullptr) override
  {
    CFATAL(RMQDB,
           predicate != nullptr,
           "RMQ database does not support storing uq-predicates")
    interface.publish(appDomain, num_elements, inputs, outputs);
  }

  /**
   * @brief Return the type of this broker
   * @return The type of the broker
   */
  std::string type() override { return "rabbitmq"; }

  /**
   * @brief Check if the surrogate model can be updated (i.e., if
   * RMQConsumer received a training message)
   * @return True if the model can be updated
   */
  bool updateModel() { return interface.updateModel(); }

  /**
   * @brief Return the path of the latest surrogate model if available
   * @return The path of the latest available surrogate model
   */
  std::string getLatestModel() { return interface.getLatestModel(); }

  /**
   * @brief Return the DB enumerationt type (File, Redis etc)
   */
  AMSDBType dbType() override { return AMSDBType::AMS_RMQ; };

  ~RabbitMQDB() {}
};  // class RabbitMQDB

#else

class RMQInterface
{
  const bool connected;

public:
  RMQInterface() : connected(false) {}
  bool connect()
  {
    FATAL(RMQInterface, "RMQ Disabled yet we are requesting to connect")
    return false;
  }

  bool isConnected() const { return false; }

  void close() {}
};

#endif  // __ENABLE_RMQ__

class FilesystemInterface
{
  std::string dbPath;
  bool connected;

public:
  FilesystemInterface() : connected(false) {}

  bool connect(std::string& path)
  {
    connected = true;
    fs::path Path(path);
    std::error_code ec;

    if (!fs::exists(Path, ec)) {
      THROW(std::runtime_error,
            ("Path: :'" + path + "' does not exist").c_str());
      exit(-1);
    }

    if (ec) {
      THROW(std::runtime_error, ("Error in file:" + ec.message()).c_str());
      exit(-1);
    }

    dbPath = path;

    return true;
  }

  bool isConnected() const { return connected; }
  std::string& path() { return dbPath; }
};


/**
 * @brief Class that manages all DB attached to AMS workflows.
 * Each DB can overload its method close() that will get called by 
 * the DB manager when the last workflow using a DB will be destructed.
 */
class DBManager
{

#ifdef __ENABLE_RMQ__
  friend RabbitMQDB;
#endif

private:
  std::unordered_map<std::string, std::shared_ptr<BaseDB>> db_instances;
  AMSDBType dbType;
  uint64_t rId;
  /** @brief If True, the DB is allowed to update the surrogate model */
  bool updateSurrogate;

  DBManager() : dbType(AMSDBType::AMS_NONE), updateSurrogate(false){};

protected:
  RMQInterface rmq_interface;
  FilesystemInterface fs_interface;

public:
  static auto& getInstance()
  {
    static DBManager instance;
    return instance;
  }

  ~DBManager()
  {
    for (auto& e : db_instances) {
      DBG(DBManager,
          "Closing DB %s (%p) (#client=%lu)",
          e.first.c_str(),
          e.second.get(),
          e.second.use_count() - 1);
      if (e.second.use_count() > 0) e.second->close();
    }

    if (rmq_interface.isConnected()) {
      rmq_interface.close();
    }
  }

  DBManager(const DBManager&) = delete;
  DBManager(DBManager&&) = delete;
  DBManager& operator=(const DBManager&) = delete;
  DBManager& operator=(DBManager&&) = delete;

  bool isInitialized() const
  {
    return fs_interface.isConnected() || rmq_interface.isConnected();
  }

  /**
  * @brief Create an object of the respective database.
  * This should never be used for large scale simulations as txt/csv format will
  * be extremely slow.
  * @param[in] domainName name of the domain model to store data for
  * @param[in] dbLabel filename to store data to (used only for hdf5)
  * @param[in] dbType Type of the database to create
  * @param[in] rId a unique Id for each process taking part in a distributed
  * @param[in] isDebug Whether this db will store both ml and physics predictions with the associated predicate
  * execution (rank-id)
  */
  std::shared_ptr<BaseDB> createDB(std::string& domainName,
                                   std::string& dbLabel,
                                   AMSDBType dbType,
                                   uint64_t rId = 0,
                                   bool isDebug = false)
  {
    CWARNING(DBManager,
             (isDebug && dbType != AMSDBType::AMS_HDF5),
             "Requesting debug database but %d db type does not support it",
             dbType);
#ifdef __ENABLE_DB__
    DBG(DBManager, "Instantiating data base");

    if ((dbType == AMSDBType::AMS_CSV || dbType == AMSDBType::AMS_HDF5) &&
        !fs_interface.isConnected()) {
      THROW(std::runtime_error,
            "File System is not configured, Please specify output directory");
    } else if (dbType == AMSDBType::AMS_RMQ && !rmq_interface.isConnected()) {
      THROW(std::runtime_error, "Rabbit MQ data base is not configured");
    }

    switch (dbType) {
      case AMSDBType::AMS_CSV:
        return std::make_shared<csvDB>(fs_interface.path(), dbLabel, rId);
#ifdef __ENABLE_HDF5__
      case AMSDBType::AMS_HDF5:
        return std::make_shared<hdf5DB>(
            fs_interface.path(), domainName, dbLabel, rId, isDebug);
#endif
#ifdef __ENABLE_RMQ__
      case AMSDBType::AMS_RMQ:
        return std::make_shared<RabbitMQDB>(rmq_interface,
                                            domainName,
                                            rId,
                                            updateSurrogate);
#endif
      default:
        return nullptr;
    }
#endif
    return nullptr;
  }

  /**
  * @brief get a data base object referred by this string.
  * This should never be used for large scale simulations as txt/csv format will
  * be extremely slow.
  * @param[in] domainName name of the domain model to store data for
  * @param[in] dbLabel filename to store data to 
  * @param[in] rId a unique Id for each process taking part in a distributed
  * execution (rank-id)
  */
  std::shared_ptr<BaseDB> getDB(std::string& domainName,
                                std::string& dbLabel,
                                uint64_t rId = 0,
                                bool isDebug = false)
  {
    DBG(DBManager,
        "Requested DB for domain: '%s' Under Name: '%s' DB Configured to "
        "operate with '%s'",
        domainName.c_str(),
        dbLabel.c_str(),
        getDBTypeAsStr(dbType).c_str())

    if (dbType == AMSDBType::AMS_NONE) return nullptr;

    if (dbLabel.empty()) return nullptr;

    std::string key = domainName;
    if (dbType == AMSDBType::AMS_HDF5) key = dbLabel;

    auto db_iter = db_instances.find(std::string(key));
    if (db_iter == db_instances.end()) {
      auto db = createDB(domainName, dbLabel, dbType, rId, isDebug);
      db_instances.insert(std::make_pair(std::string(domainName), db));
      DBG(DBManager,
          "Creating new Database writting to file: %s",
          domainName.c_str());
      return db;
    }

    auto db = db_iter->second;
    // Corner case where creation of the db failed and someone is requesting
    // the same entry point
    if (db == nullptr) {
      return db;
    }

    if (db->dbType() != dbType) {
      THROW(std::runtime_error, "Requesting databases of different types");
    }

    if (db->getId() != rId) {
      THROW(std::runtime_error, "Requesting databases from different ranks");
    }
    DBG(DBManager,
        "Using existing Database writting to file: %s",
        domainName.c_str());

    return db;
  }

  void instantiate_fs_db(AMSDBType type,
                         std::string db_path,
                         bool is_debug = false)
  {
    CWARNING(DBManager,
             isInitialized(),
             "Data Base is already initialized. Reconfiguring can result "
             "into "
             "issues")

    CWARNING(DBManager,
             dbType != AMSDBType::AMS_NONE,
             "Setting DBManager default DB when already set")
    dbType = type;

    CWARNING(DBManager,
             (is_debug && dbType != AMSDBType::AMS_HDF5),
             "Only HDF5 supports debug")

    if (dbType != AMSDBType::AMS_NONE) fs_interface.connect(db_path);
  }

  void instantiate_rmq_db(int port,
                          std::string& host,
                          std::string& rmq_name,
                          std::string& rmq_pass,
                          std::string& rmq_user,
                          std::string& rmq_vhost,
                          std::string& rmq_cert,
                          std::string& outbound_queue,
                          std::string& exchange,
                          std::string& routing_key,
                          bool update_surrogate)
  {
    fs::path Path(rmq_cert);
    std::error_code ec;
    CWARNING(AMS,
             !fs::exists(Path, ec),
             "Certificate file '%s' for RMQ server does not exist. AMS will "
             "try to connect without it.",
             rmq_cert.c_str());
    dbType = AMSDBType::AMS_RMQ;
    updateSurrogate = update_surrogate;
#ifdef __ENABLE_RMQ__
    rmq_interface.connect(rmq_name,
                          rmq_pass,
                          rmq_user,
                          rmq_vhost,
                          port,
                          host,
                          rmq_cert,
                          outbound_queue,
                          exchange,
                          routing_key);
#else
    FATAL(DBManager,
          "Requsted RMQ database but AMS is not built with such support "
          "enabled")
#endif
  }
};

}  // namespace db
}  // namespace ams

#endif  // __AMS_BASE_DB__
