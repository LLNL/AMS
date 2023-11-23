/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_BASE_DB__
#define __AMS_BASE_DB__


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
#include "resource_manager.hpp"
#include "wf/debug.h"
#include "wf/device.hpp"
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"

namespace fs = std::experimental::filesystem;

#ifdef __ENABLE_REDIS__
#include <sw/redis++/redis++.h>

#include <iomanip>
// TODO: We should comment out "using" in header files as
// it propagates to every other file including this file
#warning Redis is currently not supported/tested
using namespace sw::redis;
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
#include <thread>
#include <tuple>
#include <unordered_map>

#endif  // __ENABLE_RMQ__

/**
 * @brief A simple pure virtual interface to store data in some
 * persistent storage device
 */
template <typename TypeValue>
class BaseDB
{
  /** @brief unique id of the process running this simulation */
  uint64_t id;

public:
  BaseDB(const BaseDB&) = delete;
  BaseDB& operator=(const BaseDB&) = delete;

  BaseDB(uint64_t id) : id(id) {}
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
                     std::vector<TypeValue*>& inputs,
                     std::vector<TypeValue*>& outputs) = 0;

  uint64_t getId() const { return id; }
};

/**
 * @brief A pure virtual interface for data bases storing data using
 * some file format (filesystem DB).
 */
template <typename TypeValue>
class FileDB : public BaseDB<TypeValue>
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
  FileDB(std::string path, const std::string suffix, uint64_t rId)
      : BaseDB<TypeValue>(rId)
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
    std::string dbfn("data_");
    dbfn += std::to_string(rId) + suffix;
    Path /= fs::path(dbfn);
    fn = Path.string();
    DBG(DB, "File System DB writes to file %s", fn.c_str())
  }
};


template <typename TypeValue>
class csvDB final : public FileDB<TypeValue>
{
private:
  /** @brief file descriptor */
  bool writeHeader;
  std::fstream fd;

public:
  csvDB(const csvDB&) = delete;
  csvDB& operator=(const csvDB&) = delete;

  /**
   * @brief constructs the class and opens the file to write to
   * @param[in] fn Name of the file to store data to
   * @param[in] rId a unique Id for each process taking part in a distributed
   * execution (rank-id)
   */
  csvDB(std::string path, uint64_t rId) : FileDB<TypeValue>(path, ".csv", rId)
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

  /**
   * @brief Define the type of the DB (File, Redis etc)
   */
  std::string type() override { return "csv"; }

  /**
   * @brief Return the DB enumerationt type (File, Redis etc)
   */
  AMSDBType dbType() { return AMSDBType::CSV; };

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
  PERFFASPECT()
  virtual void store(size_t num_elements,
                     std::vector<TypeValue*>& inputs,
                     std::vector<TypeValue*>& outputs) override
  {
    DBG(DB,
        "DB of type %s stores %ld elements of input/output dimensions (%d, %d)",
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
};

#ifdef __ENABLE_HDF5__

template <typename TypeValue>
class hdf5DB final : public FileDB<TypeValue>
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

  /** @brief HDF5 associated data type with specific TypeValue type   */
  hid_t HDType;

  /** @brief create or get existing hdf5 dataset with the provided name
   * storing data as Ckunked pieces. The Chunk value controls the chunking
   * performed by HDF5 and thus controls the write performance
   * @param[in] group in which we will store data under
   * @param[in] dName name of the data set
   * @param[in] Chunk chunk size of dataset used by HDF5.
   * @reval dataset HDF5 key value
   */
  hid_t getDataSet(hid_t group,
                   std::string dName,
                   const size_t Chunk = 32L * 1024L * 1024L)
  {
    // Our datasets a.t.m are 1-D vectors
    const int nDims = 1;
    // We always start from 0
    hsize_t dims = 0;
    hid_t dset = -1;

    int exists = H5Lexists(group, dName.c_str(), H5P_DEFAULT);

    if (exists > 0) {
      dset = H5Dopen(group, dName.c_str(), H5P_DEFAULT);
      HDF5_ERROR(dset);
      // We are assuming symmetrical data sets a.t.m
      if (totalElements == 0) {
        hid_t dspace = H5Dget_space(dset);
        const int ndims = H5Sget_simple_extent_ndims(dspace);
        hsize_t dims[ndims];
        H5Sget_simple_extent_dims(dspace, dims, NULL);
        totalElements = dims[0];
      }
      return dset;
    } else {
      // We will extend the data-set size, so we use unlimited option
      hsize_t maxDims = H5S_UNLIMITED;
      hid_t fileSpace = H5Screate_simple(nDims, &dims, &maxDims);
      HDF5_ERROR(fileSpace);

      hid_t pList = H5Pcreate(H5P_DATASET_CREATE);
      HDF5_ERROR(pList);

      herr_t ec = H5Pset_layout(pList, H5D_CHUNKED);
      HDF5_ERROR(ec);

      // cDims impacts performance considerably.
      // TODO: Align this with the caching mechanism for this option to work
      // out.
      hsize_t cDims = Chunk;
      H5Pset_chunk(pList, nDims, &cDims);
      dset = H5Dcreate(group,
                       dName.c_str(),
                       HDType,
                       fileSpace,
                       H5P_DEFAULT,
                       pList,
                       H5P_DEFAULT);
      HDF5_ERROR(dset);
      H5Sclose(fileSpace);
      H5Pclose(pList);
    }
    return dset;
  }

  /**
   * @brief Create the HDF5 datasets and store their descriptors in the in/out
   * vectors
   * @param[in] num_elements of every vector
   * @param[in] numIn number of input 1-D vectors
   * @param[in] numOut number of output 1-D vectors
   */
  void createDataSets(size_t numElements,
                      const size_t numIn,
                      const size_t numOut)
  {
    for (int i = 0; i < numIn; i++) {
      hid_t dSet = getDataSet(HFile, std::string("input_") + std::to_string(i));
      HDIsets.push_back(dSet);
    }

    for (int i = 0; i < numOut; i++) {
      hid_t dSet =
          getDataSet(HFile, std::string("output_") + std::to_string(i));
      HDOsets.push_back(dSet);
    }
  }

  /**
   * @brief Write all the data in the vectors in the respective datasets.
   * @param[in] dsets Vector containing the hdf5-dataset descriptor for every
   * vector to be written
   * @param[in] data vectors containing 1-D vectors of numElements values each
   * to be written in the db.
   * @param[in] numElements The number of elements each vector has
   */
  void writeDataToDataset(std::vector<hid_t>& dsets,
                          std::vector<TypeValue*>& data,
                          size_t numElements)
  {
    int index = 0;
    for (auto* I : data) {
      writeVecToDataset(dsets[index++], static_cast<void*>(I), numElements);
    }
  }

  /** @brief Writes a single 1-D vector to the dataset
   * @param[in] dSet the dataset to write the data to
   * @param[in] data the data we need to write
   * @param[in] elements the number of data elements we have
   */
  void writeVecToDataset(hid_t dSet, void* data, size_t elements)
  {
    const int nDims = 1;
    hsize_t dims = elements;
    hsize_t start;
    hsize_t count;
    hid_t memSpace = H5Screate_simple(nDims, &dims, NULL);
    HDF5_ERROR(memSpace);

    dims = totalElements + elements;
    H5Dset_extent(dSet, &dims);

    hid_t fileSpace = H5Dget_space(dSet);
    HDF5_ERROR(fileSpace);

    // Data set starts at offset totalElements
    start = totalElements;
    // And we append additional elements
    count = elements;
    // Select hyperslab
    herr_t err = H5Sselect_hyperslab(
        fileSpace, H5S_SELECT_SET, &start, NULL, &count, NULL);
    HDF5_ERROR(err);

    H5Dwrite(dSet, HDType, memSpace, fileSpace, H5P_DEFAULT, data);
    H5Sclose(fileSpace);
  }

public:
  // Delete copy constructors. We do not want to copy the DB around
  hdf5DB(const hdf5DB&) = delete;
  hdf5DB& operator=(const hdf5DB&) = delete;

  /**
   * @brief constructs the class and opens the hdf5 file to write to
   * @param[in] fn Name of the file to store data to
   * @param[in] rId a unique Id for each process taking part in a distributed
   * execution (rank-id)
   */
  hdf5DB(std::string path, uint64_t rId) : FileDB<TypeValue>(path, ".h5", rId)
  {
    if (isDouble<TypeValue>::default_value())
      HDType = H5T_NATIVE_DOUBLE;
    else
      HDType = H5T_NATIVE_FLOAT;
    std::error_code ec;
    bool exists = fs::exists(this->fn);
    this->checkError(ec);

    if (exists)
      HFile = H5Fopen(this->fn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    else
      HFile =
          H5Fcreate(this->fn.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    HDF5_ERROR(HFile);
    totalElements = 0;
  }

  /**
   * @brief deconstructs the class and closes the file
   */
  ~hdf5DB(){
      DBG(DB, "Closing File: %s %s", type().c_str(), this->fn.c_str())
      // HDF5 Automatically closes all opened fds at exit of application.
      //    herr_t err = H5Fclose(HFile);
      //    HDF5_ERROR(err);
  }

  /**
   * @brief Define the type of the DB
   */
  std::string type() override
  {
    return "hdf5";
  }

  /**
   * @brief Return the DB enumerationt type (File, Redis etc)
   */
  AMSDBType dbType() { return AMSDBType::HDF5; };


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
  PERFFASPECT()
  virtual void store(size_t num_elements,
                     std::vector<TypeValue*>& inputs,
                     std::vector<TypeValue*>& outputs) override
  {

    DBG(DB,
        "DB of type %s stores %ld elements of input/output dimensions (%d, %d)",
        type().c_str(),
        num_elements,
        inputs.size(),
        outputs.size())
    const size_t num_in = inputs.size();
    const size_t num_out = outputs.size();

    if (HDIsets.empty()) {
      createDataSets(num_elements, num_in, num_out);
    }

    if (HDIsets.size() != num_in || HDOsets.size() != num_out) {
      std::cerr << "The data dimensionality is different than the one in the "
                   "DB\n";
      exit(-1);
    }

    writeDataToDataset(HDIsets, inputs, num_elements);
    writeDataToDataset(HDOsets, outputs, num_elements);
    totalElements += num_elements;
  }
};
#endif

#ifdef __ENABLE_REDIS__
template <typename TypeValue>
class RedisDB : public BaseDB<TypeValue>
{
  const std::string _fn;  // path to the file storing the DB access config
  uint64_t _dbid;
  Redis* _redis;
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

    ConnectionOptions connection_options;
    connection_options.type = ConnectionType::TCP;
    connection_options.host = connection_info["host"];
    connection_options.port = std::stoi(connection_info["service-port"]);
    connection_options.password = connection_info["database-password"];
    connection_options.db = 0;  // Optionnal, 0 is the default
    connection_options.tls.enabled =
        true;  // Required to connect to PDS within LC
    connection_options.tls.cacert = connection_info["cert"];

    ConnectionPoolOptions pool_options;
    pool_options.size = 100;  // Pool size, i.e. max number of connections.

    _redis = new Redis(connection_options, pool_options);
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

  PERFFASPECT()
  void store(size_t num_elements,
             std::vector<TypeValue*>& inputs,
             std::vector<TypeValue*>& outputs)
  {

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

/**
  * @brief AMS represents the header as follows:
  * The header is 12 bytes long:
  *   - 1 byte is the size of the header (here 12). Limit max: 255
  *   - 1 byte is the precision (4 for float, 8 for double). Limit max: 255
  *   - 2 bytes are the MPI rank (0 if AMS is not running with MPI). Limit max: 65535
  *   - 4 bytes are the number of elements in the message. Limit max: 2^32 - 1
  *   - 2 bytes are the input dimension. Limit max: 65535
  *   - 2 bytes are the output dimension. Limit max: 65535
  * 
  * |__Header__|__Datatype__|___Rank___|__#elems__|___InDim___|___OutDim___|...real data...|
  * ^          ^            ^          ^          ^           ^            ^               ^
  * |  Byte 1  |   Byte 2   | Byte 3-4 | Byte 4-8 | Byte 8-10 | Byte 10-12 |   Byte 12-X   |
  *
  * where X = datatype * num_element * (InDim + OutDim). Total message size is 12+X. 
  *
  * The data starts at byte 12, ends at byte X.
  * The data is structured as pairs of input/outputs. Let K be the total number of 
  * elements, then we have K pairs of inputs/outputs (either float or double):
  *
  *  |__Header_(12B)__|__Input 1__|__Output 1__|...|__Input_K__|__Output_K__|
  */
struct AMSMsgHeader {
  /** @brief Heaader size (bytes) */
  uint8_t hsize;
  /** @brief Data type size (bytes) */
  uint8_t dtype;
  /** @brief MPI rank */
  uint16_t mpi_rank;
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
               size_t num_elem,
               size_t in_dim,
               size_t out_dim,
               size_t type_size)
      : hsize(static_cast<uint8_t>(AMSMsgHeader::size())),
        dtype(static_cast<uint8_t>(type_size)),
        mpi_rank(static_cast<uint16_t>(mpi_rank)),
        num_elem(static_cast<uint32_t>(num_elem)),
        in_dim(static_cast<uint16_t>(in_dim)),
        out_dim(static_cast<uint16_t>(out_dim))
  {
  }

  /**
   * @brief Return the size of a header in the AMS protocol.
   * @return The size of a message header in AMS (in byte)
   */
  static size_t constexpr size()
  {
    return ((sizeof(hsize) + sizeof(dtype) + sizeof(mpi_rank) +
             sizeof(num_elem) + sizeof(in_dim) + sizeof(out_dim) +
             sizeof(double) - 1) /
            sizeof(double)) *
           sizeof(double);
  }

  /**
   * @brief Fill an empty buffer with a valid header.
   * @param[in] data_blob The buffer to fill
   * @return The number of bytes in the header or 0 if error
   */
  size_t encode(uint8_t* data_blob)
  {
    if (!data_blob) return 0;

    size_t current_offset = 0;
    // Header size (should be 1 bytes)
    data_blob[current_offset] = hsize;
    current_offset += sizeof(hsize);
    // Data type (should be 1 bytes)
    data_blob[current_offset] = dtype;
    current_offset += sizeof(dtype);
    // MPI rank (should be 2 bytes)
    std::memcpy(data_blob + current_offset, &(mpi_rank), sizeof(mpi_rank));
    current_offset += sizeof(mpi_rank);
    // Num elem (should be 4 bytes)
    std::memcpy(data_blob + current_offset, &(num_elem), sizeof(num_elem));
    current_offset += sizeof(num_elem);
    // Input dim (should be 2 bytes)
    std::memcpy(data_blob + current_offset, &(in_dim), sizeof(in_dim));
    current_offset += sizeof(in_dim);
    // Output dim (should be 2 bytes)
    std::memcpy(data_blob + current_offset, &(out_dim), sizeof(out_dim));
    current_offset += sizeof(out_dim);

    return AMSMsgHeader::size();
  }
};


/**
 * @brief Class representing a message for the AMSLib
 */
class AMSMessage
{
private:
  /** @brief message ID */
  int _id;
  /** @brief The MPI rank (0 if MPI is not used) */
  int _rank;
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

public:
  /**
   * @brief Constructor
   * @param[in]  num_elements        Number of elements
   * @param[in]  inputs              Inputs
   * @param[in]  outputs             Outputs
   */
  template <typename TypeValue>
  AMSMessage(int id,
             size_t num_elements,
             const std::vector<TypeValue*>& inputs,
             const std::vector<TypeValue*>& outputs)
      : _id(id),
        _num_elements(num_elements),
        _input_dim(inputs.size()),
        _output_dim(outputs.size()),
        _data(nullptr),
        _total_size(0)
  {
#ifdef __ENABLE_MPI__
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &_rank));
#endif
    AMSMsgHeader header(
        _rank, _num_elements, _input_dim, _output_dim, sizeof(TypeValue));

    _total_size = AMSMsgHeader::size() + getTotalElements() * sizeof(TypeValue);
    _data = ams::ResourceManager::allocate<uint8_t>(_total_size,
                                                    AMSResourceType::HOST);

    size_t current_offset = header.encode(_data);
    current_offset +=
        encode_data(reinterpret_cast<TypeValue*>(_data + current_offset),
                    inputs,
                    outputs);
    DBG(AMSMessage, "Allocated message: %p", _data);
  }

  AMSMessage(const AMSMessage&) = delete;
  AMSMessage& operator=(const AMSMessage&) = delete;

  AMSMessage(AMSMessage&& other) noexcept { *this = std::move(other); }

  AMSMessage& operator=(AMSMessage&& other) noexcept
  {
    DBG(AMSMessage, "Move AMSMessage : %p -- %d", other._data, other._id);
    if (this != &other) {
      _id = other._id;
      _num_elements = other._num_elements;
      _input_dim = other._input_dim;
      _output_dim = other._output_dim;
      _total_size = other._total_size;
      _data = other._data;
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
    size_t offset = 0;
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

  int rank() const { return _rank; }

  /**
   * @brief Return the size in bytes of the underlying binary blob
   * @return Byte size of data pointer
   */
  size_t size() const { return _total_size; }

  ~AMSMessage()
  {
    DBG(AMSMessage, "Destroying message with address %p %d", _data, _id)
  }
};  // class AMSMessage

/** @brief Structure that represents a received RabbitMQ message.
 * - The first field is the message content (body)
 * - The second field is the RMQ exchange from which the message
 *   has been received
 * - The third field is the routing key
 * - The fourth is the delivery tag (ID of the message)
 * - The fifth field is a boolean that indicates if that message
 *   has been redelivered by RMQ.
 */
typedef std::tuple<std::string, std::string, std::string, uint64_t, bool>
    inbound_msg;

/**
 * @brief Specific handler for RabbitMQ connections based on libevent.
 */
class RMQConsumerHandler : public AMQP::LibEventHandler
{
private:
  /** @brief Path to TLS certificate */
  std::string _cacert;
  /** @brief The MPI rank (0 if MPI is not used) */
  int _rank;
  /** @brief LibEvent I/O loop */
  std::shared_ptr<struct event_base> _loop;
  /** @brief main channel used to send data to the broker */
  std::shared_ptr<AMQP::TcpChannel> _channel;
  /** @brief RabbitMQ queue */
  std::string _queue;
  /** @brief Queue that contains all the messages received on receiver queue */
  std::shared_ptr<std::vector<inbound_msg>> _messages;

public:
  /**
   *  @brief Constructor
   *  @param[in]  loop         Event Loop
   *  @param[in]  cacert       SSL Cacert
   *  @param[in]  rank         MPI rank
   */
  RMQConsumerHandler(std::shared_ptr<struct event_base> loop,
                     std::string cacert,
                     std::string queue)
      : AMQP::LibEventHandler(loop.get()),
        _loop(loop),
        _rank(0),
        _cacert(std::move(cacert)),
        _queue(queue),
        _messages(std::make_shared<std::vector<inbound_msg>>()),
        _channel(nullptr)
  {
#ifdef __ENABLE_MPI__
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &_rank));
#endif
  }

  ~RMQConsumerHandler() = default;

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
  virtual bool onSecuring(AMQP::TcpConnection* connection, SSL* ssl)
  {
    ERR_clear_error();
    unsigned long err;
#if OPENSSL_VERSION_NUMBER < 0x10100000L
    int ret = SSL_use_certificate_file(ssl, _cacert.c_str(), SSL_FILETYPE_PEM);
#else
    int ret = SSL_use_certificate_chain_file(ssl, _cacert.c_str());
#endif
    // TODO: with openssl 3.0
    // SSL_set_options(ssl, SSL_OP_IGNORE_UNEXPECTED_EOF);

    if (ret != 1) {
      std::string error("openssl: error loading ca-chain (" + _cacert +
                        ") + from [");
      SSL_get_error(ssl, ret);
      if ((err = ERR_get_error())) {
        error += std::string(ERR_reason_error_string(err));
      }
      error += "]";
      throw std::runtime_error(error);
    } else {
      DBG(RMQConsumerHandler,
          "Success logged with ca-chain %s",
          _cacert.c_str())
      return true;
    }
  }

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
                         const SSL* ssl) override
  {
    DBG(RMQConsumerHandler,
        "[rank=%d] Secured TLS connection has been established.",
        _rank)
    return true;
  }

  /**
   *  @brief Method that is called by the AMQP library when the login attempt
   *  succeeded. After this the connection is ready to use.
   *  @param[in]  connection      The connection that can now be used
   */
  virtual void onReady(AMQP::TcpConnection* connection) override
  {
    DBG(RMQConsumerHandler,
        "[rank=%d] Sucessfuly logged in. Connection ready to use.\n",
        _rank)

    _channel = std::make_shared<AMQP::TcpChannel>(connection);
    _channel->onError([&](const char* message) {
      CFATAL(RMQConsumerHandler,
             false,
             "[rank=%d] Error on channel: %s",
             _rank,
             message)
    });

    _channel->declareQueue(_queue)
        .onSuccess([&](const std::string& name,
                       uint32_t messagecount,
                       uint32_t consumercount) {
          if (messagecount > 0 || consumercount > 1) {
            CWARNING(RMQConsumerHandler,
                     _rank == 0,
                     "[rank=%d] declared queue: %s (messagecount=%d, "
                     "consumercount=%d)",
                     _rank,
                     _queue.c_str(),
                     messagecount,
                     consumercount)
          }
          // We can now install callback functions for when we will consumme messages
          // callback function that is called when the consume operation starts
          auto startCb = [](const std::string& consumertag) {
            DBG(RMQConsumerHandler,
                "consume operation started with tag: %s",
                consumertag.c_str())
          };

          // callback function that is called when the consume operation failed
          auto errorCb = [](const char* message) {
            CFATAL(RMQConsumerHandler,
                   false,
                   "consume operation failed: %s",
                   message);
          };
          // callback operation when a message was received
          auto messageCb = [&](const AMQP::Message& message,
                               uint64_t deliveryTag,
                               bool redelivered) {
            // acknowledge the message
            _channel->ack(deliveryTag);
            std::string msg(message.body(), message.bodySize());
            DBG(RMQConsumerHandler,
                "message received [tag=%d] : '%s' of size %d B from "
                "'%s'/'%s'",
                deliveryTag,
                msg.c_str(),
                message.bodySize(),
                message.exchange().c_str(),
                message.routingkey().c_str())
            _messages->push_back(std::make_tuple(std::move(msg),
                                                 message.exchange(),
                                                 message.routingkey(),
                                                 deliveryTag,
                                                 redelivered));
          };

          /* callback that is called when the consumer is cancelled by RabbitMQ (this
          * only happens in rare situations, for example when someone removes the queue
          * that you are consuming from)
          */
          auto cancelledCb = [](const std::string& consumertag) {
            WARNING(RMQConsumerHandler,
                    "consume operation cancelled by the RabbitMQ server: %s",
                    consumertag.c_str())
          };

          // start consuming from the queue, and install the callbacks
          _channel->consume(_queue)
              .onReceived(messageCb)
              .onSuccess(startCb)
              .onCancelled(cancelledCb)
              .onError(errorCb);
        })
        .onError([&](const char* message) {
          CFATAL(RMQConsumerHandler,
                 false,
                 "[ERROR][rank=%d] Error while creating broker queue (%s): "
                 "%s",
                 _rank,
                 _queue.c_str(),
                 message)
        });
  }

  /**
    *  Method that is called when the AMQP protocol is ended. This is the
    *  counter-part of a call to connection.close() to graceful shutdown
    *  the connection. Note that the TCP connection is at this time still 
    *  active, and you will also receive calls to onLost() and onDetached()
    *  @param  connection      The connection over which the AMQP protocol ended
    */
  virtual void onClosed(AMQP::TcpConnection* connection) override
  {
    DBG(RMQConsumerHandler, "[rank=%d] Connection is closed.\n", _rank)
  }

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
                       const char* message) override
  {
    DBG(RMQConsumerHandler,
        "[rank=%d] fatal error when establishing TCP connection: %s\n",
        _rank,
        message)
  }

  /**
    *  Final method that is called. This signals that no further calls to your
    *  handler will be made about the connection.
    *  @param  connection      The connection that can be destructed
    */
  virtual void onDetached(AMQP::TcpConnection* connection) override
  {
    //  add your own implementation, like cleanup resources or exit the application
    DBG(RMQConsumerHandler, "[rank=%d] Connection is detached.\n", _rank)
  }
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
  /** @brief name of the queue to send data */
  std::string _queue;
  /** @brief TLS certificate file */
  std::string _cacert;
  /** @brief MPI rank (if MPI is used, otherwise 0) */
  int _rank;
  /** @brief The event loop for sender (usually the default one in libevent) */
  std::shared_ptr<struct event_base> _loop;
  /** @brief The handler which contains various callbacks for the sender */
  std::shared_ptr<RMQConsumerHandler> _handler;
  /** @brief Queue that contains all the messages received on receiver queue (messages can be popped in) */
  std::vector<inbound_msg> _messages;

public:
  RMQConsumer(const RMQConsumer&) = delete;
  RMQConsumer& operator=(const RMQConsumer&) = delete;

  RMQConsumer(const AMQP::Address& address,
              std::string cacert,
              std::string queue)
      : _rank(0), _queue(queue), _cacert(cacert), _handler(nullptr)
  {
#ifdef __ENABLE_MPI__
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &_rank));
#endif
#ifdef EVTHREAD_USE_PTHREADS_IMPLEMENTED
    evthread_use_pthreads();
#endif
    CDEBUG(RMQConsumer,
           _rank == 0,
           "Libevent %s (LIBEVENT_VERSION_NUMBER = %#010x)",
           event_get_version(),
           event_get_version_number());
    CDEBUG(RMQConsumer,
           _rank == 0,
           "%s (OPENSSL_VERSION_NUMBER = %#010x)",
           OPENSSL_VERSION_TEXT,
           OPENSSL_VERSION_NUMBER);
#if OPENSSL_VERSION_NUMBER < 0x10100000L
    SSL_library_init();
#else
    OPENSSL_init_ssl(0, NULL);
#endif
    CINFO(RMQConsumer,
          _rank == 0,
          "RabbitMQ address: %s:%d/%s (queue = %s)",
          address.hostname().c_str(),
          address.port(),
          address.vhost().c_str(),
          _queue.c_str())

    _loop = std::shared_ptr<struct event_base>(event_base_new(),
                                               [](struct event_base* event) {
                                                 event_base_free(event);
                                               });
    _handler = std::make_shared<RMQConsumerHandler>(_loop, _cacert, _queue);
    _connection = new AMQP::TcpConnection(_handler.get(), address);
  }

  /**
   * @brief Start the underlying I/O loop (blocking call)
   */
  void start() { event_base_dispatch(_loop.get()); }

  /**
   * @brief Stop the underlying I/O loop
   */
  void stop() { event_base_loopexit(_loop.get(), NULL); }

  /**
   * @brief Return the most recent messages and delete it
   * @return A structure inbound_msg which is a std::tuple (see typedef)
   */
  inbound_msg pop_messages()
  {
    if (!_messages.empty()) {
      inbound_msg msg = _messages.back();
      _messages.pop_back();
      return msg;
    }
    return std::make_tuple("", "", "", -1, false);
  }

  /**
   * @brief Return the message corresponding to the delivery tag. Do not delete the
   * message.
   * @param[in] delivery_tag Delivery tag that will be returned (if found)
   * @return A structure inbound_msg which is a std::tuple (see typedef)
   */
  inbound_msg get_messages(uint64_t delivery_tag)
  {
    if (!_messages.empty()) {
      auto it = std::find_if(_messages.begin(),
                             _messages.end(),
                             [&delivery_tag](const inbound_msg& e) {
                               return std::get<3>(e) == delivery_tag;
                             });
      if (it != _messages.end()) return *it;
    }
    return std::make_tuple("", "", "", -1, false);
  }

  ~RMQConsumer()
  {
    _connection->close(false);
    delete _connection;
  }
};  // class RMQConsumer

/**
 * @brief Specific handler for RabbitMQ connections based on libevent.
 */
class RMQPublisherHandler : public AMQP::LibEventHandler
{
private:
  enum ConnectionStatus { FAILED, CONNECTED, CLOSED };
  /** @brief Path to TLS certificate */
  std::string _cacert;
  /** @brief The MPI rank (0 if MPI is not used) */
  int _rank;
  /** @brief LibEvent I/O loop */
  std::shared_ptr<struct event_base> _loop;
  /** @brief main channel used to send data to the broker */
  std::shared_ptr<AMQP::TcpChannel> _channel;
  /** @brief AMQP reliable channel (wrapper of classic channel with added functionalities) */
  std::shared_ptr<AMQP::Reliable<AMQP::Tagger>> _rchannel;
  /** @brief RabbitMQ queue */
  std::string _queue;
  /** @brief Total number of messages sent */
  int _nb_msg;
  /** @brief Number of messages successfully acknowledged */
  int _nb_msg_ack;

  std::promise<ConnectionStatus> establish_connection;
  std::future<ConnectionStatus> established;

  std::promise<ConnectionStatus> close_connection;
  std::future<ConnectionStatus> closed;

public:
  std::mutex ptr_mutex;
  std::vector<uint8_t*> data_ptrs;

  /**
   *  @brief Constructor
   *  @param[in]  loop         Event Loop
   *  @param[in]  cacert       SSL Cacert
   *  @param[in]  rank         MPI rank
   */
  RMQPublisherHandler(std::shared_ptr<struct event_base> loop,
                      std::string cacert,
                      std::string queue)
      : AMQP::LibEventHandler(loop.get()),
        _loop(loop),
        _rank(0),
        _cacert(std::move(cacert)),
        _queue(queue),
        _nb_msg_ack(0),
        _nb_msg(0),
        _channel(nullptr),
        _rchannel(nullptr)
  {
#ifdef __ENABLE_MPI__
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &_rank));
#endif
    established = establish_connection.get_future();
    closed = close_connection.get_future();
  }

  /**
   *  @brief  Publish data on RMQ queue.
   *  @param[in]  data            The data pointer
   *  @param[in]  data_size       The number of bytes in the data pointer
   */
  void publish(AMSMessage&& msg)
  {
    if (_rchannel) {
      // publish a message via the reliable-channel
      _rchannel
          ->publish("", _queue, reinterpret_cast<char*>(msg.data()), msg.size())
          .onAck([_msg_ptr = msg.data(),
                  &_nb_msg_ack = _nb_msg_ack,
                  rank = msg.rank(),
                  id = msg.id(),
                  &ptr_mutex = ptr_mutex,
                  &data_ptrs = this->data_ptrs]() mutable {
            const std::lock_guard<std::mutex> lock(ptr_mutex);
            DBG(RMQPublisherHandler,
                "[rank=%d] message #%d (Addr:%p) got acknowledged successfully "
                "by "
                "RMQ "
                "server",
                rank,
                id,
                _msg_ptr)
            _nb_msg_ack++;
            data_ptrs.push_back(_msg_ptr);
          })
          .onNack([_msg_ptr = msg.data(),
                   &_nb_msg_ack = _nb_msg_ack,
                   rank = msg.rank(),
                   id = msg.id(),
                   &ptr_mutex = ptr_mutex,
                   &data_ptrs = this->data_ptrs]() mutable {
            const std::lock_guard<std::mutex> lock(ptr_mutex);
            WARNING(RMQPublisherHandler,
                    "[rank=%d] message #%d received negative acknowledged by "
                    "RMQ "
                    "server",
                    rank,
                    id)
            data_ptrs.push_back(_msg_ptr);
          })
          .onLost([_msg_ptr = msg.data(),
                   &_nb_msg_ack = _nb_msg_ack,
                   rank = msg.rank(),
                   id = msg.id(),
                   &ptr_mutex = ptr_mutex,
                   &data_ptrs = this->data_ptrs]() mutable {
            const std::lock_guard<std::mutex> lock(ptr_mutex);
            CFATAL(RMQPublisherHandler,
                   false,
                   "[rank=%d] message #%d likely got lost by RMQ server",
                   rank,
                   id)
            data_ptrs.push_back(_msg_ptr);
          })
          .onError(
              [_msg_ptr = msg.data(),
               &_nb_msg_ack = _nb_msg_ack,
               rank = msg.rank(),
               id = msg.id(),
               &ptr_mutex = ptr_mutex,
               &data_ptrs = this->data_ptrs](const char* err_message) mutable {
                const std::lock_guard<std::mutex> lock(ptr_mutex);
                CFATAL(RMQPublisherHandler,
                       false,
                       "[rank=%d] message #%d did not get send: %s",
                       rank,
                       id,
                       err_message)
                data_ptrs.push_back(_msg_ptr);
              });
    } else {
      WARNING(RMQPublisherHandler,
              "[rank=%d] The reliable channel was not ready for message #%d.",
              _rank,
              _nb_msg)
    }
    _nb_msg++;
  }

  bool waitToEstablish(unsigned ms, int repeat = 1)
  {
    if (waitFuture(established, ms, repeat)) {
      auto status = established.get();
      DBG(RMQPublisherHandler, "Connection Status: %d", status);
      return status == CONNECTED;
    }
    return false;
  }

  bool waitToClose(unsigned ms, int repeat = 1)
  {
    if (waitFuture(closed, ms, repeat)) {
      return closed.get() == CLOSED;
    }
    return false;
  }

  ~RMQPublisherHandler() = default;

  void release_message_buffers()
  {
    const std::lock_guard<std::mutex> lock(ptr_mutex);
    for (auto& dp : data_ptrs) {
      DBG(RMQPublisherHandler, "deallocate address %p", dp)
      ams::ResourceManager::deallocate(dp, AMSResourceType::HOST);
    }
    data_ptrs.erase(data_ptrs.begin(), data_ptrs.end());
  }

  unsigned unacknowledged() const { return _rchannel->unacknowledged(); }

  void flush()
  {
    uint32_t tries = 0;
    while (auto unAck = _rchannel->unacknowledged()) {
      DBG(RMQPublisherHandler,
          "Waiting for %lu messages to be acknowledged",
          unAck);

      if (++tries > 10) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(50 * tries));
    }
  }

  //  void purge()
  //  {
  //    std::promise<bool> purge_queue;
  //    std::future<bool> purged;
  //    purged = purge_queue.get_future();
  //
  //    _channel->purgeQueue(_queue)
  //        .onSuccess([&](uint32_t messageCount) {
  //          DBG(RMQPublisherHandler,
  //              "Sucessfuly purged queue with (%u) remaining messages",
  //              messageCount);
  //          purge_queue.set_value(true);
  //        })
  //        .onError([&](const char* message) {
  //          DBG(RMQPublisherHandler,
  //              "Error '%s' when purging queue %s",
  //              message,
  //              _queue.c_str());
  //          purge_queue.set_value(false);
  //        })
  //        .onFinalize([&]() {
  //          DBG(RMQPublisherHandler, "Finalizing queue %s", _queue.c_str())
  //        });
  //
  //    if (purged.get()) {
  //      DBG(RMQPublisherHandler, "Successfull destruction of RMQ queue");
  //      return;
  //    }
  //
  //    DBG(RMQPublisherHandler, "Non-successfull destruction of RMQ queue");
  //  }


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
  virtual bool onSecuring(AMQP::TcpConnection* connection, SSL* ssl)
  {
    ERR_clear_error();
    unsigned long err;
#if OPENSSL_VERSION_NUMBER < 0x10100000L
    int ret = SSL_use_certificate_file(ssl, _cacert.c_str(), SSL_FILETYPE_PEM);
#else
    int ret = SSL_use_certificate_chain_file(ssl, _cacert.c_str());
#endif
    if (ret != 1) {
      std::string error("openssl: error loading ca-chain (" + _cacert +
                        ") + from [");
      SSL_get_error(ssl, ret);
      if ((err = ERR_get_error())) {
        error += std::string(ERR_reason_error_string(err));
      }
      error += "]";
      establish_connection.set_value(FAILED);
      return false;
    } else {
      DBG(RMQPublisherHandler,
          "Success logged with ca-chain %s",
          _cacert.c_str())
      return true;
    }
  }

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
                         const SSL* ssl) override
  {
    DBG(RMQPublisherHandler,
        "[rank=%d] Secured TLS connection has been established.",
        _rank)
    return true;
  }

  /**
   *  @brief Method that is called by the AMQP library when the login attempt
   *  succeeded. After this the connection is ready to use.
   *  @param[in]  connection      The connection that can now be used
   */
  virtual void onReady(AMQP::TcpConnection* connection) override
  {
    DBG(RMQPublisherHandler,
        "[rank=%d] Sucessfuly logged in. Connection ready to use.\n",
        _rank)

    _channel = std::make_shared<AMQP::TcpChannel>(connection);
    _channel->onError([&](const char* message) {
      CFATAL(RMQPublisherHandler,
             false,
             "[rank=%d] Error on channel: %s",
             _rank,
             message)
    });

    _channel->declareQueue(_queue)
        .onSuccess([&](const std::string& name,
                       uint32_t messagecount,
                       uint32_t consumercount) {
          if (messagecount > 0 || consumercount > 1) {
            CWARNING(RMQPublisherHandler,
                     _rank == 0,
                     "[rank=%d] declared queue: %s (messagecount=%d, "
                     "consumercount=%d)",
                     _rank,
                     _queue.c_str(),
                     messagecount,
                     consumercount)
          }
          // We can now instantiate the shared buffer between AMS and RMQ
          DBG(RMQPublisherHandler,
              "[rank=%d] declared queue: %s",
              _rank,
              _queue.c_str())
          _rchannel =
              std::make_shared<AMQP::Reliable<AMQP::Tagger>>(*_channel.get());
          establish_connection.set_value(CONNECTED);
        })
        .onError([&](const char* message) {
          CFATAL(RMQPublisherHandler,
                 false,
                 "[ERROR][rank=%d] Error while creating broker queue (%s): "
                 "%s",
                 _rank,
                 _queue.c_str(),
                 message)
          establish_connection.set_value(FAILED);
        });
  }

  /**
    *  Method that is called when the AMQP protocol is ended. This is the
    *  counter-part of a call to connection.close() to graceful shutdown
    *  the connection. Note that the TCP connection is at this time still 
    *  active, and you will also receive calls to onLost() and onDetached()
    *  @param  connection      The connection over which the AMQP protocol ended
    */
  virtual void onClosed(AMQP::TcpConnection* connection) override
  {
    DBG(RMQPublisherHandler, "[rank=%d] Connection is closed.\n", _rank)
  }

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
                       const char* message) override
  {
    FATAL(RMQPublisherHandler,
          "[rank=%d] fatal error on TCP connection: %s\n",
          _rank,
          message)
  }

  /**
    *  Final method that is called. This signals that no further calls to your
    *  handler will be made about the connection.
    *  @param  connection      The connection that can be destructed
    */
  virtual void onDetached(AMQP::TcpConnection* connection) override
  {
    //  add your own implementation, like cleanup resources or exit the application
    DBG(RMQPublisherHandler, "[rank=%d] Connection is detached.\n", _rank)
    close_connection.set_value(CLOSED);
  }

  bool waitFuture(std::future<ConnectionStatus>& future,
                  unsigned ms,
                  int repeat)
  {
    std::chrono::milliseconds span(ms);
    int iters = 0;
    std::future_status status;
    while ((status = future.wait_for(span)) == std::future_status::timeout &&
           (iters++ < repeat))
      std::future<ConnectionStatus> established;
    return status == std::future_status::ready;
  }
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

public:
  RMQPublisher(const RMQPublisher&) = delete;
  RMQPublisher& operator=(const RMQPublisher&) = delete;

  RMQPublisher(const AMQP::Address& address,
               std::string cacert,
               std::string queue)
      : _rank(0), _queue(queue), _cacert(cacert), _handler(nullptr)
  {
#ifdef __ENABLE_MPI__
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &_rank));
#endif
#ifdef EVTHREAD_USE_PTHREADS_IMPLEMENTED
    evthread_use_pthreads();
#endif
    CDEBUG(RMQPublisher,
           _rank == 0,
           "Libevent %s (LIBEVENT_VERSION_NUMBER = %#010x)",
           event_get_version(),
           event_get_version_number());
    CDEBUG(RMQPublisher,
           _rank == 0,
           "%s (OPENSSL_VERSION_NUMBER = %#010x)",
           OPENSSL_VERSION_TEXT,
           OPENSSL_VERSION_NUMBER);
#if OPENSSL_VERSION_NUMBER < 0x10100000L
    SSL_library_init();
#else
    OPENSSL_init_ssl(0, NULL);
#endif
    CINFO(RMQPublisher,
          _rank == 0,
          "RabbitMQ address: %s:%d/%s (queue = %s)",
          address.hostname().c_str(),
          address.port(),
          address.vhost().c_str(),
          _queue.c_str())

    _loop = std::shared_ptr<struct event_base>(event_base_new(),
                                               [](struct event_base* event) {
                                                 event_base_free(event);
                                               });

    _handler = std::make_shared<RMQPublisherHandler>(_loop, _cacert, _queue);
    _connection = new AMQP::TcpConnection(_handler.get(), address);
  }

  /**
   * @brief Check if the underlying RabbitMQ connection is ready and usable
   * @return True if the publisher is ready to publish
   */
  bool ready_publish() { return _connection->ready() && _connection->usable(); }

  /**
   * @brief Wait that the connection is ready (blocking call)
   * @return True if the publisher is ready to publish
   */
  bool waitToEstablish(unsigned ms, int repeat = 1)
  {
    return _handler->waitToEstablish(ms, repeat);
  }

  unsigned unacknowledged() const { return _handler->unacknowledged(); }


  /**
   * @brief Start the underlying I/O loop (blocking call)
   */
  void start() { event_base_dispatch(_loop.get()); }

  /**
   * @brief Stop the underlying I/O loop
   */
  void stop() { event_base_loopexit(_loop.get(), NULL); }

  void release_messages() { _handler->release_message_buffers(); }

  void publish(AMSMessage&& message) { _handler->publish(std::move(message)); }

  bool close(unsigned ms, int repeat = 1)
  {
    _handler->flush();
    _connection->close(false);
    return _handler->waitToClose(ms, repeat);
  }

  ~RMQPublisher() {}

};  // class RMQPublisher

/**
 * @brief Class that manages a RabbitMQ broker and handles connection, event
 * loop and set up various handlers.
 * @details This class manages a specific type of database backend in AMSLib.
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
 *    "rabbitmq-inbound-queue": "test4",
 *    "rabbitmq-outbound-queue": "test3"
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
 * 1. Publishing data: When the store() method is being called, it triggers a series of calls:
 *
 *        RabbitMQDB::store() -> RMQPublisher::publish() -> RMQPublisherHandler::publish()
 *
 * Here, RMQPublisherHandler::publish() has access to internal RabbitMQ channels and can publish the message 
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
 * Glabal note: Most calls dealing with RabbitMQ (to establish a RMQ connection, opening a channel, publish data etc)
 * are asynchronous callbacks (similar to asyncio in Python or future in C++).
 * So, the simulation can have already started and the RMQ connection might not be valid which is why most part
 * of the code that deals with RMQ are wrapped into callbacks that will get run only in case of success.
 * For example, we create a channel only if the underlying connection has been succesfuly initiated
 * (see RMQPublisherHandler::onReady()).
 */
template <typename TypeValue>
class RabbitMQDB final : public BaseDB<TypeValue>
{
private:
  /** @brief Path of the config file (JSON) */
  std::string _config;
  /** @brief name of the queue to send data */
  std::string _queue_sender;
  /** @brief name of the queue to receive data */
  std::string _queue_receiver;
  /** @brief MPI rank (if MPI is used, otherwise 0) */
  int _rank;
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

  /**
   * @brief Read a JSON and create a hashmap
   * @param[in] fn Path of the RabbitMQ JSON config file
   * @return a hashmap (std::unordered_map) of the JSON file
   */
  std::unordered_map<std::string, std::string> _read_config(std::string fn)
  {
    std::ifstream config;
    std::unordered_map<std::string, std::string> connection_info = {
        {"rabbitmq-erlang-cookie", ""},
        {"rabbitmq-name", ""},
        {"rabbitmq-password", ""},
        {"rabbitmq-user", ""},
        {"rabbitmq-vhost", ""},
        {"service-port", ""},
        {"service-host", ""},
        {"rabbitmq-cert", ""},
        {"rabbitmq-inbound-queue", ""},
        {"rabbitmq-outbound-queue", ""},
    };

    config.open(fn, std::ifstream::in);

    if (config.is_open()) {
      std::string line;
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
      }
      config.close();
    } else {
      std::string err = "Could not open JSON file: " + fn;
      CFATAL(RabbitMQDB, false, err.c_str());
    }
    return connection_info;
  }

public:
  RabbitMQDB(const RabbitMQDB&) = delete;
  RabbitMQDB& operator=(const RabbitMQDB&) = delete;

  RabbitMQDB(char* config, uint64_t id)
      : BaseDB<TypeValue>(id),
        _rank(0),
        _msg_tag(0),
        _config(std::string(config)),
        _publisher(nullptr),
        _consumer(nullptr)
  {
    std::unordered_map<std::string, std::string> rmq_config =
        _read_config(_config);
    _queue_sender =
        rmq_config["rabbitmq-outbound-queue"];  // Queue to send data to
    _queue_receiver =
        rmq_config["rabbitmq-inbound-queue"];  // Queue to receive data from PDS
    bool is_secure = true;

    if (rmq_config["service-port"].empty()) {
      CFATAL(RabbitMQDB,
             false,
             "service-port is empty, make sure the port number is present in "
             "the JSON configuration")
      return;
    }
    if (rmq_config["service-host"].empty()) {
      CFATAL(RabbitMQDB,
             false,
             "service-host is empty, make sure the host is present in the "
             "JSON "
             "configuration")
      return;
    }

    uint16_t port = std::stoi(rmq_config["service-port"]);
    if (_queue_sender.empty() || _queue_receiver.empty()) {
      CFATAL(RabbitMQDB,
             false,
             "Queues are empty, please check your credentials file and make "
             "sure rabbitmq-inbound-queue and rabbitmq-outbound-queue exist")
      return;
    }

    AMQP::Login login(rmq_config["rabbitmq-user"],
                      rmq_config["rabbitmq-password"]);
    AMQP::Address address(rmq_config["service-host"],
                          port,
                          login,
                          rmq_config["rabbitmq-vhost"],
                          is_secure);

    std::string cacert = rmq_config["rabbitmq-cert"];
    _publisher = std::make_shared<RMQPublisher>(address, cacert, _queue_sender);

    _publisher_thread = std::thread([&]() { _publisher->start(); });

    bool status = _publisher->waitToEstablish(100, 10);
    if (!status) {
      _publisher->stop();
      _publisher_thread.join();
      FATAL(RabbitMQDB, "Could not establish connection");
    }

    //_consumer_thread = std::thread([&]() { _consumer->start(); });
    //_consumer = std::make_shared<RMQConsumer<TypeValue>>(address,
    //                                                     cacert,
    //                                                     _queue_receiver);
  }

  /**
   * @brief Takes an input and an output vector each holding 1-D vectors data, and push
   * it onto the libevent buffer.
   * @param[in] num_elements Number of elements of each 1-D vector
   * @param[in] inputs Vector of 1-D vectors containing the inputs to be sent
   * @param[in] outputs Vector of 1-D vectors, each 1-D vectors contains
   * 'num_elements' values to be sent
   */
  PERFFASPECT()
  void store(size_t num_elements,
             std::vector<TypeValue*>& inputs,
             std::vector<TypeValue*>& outputs) override
  {
    DBG(RabbitMQDB,
        "[tag=%d] %s stores %ld elements of input/output "
        "dimensions (%d, %d)",
        _msg_tag,
        type().c_str(),
        num_elements,
        inputs.size(),
        outputs.size())

    _publisher->release_messages();
    _publisher->publish(AMSMessage(_msg_tag, num_elements, inputs, outputs));
    _msg_tag++;
  }

  /**
   * @brief Return the type of this broker
   * @return The type of the broker
   */
  std::string type() override { return "rabbitmq"; }

  /**
   * @brief Return the DB enumerationt type (File, Redis etc)
   */
  AMSDBType dbType() { return AMSDBType::RMQ; };

  ~RabbitMQDB()
  {

    bool status = _publisher->close(100, 10);
    CWARNING(RabbitMQDB, !status, "Could not gracefully close TCP connection")
    DBG(RabbitMQDB,
        "Number of unacknowledged messages are %d",
        _publisher->unacknowledged())
    _publisher->stop();
    //_publisher->release_messages();
    //_consumer->stop();
    _publisher_thread.join();
    //_consumer_thread.join();
  }
};  // class RabbitMQDB

#endif  // __ENABLE_RMQ__


/**
 * @brief Create an object of the respective database.
 * This should never be used for large scale simulations as txt/csv format will
 * be extremely slow.
 * @param[in] dbPath path to the directory storing the data
 * @param[in] dbType Type of the database to create
 * @param[in] rId a unique Id for each process taking part in a distributed
 * execution (rank-id)
 */
template <typename TypeValue>
BaseDB<TypeValue>* createDB(char* dbPath, AMSDBType dbType, uint64_t rId = 0)
{
  DBG(DB, "Instantiating data base");
#ifdef __ENABLE_DB__
  if (dbPath == nullptr) {
    std::cerr << " [WARNING] Path of DB is NULL, Please provide a valid path "
                 "to enable db\n";
    std::cerr << " [WARNING] Continueing\n";
    return nullptr;
  }

  switch (dbType) {
    case AMSDBType::CSV:
      return new csvDB<TypeValue>(dbPath, rId);
#ifdef __ENABLE_REDIS__
    case AMSDBType::REDIS:
      return new RedisDB<TypeValue>(dbPath, rId);
#endif
#ifdef __ENABLE_HDF5__
    case AMSDBType::HDF5:
      return new hdf5DB<TypeValue>(dbPath, rId);
#endif
#ifdef __ENABLE_RMQ__
    case AMSDBType::RMQ:
      return new RabbitMQDB<TypeValue>(dbPath, rId);
#endif
    default:
      return nullptr;
  }
#else
  return nullptr;
#endif
}


/**
 * @brief get a data base object referred by this string.
 * This should never be used for large scale simulations as txt/csv format will
 * be extremely slow.
 * @param[in] dbPath path to the directory storing the data
 * @param[in] dbType Type of the database to create
 * @param[in] rId a unique Id for each process taking part in a distributed
 * execution (rank-id)
 */
template <typename TypeValue>
std::shared_ptr<BaseDB<TypeValue>> getDB(char* dbPath,
                                         AMSDBType dbType,
                                         uint64_t rId = 0)
{
  static std::unordered_map<std::string, std::shared_ptr<BaseDB<TypeValue>>>
      instances;
  if (dbPath == nullptr) {
    std::cerr << " [WARNING] Path of DB is NULL, Please provide a valid path "
                 "to enable db\n";
    std::cerr << " [WARNING] Continueing\n";
    return nullptr;
  }

  auto db_iter = instances.find(std::string(dbPath));
  if (db_iter == instances.end()) {
    DBG(DB, "Creating new Database writting to file: %s", dbPath);
    std::shared_ptr<BaseDB<TypeValue>> db = std::shared_ptr<BaseDB<TypeValue>>(
        createDB<TypeValue>(dbPath, dbType, rId));
    instances.insert(std::make_pair(std::string(dbPath), db));
    return db;
  }

  auto db = db_iter->second;
  // Corner case where creation of the db failed and someone is requesting
  // the same entry point
  if (db == nullptr) {
    return db;
  }

  if (db->dbType() != dbType) {
    throw std::runtime_error("Requesting databases of different types");
  }

  if (db->getId() != rId) {
    throw std::runtime_error("Requesting databases from different ranks");
  }
  DBG(DB, "Using existing Database writting to file: %s", dbPath);

  return db;
}

#endif  // __AMS_BASE_DB__
