/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_BASE_DB__
#define __AMS_BASE_DB__

#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "AMS.h"
#include "wf/debug.h"
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
#include <event2/buffer.h>
#include <event2/event-config.h>
#include <event2/event.h>
#include <event2/thread.h>
#include <openssl/err.h>
#include <openssl/opensslv.h>
#include <openssl/ssl.h>

#include <algorithm>
#include <chrono>
#include <deque>
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
    fd.open(this->fn, std::ios_base::app | std::ios_base::out);
    if (!fd.is_open()) {
      std::cerr << "Cannot open db file: " << this->fn << std::endl;
    }
    DBG(DB, "DB Type: %s", type().c_str())
  }

  /**
   * @brief deconstructs the class and closes the file
   */
  ~csvDB() { fd.close(); }

  /**
   * @brief Define the type of the DB (File, Redis etc)
   */
  std::string type() override { return "csv"; }

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
  ~hdf5DB()
  {
    // HDF5 Automatically closes all opened fds at exit of application.
    //    herr_t err = H5Fclose(HFile);
    //    HDF5_ERROR(err);
  }

  /**
   * @brief Define the type of the DB
   */
  std::string type() override { return "hdf5"; }

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

#endif  // __ENABLE_HDF5__

#ifdef __ENABLE_RMQ__

/** @brief Structure that represents a JSON structure */
typedef std::unordered_map<std::string, std::string> json;
/** @brief Structure that represents a received RabbitMQ message */
typedef std::tuple<std::string, std::string, std::string, uint64_t, bool>
    inbound_msg;

/**
 * @brief Specific handler for RabbitMQ connections based on libevent.
 */
template <typename TypeValue>
class RMQConsumerHandler : public AMQP::LibEventHandler
{
private:
  /** @brief Path to TLS certificate */
  std::string _cacert;
  /** @brief The MPI rank (0 if MPI is not used) */
  int _rank;
  /** @brief LibEvent I/O loop */
  struct event_base* _loop;
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
  RMQConsumerHandler(int rank, struct event_base* loop, std::string cacert, std::string queue)
      : AMQP::LibEventHandler(loop), _loop(loop), _rank(rank), _cacert(std::move(cacert)), _queue(queue), _messages(std::make_shared<std::vector<inbound_msg>>()), _channel(nullptr)
  {
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
    if (ret != 1) {
      std::string error("openssl: error loading ca-chain (" + _cacert + ") + from [");
      SSL_get_error(ssl, ret);
      if ((err = ERR_get_error())) {
        error += std::string(ERR_reason_error_string(err));
      }
      error += "]";
      throw std::runtime_error(error);
    } else {
      DBG(RMQConsumerHandler, "Success logged with ca-chain %s", _cacert.c_str())
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
      FATAL(RMQConsumerHandler,
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
            CFATAL(RMQConsumerHandler, false, "consume operation failed: %s", message);
          };
          // callback operation when a message was received
          auto messageCb = [&](const AMQP::Message& message,
                              uint64_t deliveryTag,
                              bool redelivered) {
            // acknowledge the message
            _channel->ack(deliveryTag);
            std::string msg(message.body(), message.bodySize());
            _messages->push_back(std::make_tuple(std::move(msg),
                                                  message.exchange(),
                                                  message.routingkey(),
                                                  deliveryTag,
                                                  redelivered));
            fprintf(stderr, "TOTO %s\n", msg.c_str());
            DBG(RMQConsumerHandler,
                "message received [tag=%d] : '%s' of size %d B from '%s'/'%s'",
                deliveryTag,
                msg.c_str(),
                message.bodySize(),
                message.exchange().c_str(),
                message.routingkey().c_str())
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
          FATAL(RMQConsumerHandler,
                "[ERROR][rank=%d] Error while creating broker queue (%s): %s",
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
  virtual void onClosed(AMQP::TcpConnection *connection) override 
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
  virtual void onDetached(AMQP::TcpConnection *connection) override 
  {
    //  add your own implementation, like cleanup resources or exit the application
    DBG(RMQConsumerHandler, "[rank=%d] Connection is detached.\n", _rank)
  }
};  // class RMQConsumerHandler

/**
 * @brief Class that manages a RabbitMQ broker and handles connection, event
 * loop and set up various handlers.
 */
template <typename TypeValue>
class RMQConsumer
{
private:
  /** @brief Connection to the broker */
  AMQP::TcpConnection* _connection;
  /** @brief main channel used to send data to the broker */
  std::shared_ptr<AMQP::TcpChannel> _channel;
  /** @brief name of the queue to send data */
  std::string _queue;
  /** @brief TLS certificate file */
  std::string _cacert;
  /** @brief MPI rank (if MPI is used, otherwise 0) */
  int _rank;
  /** @brief The event loop for sender (usually the default one in libevent) */
  struct event_base* _loop;
  /** @brief The handler which contains various callbacks for the sender */
  std::shared_ptr<RMQConsumerHandler<TypeValue>> _handler;
  /** @brief Queue that contains all the messages received on receiver queue */
  std::shared_ptr<std::vector<inbound_msg>> _messages;

public:
  RMQConsumer(const RMQConsumer&) = delete;
  RMQConsumer& operator=(const RMQConsumer&) = delete;

  RMQConsumer(const AMQP::Address& address, std::string cacert, std::string queue)
      : _rank(0), _queue(queue), _cacert(cacert), _handler(nullptr)
  {
#ifdef __ENABLE_MPI__
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &_rank));
#endif
#ifdef EVTHREAD_USE_PTHREADS_IMPLEMENTED
    evthread_use_pthreads();
#endif
    _loop = event_base_new();
    CFATAL(RMQPublisher, !_loop, "libevent loop is NULL")

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

    _handler = std::make_shared<RMQConsumerHandler<TypeValue>>(_rank, _loop,
                                            _cacert,
                                            _queue);
    CFATAL(RMQConsumer, !_handler, "_handler is NULL")

    _connection = new AMQP::TcpConnection(_handler.get(), address);
    CFATAL(RMQConsumer, !_connection, "_connection is NULL")
  }

  void start() {
    // We start the event loop (blocking call)
    event_base_dispatch(_loop);
  }

  void stop() {
    // We stop the event loop
    event_base_loopexit(_loop, NULL);
  }

  /**
   * @brief Return the most recent messages and delete it
   * @return A structure inbound_msg which is a std::tuple (see typedef)
   */
  inbound_msg pop_messages()
  {
    if (!_messages->empty()) {
      inbound_msg msg = _messages->back();
      _messages->pop_back();
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
    if (!_messages->empty()) {
      auto it = std::find_if(_messages->begin(),
                             _messages->end(),
                             [&delivery_tag](const inbound_msg& e) {
                               return std::get<3>(e) == delivery_tag;
                             });
      if (it != _messages->end()) return *it;
    }
    return std::make_tuple("", "", "", -1, false);
  }

  ~RMQConsumer()
  {
    event_base_free(_loop);
    _connection->close(false);
    delete _connection;
  }
};  // class RMQConsumer


/**
 * @brief An EventBuffer encapsulates an evbuffer (libevent structure).
 * Each time data is pushed to the underlying evbuffer, the callback will be
 * called.
 */
template <typename TypeValue>
class EventBuffer
{
private:
  /** @brief AMQP reliable channel (wrapper of a classic channel with added functionalities) */
  std::shared_ptr<AMQP::Reliable<AMQP::Tagger>> _rchannel;
  /** @brief Name of the RabbitMQ queue */
  std::string _queue;
  /** @brief Internal queue of messages to send (data, size in bytes) */
  std::deque<std::tuple<uint8_t*, size_t>> _messages;
  /** @brief MPI rank */
  int _rank;
  /** @brief Thread ID */
  pthread_t _tid;
  /** @brief Event loop */
  struct event_base* _loop;
  /** @brief The buffer event structure */
  struct evbuffer* _buffer;
  /** @brief Internal counter for number of messages acknowledged */
  int _counter_ack;
  /** @brief Internal counter for number of messages negatively acknowledged */
  int _counter_nack;

  /**
   *  @brief  Callback method that is called by libevent when data is being
   * added to the buffer event
   *  @param[in]  fd          The loop in which the event was triggered
   *  @param[in]  event       Internal timer object
   *  @param[in]  argc        The events that triggered this call
   */
  static void callback_commit(struct evbuffer* buffer,
                              const struct evbuffer_cb_info* info,
                              void* arg)
  {
    EventBuffer* self = static_cast<EventBuffer*>(arg);
    // we remove only if some byte got added (this callback will get
    // trigger when data is added AND removed from the buffer
    DBG(EventBuffer,
        "evbuffer_cb_info(lenght=%zu): n_added=%zu B, n_deleted=%zu B, "
        "orig_size=%zu B",
        evbuffer_get_length(buffer),
        info->n_added,
        info->n_deleted,
        info->orig_size)

    if (info->n_added > 0) {
      size_t datlen = info->n_added;  // Total number of bytes
      // We get a blob of bytes here
      auto data = std::make_unique<uint8_t[]>(datlen);
      // This assert is to check that char is 1 bytes
      // as we have to cast uint8_t -> char for AMQP-CPP
      // On the vast majority of machines this should be true
      assert(sizeof(char) == sizeof(uint8_t));

      evbuffer_lock(buffer);
      // Now we drain the evbuffer structure to fill up the destination buffer
      if (evbuffer_remove(buffer, data.get(), datlen) == -1) {
        WARNING(EventBuffer,
                "evbuffer_remove(): cannot remove data from buffer");
      }
      evbuffer_unlock(buffer);
      // publish a message via the reliable-channel
      self->_rchannel
          ->publish("",
                    self->_queue,
                    reinterpret_cast<char*>(data.get()),
                    datlen)
          .onAck([self]() {
            DBG(EventBuffer,
                "[rank=%d] message got acknowledged successfully",
                self->_rank)
            self->_counter_ack++;
          })
          .onNack([self]() {
            self->_counter_nack++;
            WARNING(EventBuffer,
                    "[rank=%d] message received negative acknowledged",
                    self->_rank)
          })
          .onLost([self]() {
            CFATAL(EventBuffer,
                   false,
                   "[rank=%d] message likely got lost",
                   self->_rank)
          })
          .onError([self](const char* message) {
            CFATAL(EventBuffer,
                   false,
                   "[rank=%d] message did not get send: %s",
                   self->_rank,
                   message)
          });
    }
  }

public:
  /**
   *  @brief Constructor
   *  @param[in]  loop        Event loop (Libevent in this case)
   *  @param[in]  channel     AMQP TCP channel
   *  @param[in]  queue       Name of the queue the Event Buffer will publish on
   */
  EventBuffer(int rank,
              struct event_base* loop,
              std::shared_ptr<AMQP::TcpChannel> channel,
              std::string queue)
      : _rank(rank),
        _loop(loop),
        _buffer(nullptr),
        _rchannel(
            std::make_shared<AMQP::Reliable<AMQP::Tagger>>(*channel.get())),
        _queue(std::move(queue)),
        _counter_ack(0),
        _counter_nack(0)
  {
    pthread_t _tid = pthread_self();
    // initialize the libev buff event structure
    _buffer = evbuffer_new();
    evbuffer_add_cb(_buffer, callback_commit, this);
    /**
     * Force all the callbacks on an evbuffer to be run not immediately after
     * the evbuffer is altered, but instead from inside the event loop.
     * Without that, the call to callback() would block the main thread.
     */
    evbuffer_defer_callbacks(_buffer, _loop);
  }

  /**
   *  @brief   Return the size of the buffer in bytes.
   *  @return  Buffer size in bytes.
   */
  size_t size() { return evbuffer_get_length(_buffer); }

  /**
   *  @brief   Check if the evbuffer is empty.
   *  @return  True if the buffer is empty.
   */
  bool is_drained() { return size() == 0; }

  /**
   *  @brief   Callback that will be called by libevent when
   * the memory is no longer referenced by the evbuffer.
   *  @param[in]  data         The data pointer
   *  @param[in]  datalen      Number of bytes
   *  @param[in]  extra        extra argument
   */
  static void cb_cleanup_msg(const void* data, size_t datalen, void* extra)
  {
    ams::ResourceManager::deallocate<uint8_t>(reinterpret_cast<uint8_t*>(
                                                  const_cast<void*>(data)),
                                              AMSResourceType::HOST);
  }

  /**
   *  @brief  Push data to the underlying event buffer, which
   * will trigger the callback.
   *  @param[in]  data            The data pointer
   *  @param[in]  data_size       The number of bytes in the data pointer
   */
  void push(uint8_t* data, size_t data_size)
  {
    evbuffer_lock(_buffer);
    DBG(EventBuffer,
        "[push()] adding %zu B to buffer => "
        "size of evbuffer %zu",
        data_size,
        size())

    if (evbuffer_add_reference(
            _buffer, data, data_size, cb_cleanup_msg, nullptr) == -1) {
      WARNING(EventBuffer, "evbuffer_add(): %s", strerror(errno));
    }
    // _messages.push_back(std::make_tuple(data, data_size));
    evbuffer_unlock(_buffer);
  }

  /** @brief Destructor */
  ~EventBuffer()
  {
    evbuffer_free(_buffer);
  }
};  // class EventBuffer



/**
 * @brief Specific handler for RabbitMQ connections based on libevent.
 */
template <typename TypeValue>
class RMQPublisherHandler : public AMQP::LibEventHandler
{
private:
  /** @brief Path to TLS certificate */
  std::string _cacert;
  /** @brief The MPI rank (0 if MPI is not used) */
  int _rank;
  /** @brief LibEvent I/O loop */
  struct event_base* _loop;
  /** @brief main channel used to send data to the broker */
  std::shared_ptr<AMQP::TcpChannel> _channel;
  /** @brief RabbitMQ queue */
  std::string _queue;
  /** @brief evbuffer that is responsible to offload data to RabbitMQ*/
  EventBuffer<TypeValue>* _evbuffer;

public:
  /**
   *  @brief Constructor
   *  @param[in]  loop         Event Loop
   *  @param[in]  cacert       SSL Cacert
   *  @param[in]  rank         MPI rank
   */
  RMQPublisherHandler(int rank, struct event_base* loop, std::string cacert, std::string queue)
      : AMQP::LibEventHandler(loop), _loop(loop), _rank(rank), _cacert(std::move(cacert)), _queue(queue), _channel(nullptr)
  {
  }

  /**
   *  @brief  Publish data on RMQ queue.
   *  @param[in]  data            The data pointer
   *  @param[in]  data_size       The number of bytes in the data pointer
   */
  void publish(uint8_t* data, size_t data_size) {
    _evbuffer->push(data, data_size);
  }

  /**
   * @brief Make sure the evbuffer is being drained.
   * This function blocks until the buffer is empty.
   * @param[in] sleep_time Number of milliseconds between two active pooling)
   */
  void drain(int sleep_time = 100)
  {
    if (!_evbuffer) return;
    while (true) {
      if (_evbuffer->is_drained()) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }
  }

  ~RMQPublisherHandler() {
    drain();
    delete _evbuffer;
  }

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
      std::string error("openssl: error loading ca-chain (" + _cacert + ") + from [");
      SSL_get_error(ssl, ret);
      if ((err = ERR_get_error())) {
        error += std::string(ERR_reason_error_string(err));
      }
      error += "]";
      throw std::runtime_error(error);
    } else {
      DBG(RMQPublisherHandler, "Success logged with ca-chain %s", _cacert.c_str())
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
      FATAL(RMQPublisherHandler,
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
          DBG(RMQPublisherHandler, "[rank=%d] declared queue: %s", _rank, _queue.c_str())
          _evbuffer = new EventBuffer<TypeValue>(_rank, _loop, _channel, _queue);
        })
        .onError([&](const char* message) {
          FATAL(RMQPublisherHandler,
                "[ERROR][rank=%d] Error while creating broker queue (%s): %s",
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
  virtual void onClosed(AMQP::TcpConnection *connection) override 
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
    DBG(RMQPublisherHandler,
        "[rank=%d] fatal error when establishing TCP connection: %s\n",
        _rank,
        message)
  }

  /**
    *  Final method that is called. This signals that no further calls to your
    *  handler will be made about the connection.
    *  @param  connection      The connection that can be destructed
    */
  virtual void onDetached(AMQP::TcpConnection *connection) override 
  {
    //  add your own implementation, like cleanup resources or exit the application
    DBG(RMQPublisherHandler, "[rank=%d] Connection is detached.\n", _rank)
  }
};  // class RMQPublisherHandler

/**
 * @brief Class that manages a RabbitMQ broker and handles connection, event
 * loop and set up various handlers.
 */
template <typename TypeValue>
class RMQPublisher
{
private:
  /** @brief Connection to the broker */
  AMQP::TcpConnection* _connection;
  /** @brief main channel used to send data to the broker */
  std::shared_ptr<AMQP::TcpChannel> _channel;
  /** @brief name of the queue to send data */
  std::string _queue;
  /** @brief TLS certificate file */
  std::string _cacert;
  /** @brief MPI rank (if MPI is used, otherwise 0) */
  int _rank;
  /** @brief The event loop for sender (usually the default one in libevent) */
  struct event_base* _loop;
  /** @brief The handler which contains various callbacks for the sender */
  std::shared_ptr<RMQPublisherHandler<TypeValue>> _handler;

public:
  RMQPublisher(const RMQPublisher&) = delete;
  RMQPublisher& operator=(const RMQPublisher&) = delete;

  RMQPublisher(const AMQP::Address& address, std::string cacert, std::string queue)
      : _rank(0), _queue(queue), _cacert(cacert), _handler(nullptr)
  {
#ifdef __ENABLE_MPI__
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &_rank));
#endif
#ifdef EVTHREAD_USE_PTHREADS_IMPLEMENTED
    evthread_use_pthreads();
#endif
    _loop = event_base_new();
    CFATAL(RMQPublisher, !_loop, "libevent loop is NULL")

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

    _handler = std::make_shared<RMQPublisherHandler<TypeValue>>(_rank, _loop, _cacert, _queue);
    CFATAL(RMQPublisher, !_handler, "_handler is NULL")

    _connection = new AMQP::TcpConnection(_handler.get(), address);
    CFATAL(RMQPublisher, !_connection, "_connection is NULL")
  }

  void start() {
    // We start the event loop (blocking call)
    event_base_dispatch(_loop);
  }

  void stop() {
    // We stop the event loop (blocking call)
    event_base_loopexit(_loop, NULL);
  }

  /**
   * @brief Check if the underlying RabbitMQ connection is ready and usable
   * @return True if the publisher is ready to publish
   */
  bool ready_publish() {
    return _connection->ready() && _connection->usable();
  }

  /**
   * @brief Wait that the connection is ready (blocking call)
   * @return True if the publisher is ready to publish
   */
  void wait_ready(int ms = 300, int timeout_sec = 10) {
    // We wait for the connection to be ready
    int total_time = 0;
    while(!ready_publish()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(ms));
      DBG(RMQPublisher, "Waiting for connection to be ready...")
      total_time += ms;
      if (total_time > timeout_sec*1000) {
        DBG(RMQPublisher, "Connection timeout")
        break; 
      }
    }
  }

  void publish(uint8_t* data, size_t data_size) {
    // We wait for the connection to be ready
    wait_ready();
    _handler->publish(data, data_size);
  }

  ~RMQPublisher() {
    event_base_free(_loop);
    delete _connection;
  }
};  // class RMQPublisher

/**
 * @brief Class that manages a RabbitMQ broker and handles connection, event
 * loop and set up various handlers.
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
  /** @brief Publisher sending messages to RMQ server */
  std::shared_ptr<RMQPublisher<TypeValue>> _publisher;
  /** @brief Thread in charge of the publisher */
  std::thread _publisher_thread;
  /** @brief Consumer listening to RMQ and consuming messages */
  std::shared_ptr<RMQConsumer<TypeValue>> _consumer;
  /** @brief Thread in charge of the consumer */
  std::thread _consumer_thread;

  /**
   * @brief Read a JSON and create a hashmap
   * @param[in] fn Path of the RabbitMQ JSON config file
   * @return a hashmap (std::unordered_map) of the JSON file
   */
  json _read_config(std::string fn)
  {
    std::ifstream config;
    json connection_info = {
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
      throw std::runtime_error(err);
    }
    return connection_info;
  }

public:
  RabbitMQDB(const RabbitMQDB&) = delete;
  RabbitMQDB& operator=(const RabbitMQDB&) = delete;

  RabbitMQDB(char* config, uint64_t id)
      : BaseDB<TypeValue>(id),
        _rank(0),
        _config(std::string(config)),
        _publisher(nullptr),
        _consumer(nullptr)
  {
    json rmq_config = _read_config(_config);
    _queue_sender =
        rmq_config["rabbitmq-outbound-queue"];  // Queue to send data to
    _queue_receiver =
        rmq_config["rabbitmq-inbound-queue"];  // Queue to receive data from PDS
    bool is_secure = true;

    if (rmq_config["service-port"].empty()) {
      FATAL(RabbitMQDB,
            "service-port is empty, make sure the port number is present in "
            "the JSON configuration")
    }
    if (rmq_config["service-host"].empty()) {
      FATAL(RabbitMQDB,
            "service-host is empty, make sure the host is present in the JSON "
            "configuration")
    }

    uint16_t port = std::stoi(rmq_config["service-port"]);
    if (_queue_sender.empty() || _queue_receiver.empty()) {
      FATAL(RabbitMQDB,
            "Queues are empty, please check your credentials file and make "
            "sure rabbitmq-inbound-queue and rabbitmq-outbound-queue exist")
    }

    AMQP::Login login(rmq_config["rabbitmq-user"],
                      rmq_config["rabbitmq-password"]);
    AMQP::Address address(rmq_config["service-host"],
                                 port,
                                 login,
                                 rmq_config["rabbitmq-vhost"],
                                 is_secure);

    std::string cacert = rmq_config["rabbitmq-cert"];
    _publisher = std::make_shared<RMQPublisher<TypeValue>>(address, cacert, _queue_sender);
    _consumer = std::make_shared<RMQConsumer<TypeValue>>(address, cacert, _queue_receiver);

    _publisher_thread = std::thread([&]() { _publisher->start(); });
    _consumer_thread = std::thread([&]() { _consumer->start(); });
  }

  /**
   * Custom deleter for opaque libevent structure
   * This allows us to use smart pointers with event_base
   */
  struct event_deleter {
    void operator ()(struct event_base* event)
    {
      event_base_free(event);
    }
  };

  /**
   * @brief Fill an empty buffer with a valid header. We encode the header as follow:
   * The header is 12 bytes long and structured as follows:
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
   * 
   * @param[in] data_blob The buffer to fill
   * @param[in] header_size Size of the header in bytes
   * @param[in] num_elements Total number of elements
   * @param[in] input_dim The dimensions of inputs
   * @param[in] output_dim The dimensions of outputs
   * @return The number of bytes in the header or -1 if error
   */
  int create_header_msg(uint8_t* const data_blob,
                        const size_t header_size,
                        const size_t num_elements,
                        const size_t input_dim,
                        const size_t output_dim)
  {
    if (!data_blob) return -1;

    uint16_t mpirank = static_cast<uint16_t>(_rank);
    uint32_t num_elem = static_cast<uint32_t>(num_elements);
    uint16_t num_in = static_cast<uint16_t>(input_dim);
    uint16_t num_out = static_cast<uint16_t>(output_dim);

    size_t current_offset = 0;
    data_blob[current_offset] = static_cast<uint8_t>(
        header_size);  // Header size (should be 12 bytes here)
    current_offset += sizeof(uint8_t);

    data_blob[current_offset] = static_cast<uint8_t>(
        sizeof(TypeValue));  // Data type (should be 1 bytes)
    current_offset += sizeof(uint8_t);

    std::memcpy(data_blob + current_offset,
                &(mpirank),
                sizeof(uint16_t));  // MPI rank (should be 2 bytes)
    current_offset += sizeof(uint16_t);

    std::memcpy(data_blob + current_offset,
                &(num_elem),
                sizeof(uint32_t));  // Num elem (should be 4 bytes)
    current_offset += sizeof(uint32_t);

    std::memcpy(data_blob + current_offset,
                &(num_in),
                sizeof(uint16_t));  // Input dim (should be 2 bytes)
    current_offset += sizeof(uint16_t);

    std::memcpy(data_blob + current_offset,
                &(num_out),
                sizeof(uint16_t));  // Output dim (should be 2 bytes)
    current_offset += sizeof(uint16_t);

    return current_offset;
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
        "%s stores %ld elements of input/output "
        "dimensions (%d, %d)",
        type().c_str(),
        num_elements,
        inputs.size(),
        outputs.size())

    const size_t num_in = inputs.size();
    const size_t num_out = outputs.size();

    const size_t header_size =
        2 * sizeof(uint8_t) + sizeof(uint32_t) + 3 * sizeof(uint16_t);
    const size_t blob_size =
        header_size + (num_elements * (num_in + num_out)) * sizeof(TypeValue);
    // uint8_t* data_blob = static_cast<uint8_t*>(std::malloc(blob_size*sizeof(uint8_t)));
    uint8_t* data_blob =
        ams::ResourceManager::allocate<uint8_t>(blob_size,
                                                AMSResourceType::HOST);

    size_t current_offset = create_header_msg(
        data_blob, header_size, num_elements, num_in, num_out);
    CWARNING(RabbitMQDB,
           current_offset != header_size,
           "message header is likely corrupted (size = %zu / expected = % zu)",
           current_offset,
           header_size);

    for (size_t i = 0; i < num_elements; i++) {
      for (size_t j = 0; j < num_in; j++) {
        std::memcpy(data_blob + current_offset,
                    &(inputs[j][i]),
                    sizeof(TypeValue));
        current_offset +=
            sizeof(TypeValue);  // We move on to the next TypeValue
      }
      for (size_t j = 0; j < num_out; j++) {
        std::memcpy(data_blob + current_offset,
                    &(outputs[j][i]),
                    sizeof(TypeValue));
        current_offset += sizeof(TypeValue);
      }
    }
    CWARNING(RabbitMQDB,
           current_offset != blob_size,
           "message is likely corrupted (size = %zu / expected = % zu)",
           current_offset,
           blob_size);

    _publisher->publish(data_blob, blob_size);

    if (event_get_version_number() <= 0x02001500L) {
      // FIXME: add condition in CMake once old libevent is not installed on Lassen anymore
      // With old libevent version (<=2.0.21-stable)
      // there is a bug in buffer.c:1066:
      //    Assertion chain || datlen==0 failed in evbuffer_copyout
      // To bypass that, we add an artificial sleep
      // so libevent will not buffer multiple messages together.
      // if the bug still persists, increase value of the sleep.
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    ams::ResourceManager::deallocate<uint8_t>(data_blob, AMSResourceType::HOST);
  }

  /**
   * @brief Return the type of this broker
   * @return The type of the broker
   */
  std::string type() override { return "rabbitmq"; }

  ~RabbitMQDB() {
    _publisher->stop();
    _consumer->stop();
    _publisher_thread.join();
    _consumer_thread.join();
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

#endif  // __AMS_BASE_DB__