// Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

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
#include <unordered_map>
#include <algorithm> // for std::remove
#include <pthread.h>
#include <unistd.h>
#include <signal.h>

#include <amqpcpp.h>
#include <amqpcpp/linux_tcp.h>
#include <amqpcpp/libevent.h>
#include <event2/event.h>
#include <event2/event-config.h>
#include <event2/thread.h>
#include <event2/buffer.h>
#include <amqpcpp/libevent.h>
#include <openssl/ssl.h>
#include <openssl/opensslv.h>
#include <openssl/err.h>

#ifdef __cplusplus
extern "C" {
#endif
    #include "base64.h"
#ifdef __cplusplus
}
#endif

#endif // __ENABLE_RMQ__

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
  virtual ~BaseDB() { }

  /**
   * @brief Define the type of the DB (File, Redis etc)
   */
  virtual std::string type() = 0;

  /**
   * Takes an input and an output vector each holding 1-D vectors data, and
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
   * Takes an input and an output vector each holding 1-D vectors data, and
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
    DBG(DB, "DB Type: %s", type())
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
   * Takes an input and an output vector each holding 1-D vectors data, and
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
    DBG(DB, "DB of type %s stores %ld elements of input/output dimensions (%d, %d)", type().c_str(), num_elements, inputs.size(), outputs.size())

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

  /* @brief Writes a single 1-D vector to the dataset
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
   * @brief Takes an input and an output vector each holding 1-D vectors data, and
   * store them into a hdf5 file delimited by ':'. This should never be used for
   * large scale simulations as txt/hdf5 format will be extremely slow.
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

    DBG(DB, "DB of type %s stores %ld elements of input/output dimensions (%d, %d)", type().c_str(), num_elements, inputs.size(), outputs.size())
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

#endif // __ENABLE_HDF5__

#ifdef __ENABLE_RMQ__

typedef std::unordered_map<std::string, std::string> json;

/** \brief Structure that is passed to each worker thread. */
struct worker {
    struct event_base* loop;
    pthread_t id;
};

/** 
 * Worker function responsible of starting the event loop for each thread.
 * @param[in] arg a pointer on a worker structure
 * @return NULL
*/
void* start_worker(void* arg) {
    struct worker* w = (struct worker*) arg;
    event_base_dispatch(w->loop);
    return NULL;
}

/**
* \brief Specific handler for RabbitMQ connections based on libevent.
*/
class RabbitMQHandler : public AMQP::LibEventHandler {
private:
    /** \brief Path to TLS certificate */
    const char* _cacert;
    /** \brief The MPI rank (0 if MPI is not used) */
    int _rank;

public:
    /**
     *  \brief Constructor
     *  @param[in]  loop         Event Loop
     *  @param[in]  cacert       SSL Cacert
     *  @param[in]  rank         MPI rank
     */
    RabbitMQHandler(int rank, struct event_base *loop, std::string cacert) : AMQP::LibEventHandler(loop), _rank(rank), _cacert(cacert.c_str()) {}
    virtual ~RabbitMQHandler() = default;

private:
    /**
     *  \brief Method that is called after a TCP connection has been set up, and right before
     *  the SSL handshake is going to be performed to secure the connection (only for
     *  amqps:// connections). This method can be overridden in user space to load
     *  client side certificates.
     *  @param[in]  connection      The connection for which TLS was just started
     *  @param[in]  ssl             Pointer to the SSL structure that can be modified
     *  @return     bool            True to proceed / accept the connection, false to break up
     */
    virtual bool onSecuring(AMQP::TcpConnection *connection, SSL *ssl) {
        ERR_clear_error();
        unsigned long err;
#if OPENSSL_VERSION_NUMBER < 0x10100000L
        int ret = SSL_use_certificate_file(ssl, _cacert, SSL_FILETYPE_PEM);
#else
        int ret = SSL_use_certificate_chain_file(ssl, _cacert);
#endif
        if (ret != 1) {
            std::string error("openssl: error loading ca-chain from [");
            SSL_get_error(ssl, ret);
            if ((err = ERR_get_error())) {
                error += std::string(ERR_reason_error_string(err));
            }
            error += "]";
            throw std::runtime_error(error);
        } else {
            DBG(RabbitMQHandler, "Success logged with ca-chain %s", _cacert)
            return true;
        }
    }

    /**
     *  \brief Method that is called when the secure TLS connection has been established. 
     *  This is only called for amqps:// connections. It allows you to inspect
     *  whether the connection is secure enough for your liking (you can
     *  for example check the server certificate). The AMQP protocol still has
     *  to be started.
     *  @param[in]  connection      The connection that has been secured
     *  @param[in]  ssl             SSL structure from openssl library
     *  @return     bool            True if connection can be used
     */
    virtual bool onSecured(AMQP::TcpConnection *connection, const SSL *ssl) override {
        DBG(DB, "[rank=%d][ info ] Secured TLS connection has been established", _rank)
        return true;
    }

    /**
     *  \brief Method that is called by the AMQP library when the login attempt
     *  succeeded. After this the connection is ready to use.
     *  @param[in]  connection      The connection that can now be used
     */
    virtual void onReady(AMQP::TcpConnection *connection) override {
        DBG(DB, "[rank=%d][  ok  ] Sucessfuly logged in. Connection ready to use!\n", _rank)
    }

    /**
     *  \brief Method that is called by the AMQP library when a fatal error occurs
     *  on the connection, for example because data received from RabbitMQ
     *  could not be recognized, or the underlying connection is lost. This
     *  call is normally followed by a call to onLost() (if the error occurred
     *  after the TCP connection was established) and onDetached().
     *  @param[in]  connection      The connection on which the error occurred
     *  @param[in]  message         A human readable error message
     */
    virtual void onError(AMQP::TcpConnection *connection, const char *message) override {
        DBG(DB, "[rank=%d] fatal error when establishing TCP connection: %s\n", _rank, message)
    }
}; // class RabbitMQHandler

/**
 * \brief An EventBuffer encapsulates an evbuffer (libevent structure).
 * Each time data is pushed to the underlying evbuffer, the callback will be called.
 */
template <typename TypeValue>
class EventBuffer {
public:
    /**
     *  \brief Constructor
     *  @param[in]  loop
     *  @param[in]  channel
     *  @param[in]  queue
     */
    EventBuffer(int rank, struct event_base *loop, AMQP::TcpChannel *channel, std::string queue) : 
        _rank(rank), _loop(loop), _buffer(nullptr), _byte_to_send(0), _channel(channel), _queue(std::move(queue)) {
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
        // We install signal callbacks
        _sig_exit = SIGUSR1;
        _signal_exit = evsignal_new(_loop, _sig_exit, callback_exit, this);
        event_add(_signal_exit, NULL);
        _signal_term = evsignal_new(_loop, SIGTERM, callback_exit, this);
        event_add(_signal_term, NULL);
    }

    /**
     *  \brief   Return the size of the buffer in bytes.
     *  @return  Buffer size in bytes.
     */
    size_t size() {
        return evbuffer_get_length(_buffer);
    }
    
    /**
     *  \brief   Return True if the buffer is empty.
     *  @return  True if the number of bytes that has to be sent is equals to 0.
     */
    bool is_drained() {
        return _byte_to_send == 0;
    }

    /**
     *  \brief   Push data to the underlying event buffer, which 
     * will trigger the callback.
     *  @return  The number of bytes that has to be sent.
     */
    size_t get_byte_to_send() {
        return _byte_to_send;
    }

    /**
     *  \brief  Push data to the underlying event buffer, which 
     * will trigger the callback.
     *  @param[in]  data            The data pointer
     *  @param[in]  data_size       The number of bytes in the data pointer
     */
    void push(void* data, ssize_t data_size) {
        evbuffer_lock(_buffer);
        evbuffer_add(_buffer, data, data_size);
        _byte_to_send = _byte_to_send + data_size;
        evbuffer_unlock(_buffer);
    }

    /**
     *  \brief  Method to encode a string into base64
     *  @param[in]  input       The input string
     *  @return                 The encoded string
     */
    std::string encode64(const std::string& input) {
        if (input.size() == 0) return "";
        size_t unencoded_length = input.size();
        size_t encoded_length = base64_encoded_length(unencoded_length);
        char *base64_encoded_string = (char *)malloc((encoded_length+1)*sizeof(char));
        ssize_t encoded_size = base64_encode(base64_encoded_string, encoded_length+1, input.c_str(), unencoded_length);
        std::string result(base64_encoded_string);
        free(base64_encoded_string);
        return result;
    }

    /** \brief Destructor */
    ~EventBuffer() {
        evbuffer_free(_buffer);
        event_free(_signal_exit);
        event_free(_signal_term);
    }

private:
    /** \brief Pointer towards the AMQP channel */
    AMQP::TcpChannel *_channel;
    /** \brief Name of the RabbitMQ queue */
    std::string _queue;
    /** \brief Total number of bytes that must be send */
    size_t _byte_to_send;
    /** \brief MPI rank */
    int _rank;
    /** \brief Thread ID */
    pthread_t _tid;
    /** \brief Event loop */
    struct event_base* _loop;
    /** \brief The buffer event structure */
    struct evbuffer* _buffer;
    /** \brief Signal events for exiting properly the loop */
    struct event* _signal_exit;
    struct event* _signal_term;
    /** \brief Custom signal code (by default SIGUSR1) that can be intercepted */
    int _sig_exit;

    /**
     *  \brief  Callback method that is called by libevent when data is being added to the buffer event
     *  @param[in]  fd          The loop in which the event was triggered
     *  @param[in]  event       Internal timer object
     *  @param[in]  argc        The events that triggered this call
     */
    static void callback_commit(struct evbuffer *buffer, const struct evbuffer_cb_info *info, void *arg) {
        EventBuffer *self = static_cast<EventBuffer*>(arg);
        int rank = self->_rank;
        pthread_t tid = self->_tid;

        // we remove only if some byte got added (this callback will get
        // trigger when data is added AND removed from the buffer
        if (info->n_added > 0) {
            // Destination buffer (of TypeValue size, either float or double)
            size_t datlen = info->n_added; // Total number of bytes
            int k = datlen / sizeof(TypeValue);
            if (datlen % sizeof(TypeValue) != 0) k++; // That case should not happen, but that's a safeguard
            TypeValue* data = (TypeValue*)malloc(datlen);

            evbuffer_lock(buffer);
            // Now we drain the evbuffer structure to fill up the destination bufferå
            int nbyte_drained = evbuffer_remove(buffer, data, datlen);
            if (nbyte_drained < 0)
                WARNING(EventBuffer, "evbuffer_remove(): cannot remove data from buffer");
            evbuffer_unlock(buffer);

            std::string result = std::to_string(rank)+":";
            for (int i = 0; i < k-1; i++) {
                result.append(std::to_string(data[i])+":");
            }
            result.append(std::to_string(data[k-1]));
            // For resiliency reasons we encode the result in base64
            std::string result_b64 = self->encode64(result);
            DBG(RabbitMQDB, 
                "#elements (float/double) = %d, stringify size = %d, size in base64 = %d",
                k, result.size(), result_b64.size())
            if (result_b64.size() % 4 != 0) {
                WARNING(EventBuffer, "[rank=%d] Frame size (%d elements)"
                    "cannot be %d more than a multiple of 4!", 
                    rank, result_b64.size(), result_b64.size() % 4)
            }

            // publish the data in the buffer
            self->_channel->startTransaction();
            self->_channel->publish("", self->_queue, result_b64);
            self->_channel->commitTransaction().onSuccess([self, rank, tid, nbyte_drained]() {
                self->_byte_to_send = self->_byte_to_send - nbyte_drained;
            }).onError([self, rank, tid, nbyte_drained](const char *message) {
                WARNING(EventBuffer, "[rank=%d] messages did not get send: %s", rank, message)
                self->_byte_to_send = self->_byte_to_send - nbyte_drained;
            });

            free(data);
        }
    }
    
    /**
     *  \brief Callback method that is called by libevent when the signal sig is intercepted
     *  @param[in]  fd          The loop in which the event was triggered
     *  @param[in]  event       Internal event object (evsignal in this case)
     *  @param[in]  argc        The events that triggered this call
     */
    static void callback_exit(int fd, short event, void* argc) {
        EventBuffer *self = static_cast<EventBuffer*>(argc);
        DBG(RabbitMQHandler, "caught an interrupt signal; exiting cleanly event loop...")
        event_base_loopexit(self->_loop, NULL);
    }
}; // class EventBuffer

/**
* \brief Class that manages a RabbitMQ broker and handles connection, event loop and
* set up various handlers.
*/
template <typename TypeValue>
class RabbitMQDB final : public BaseDB<TypeValue> {
private:
    /** \brief Path of the config file (JSON) */
    std::string _config;
    /** \brief Connection to the broker */
    AMQP::TcpConnection* _connection;
    /** \brief main channel used to communicate with the broker */
    AMQP::TcpChannel* _channel;
    /** \brief Broker address */
    AMQP::Address* _address;
    /** \brief name of the queue */
    std::string _queue_name;
    /** \brief MPI rank (if MPI is used, otherwise 0) */
    int _rank;
    /** \brief The event loop (usually the default one in libevent) */
    struct event_base* _loop;
    /** \brief The handler which contains various callbacks */
    RabbitMQHandler* _handler;
    /** \brief evbuffer that is responsible to offload data to RabbitMQ*/
    EventBuffer<TypeValue>* _evbuffer;
    /** \brief The worker in charge of sending data to the broker (dedicated thread) */
    struct worker* _worker;

    /** 
     * \brief Read a JSON and create a hashmap
     * @param[in] fn Path of the RabbitMQ JSON config file
     * @return a hashmap (std::unordered_map) of the JSON file
    */
    json _read_config(std::string fn) {
        std::ifstream config;
        json connection_info = {
            {"rabbitmq-erlang-cookie", ""},
            {"rabbitmq-name", ""},
            {"rabbitmq-password", ""},
            {"rabbitmq-user", ""},
            {"rabbitmq-vhost", ""},
            {"rabbitmq-port", ""},
            {"rabbitmq-host", ""},
            {"rabbitmq-cert", ""},
            {"rabbitmq-queue-data", ""},
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
        } 
        else {
            std::string err = "Could not open JSON file: " + fn;
            throw std::runtime_error(err);
        }
        return connection_info;
    }

public:
    RabbitMQDB(const RabbitMQDB&) = delete;
    RabbitMQDB& operator=(const RabbitMQDB&) = delete;

    RabbitMQDB(char* config, uint64_t id) : BaseDB<TypeValue>(id),
            _rank(0), _handler(nullptr), _evbuffer(nullptr), _address(nullptr), _worker(nullptr) {
        _config = std::string(config);
        auto rmq_config = _read_config(this->_config);
#ifdef __ENABLE_MPI__
        MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &_rank));
#endif
#ifdef EVTHREAD_USE_PTHREADS_IMPLEMENTED
        evthread_use_pthreads();
#endif
        _loop = event_base_new();
        CDEBUG(RabbitMQDB, _rank == 0, "Libevent %s\n", event_get_version());
        CDEBUG(RabbitMQDB, _rank == 0, "%s (OPENSSL_VERSION_NUMBER = %#010x)\n", 
            OPENSSL_VERSION_TEXT, OPENSSL_VERSION_NUMBER);
#if OPENSSL_VERSION_NUMBER < 0x10100000L
        SSL_library_init();
#else
        OPENSSL_init_ssl(0, NULL);
#endif
        _handler = new RabbitMQHandler(_rank, _loop, rmq_config["rabbitmq-cert"]);
        _queue_name = rmq_config["rabbitmq-queue-data"];

        uint16_t port = std::stoi(rmq_config["rabbitmq-port"]);
        bool is_secure = true;
        AMQP::Login login(rmq_config["rabbitmq-user"], rmq_config["rabbitmq-password"]);

        _address = new AMQP::Address(
            rmq_config["rabbitmq-host"],
            port, login, rmq_config["rabbitmq-vhost"],
            is_secure
        );

        if (_address == nullptr) {
            throw std::runtime_error("something is wrong, address is NULL");
        }

        CINFO(RabbitMQDB, _rank == 0, "RabbitMQ address: %s:%d/%s (queue = %s)", 
                    _address->hostname().c_str(),
                    _address->port(),
                    _address->vhost().c_str(),
                    _queue_name.c_str()
        )

        start(*_address, _queue_name);
        // mandatory to give some time to OpenSSL and RMQ to set things up, otherwise it will fail
        // TODO: find a way to remove that magic sleep and actually check if OpenSSL + RMQ are up and running
        sleep(3);
    }

    /**
     * Initialize the connection with the broker, open a channel and set up a queue.
     * Then it also sets up a worker thread and start its even loop. Now the broker is
     * ready for push operation.
     * @param[in] addr The address of the broker
     * @param[in] queue The name of the queue to declare
     */
    void start(const AMQP::Address& addr, const std::string& queue) {
        _connection = new AMQP::TcpConnection(_handler, addr);
        _channel = new AMQP::TcpChannel(_connection);
        _channel->onError([&_rank = _rank](const char* message) {
            fprintf(stderr, "[ERROR][rank=%d] Error while creating broker channel: %s\n", _rank, message);
            // TODO: throw dedicated excpetion or/and try to recover
            // from it (re-trying to open a queue, testing if the RM server is alive etc)
            throw std::runtime_error(message);
        });

        _channel->declareQueue(queue).onSuccess(
                [queue, &_rank = _rank]
                (const std::string &name, uint32_t messagecount, uint32_t consumercount) 
        {
            if (messagecount > 0 || consumercount > 1) {
                WARNING(RabbitMQDB,
                    "[rank=%d] declared queue: %s (messagecount=%d, consumercount=%d)",
                    _rank, queue.c_str(), messagecount, consumercount)
            }
        }).onError([queue, &_rank = _rank](const char *message) {
            DBG(RabbitMQDB,
                "[ERROR][rank=%d] Error while creating broker queue (%s): %s\n",
                _rank, queue.c_str(), message)
            // TODO: throw dedicated excpetion or/and try to recover
            // from it (re-trying to open a queue, testing if the RM server is alive etc)
            throw std::runtime_error(message);
        });

        _worker = new struct worker;
        _worker->loop = _loop;
        _evbuffer = new EventBuffer<TypeValue>(_rank, _loop, _channel, queue);
        
        if (pthread_create(&_worker->id, NULL, start_worker, _worker)) {
            perror("error pthread_create");
            exit(-1);
        }
    }

    /**
     * Make sure the buffer is being drained.
     * This function blocks until the buffer is empty.
     * @param[in] sleep_time Number of seconds between two checking (active pooling)
     */
    void drain(int sleep_time = 1) {
        if (!(_worker && _evbuffer)) { return; }
        while(true) {
            if (_evbuffer->is_drained()) {
                // pthread_kill(_worker->id, SIGUSR1);
                break;
            }
            sleep(sleep_time);
            // fprintf(stderr,
            //     "[rank=%d][ info ] buffer size = %d, byte_to_send = %d.\n",
            //     _rank, _evbuffer->size(), _evbuffer->get_byte_to_send()
            // );
        }
    }

    // /**
    //  * Takes a data buffer and push it onto the libevent buffer.
    //  * @param[in] num_elements Number of elements of each 1-D vector
    //  * @param[in] data Array containing 'num_elements' values to be sent
    //  */
    // void store(ssize_t num_elements, TypeValue* data) const override {
    //     ssize_t datlen = num_elements * sizeof(TypeValue);
    //     fprintf(stderr, "push() send data event buffer: %d elements, size %d B\n", num_elements, datlen);
    //     _evbuffer->push(static_cast<void*>(data), datlen);
    //     // Necessary for some reasons, other the event buffer overheat and potentially segfault
    //     // TODO: investigate, error -> "[err] buffer.c:1066: Assertion chain || datlen==0 failed in evbuffer_copyout"
    //     sleep(1);
    // }

    /**
     * Takes an input and an output vector each holding 1-D vectors data, and push it onto the libevent buffer.
     * @param[in] num_elements Number of elements of each 1-D vector
     * @param[in] inputs Vector of 1-D vectors containing the inputs to be sent
     * @param[in] inputs Vector of 1-D vectors, each 1-D vectors contains 'num_elements' values to be sent
     * @param[in] outputs Vector of 1-D vectors, each 1-D vectors contains 'num_elements' values to be sent
     */
    PERFFASPECT()
    void store(size_t num_elements, std::vector<TypeValue*>& inputs, std::vector<TypeValue*>& outputs) override {
        DBG(RabbitMQDB, 
            "RabbitMQDB of type %s stores %ld elements of input/output dimensions (%d, %d)", 
            type().c_str(), num_elements, inputs.size(), outputs.size()
        )
        // TODO: This is likely not efficient for RabbitMQ backend at scale
        //       We could just linearize the whole input+output and do one send per cycle
        ssize_t datlen = num_elements * sizeof(TypeValue);
        for (int i = 0; i < inputs.size(); ++i) {
            _evbuffer->push(static_cast<void*>(inputs[i]), datlen);
            // TODO: investigate that
            // Necessary for some reasons, other the event buffer overheat
            // potentially segfault, and with CUDA could also lead to packet losses
            usleep(500000); //0.5 seconds
        }
        for (int i = 0; i < outputs.size(); ++i) {
            _evbuffer->push(static_cast<void*>(outputs[i]), datlen);
            usleep(500000);
        }
    }

    /** \brief Return the type of this broker */
    std::string type() override { return "rabbitmq"; }

    ~RabbitMQDB() {
        drain();
        pthread_kill(_worker->id, SIGUSR1);
        _channel->close();
        _connection->close();
        event_base_free(_loop);
        delete _evbuffer;
        delete _worker;
        delete _handler;
        delete _channel;
        delete _address;
        free(_connection);
    }
}; // class RabbitMQDB

#endif // __ENABLE_RMQ__


/**
 * Create an object of the respective database.
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
      break;
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
