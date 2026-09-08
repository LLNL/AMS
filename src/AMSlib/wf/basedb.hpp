/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef __AMS_BASE_DB__
#define __AMS_BASE_DB__

#include <ATen/core/TensorBody.h>
#include <torch/torch.h>

#include <cstdint>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "AMS.h"
#include "ArrayRef.hpp"
#include "debug.h"
#include "macro.h"
#include "wf/debug.h"
#include "wf/resource_manager.hpp"
#include "wf/utils.hpp"

// Forward declarations for graph types
namespace ams
{
struct AMSHomogeneousGraph;
struct AMSHomogeneousGraphFields;
struct AMSHeterogeneousGraph;
struct AMSHeterogeneousGraphFields;
}  // namespace ams

#ifdef __AMS_ENABLE_HDF5__
#include <H5Ipublic.h>
#include <hdf5.h>
#define HDF5_ERROR(Eid)                                             \
  if (Eid < 0) {                                                    \
    std::cerr << "[Error] Happened in " << __FILE__ << ":"          \
              << __PRETTY_FUNCTION__ << " ( " << __LINE__ << ")\n"; \
    exit(-1);                                                       \
  }
#endif

#ifdef __AMS_ENABLE_CALIPER__
#include <caliper/cali_macros.h>
#endif

#ifdef __AMS_ENABLE_RMQ__
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
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <deque>
#include <future>
#include <list>
#include <queue>
#include <random>
#include <thread>
#include <tuple>

#endif  // __AMS_ENABLE_RMQ__

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
   * @brief Takes an input and an output Tensor.
   * @param[in] num_elements Number of elements of each 1-D vector
   * @param[in] inputs Vector of 1-D vectors containing the inputs to be stored
   * @param[in] inputs Vector of 1-D vectors, each 1-D vectors contains
   * 'num_elements'  values to be stored
   * @param[in] outputs Vector of 1-D vectors, each 1-D vectors contains
   * 'num_elements'  values to be stored
   */

  virtual void store(ArrayRef<torch::Tensor> Inputs,
                     ArrayRef<torch::Tensor> Outputs) = 0;

  /**
   * @brief Store graph data with outputs/targets for training.
   * Default implementation throws - only backends that support graphs override this.
   * @param[in] graph The homogeneous graph containing input features
   * @param[in] outputs The graph fields containing output/target data
   */
  virtual void store(
      [[maybe_unused]] const ams::AMSHomogeneousGraph& graph,
      [[maybe_unused]] const ams::AMSHomogeneousGraphFields& outputs)
  {
    THROW(std::runtime_error,
          (this->type() + " database does not support graph storage").c_str());
  }

  /**
   * @brief Store heterogeneous graph data with outputs/targets for training.
   * Default implementation throws - only backends that support graphs override this.
   * @param[in] graph The heterogeneous graph containing input features
   * @param[in] outputs The graph fields containing output/target data
   */
  virtual void store(
      [[maybe_unused]] const ams::AMSHeterogeneousGraph& graph,
      [[maybe_unused]] const ams::AMSHeterogeneousGraphFields& outputs)
  {
    THROW(std::runtime_error,
          (this->type() + " database does not support heterogeneous graph "
                          "storage")
              .c_str());
  }

  uint64_t getId() const { return id; }

  bool allowModelUpdate() { return allowUpdate; }

  virtual bool updateModel() { return false; }

  virtual std::string getLatestModel() { return {}; }

  virtual std::string getFilename() const { return ""; }
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
    std::experimental::filesystem::path Path(path);
    std::error_code ec;

    if (!std::experimental::filesystem::exists(Path, ec)) {
      std::cerr << "[ERROR]: Path:'" << path << "' does not exist\n";
      exit(-1);
    }

    checkError(ec);

    if (!std::experimental::filesystem::is_directory(Path, ec)) {
      std::cerr << "[ERROR]: Path:'" << path << "' is a file NOT a directory\n";
      exit(-1);
    }

    Path = std::experimental::filesystem::absolute(Path);
    fp = Path.string();

    // We can now create the filename
    std::string dbfn(fn + "_");
    dbfn += std::to_string(rId) + suffix;
    Path /= std::experimental::filesystem::path(dbfn);
    this->fn = std::experimental::filesystem::absolute(Path).string();
    AMS_DBG(DB, "File System DB writes to file {}", this->fn)
  }

  std::string getFilename() const { return fn; }
};


#ifdef __AMS_ENABLE_HDF5__
class hdf5DB final : public FileDB
{
private:
  /** @brief file descriptor */
  hid_t HFile;
  /** @brief The hdf5 dataset descriptor for input data.
   */
  hid_t HDIset;

  /** @brief the hdf5 dataset descriptor for output data.
   */
  hid_t HDOset;

  hid_t HDType;

  ams::SmallVector<hsize_t> currentInputShape;
  ams::SmallVector<hsize_t> currentOutputShape;

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
                   ams::SmallVector<hsize_t>& currentShape,
                   const at::IntArrayRef Shape,
                   hid_t dataType,
                   const size_t Chunk = 1024L);


  /**
   * @brief Create the HDF5 datasets and store their descriptors in the in/out
   * vectors
   * @param[in] num_elements of every vector
   * @param[in] numIn number of input 1-D vectors
   * @param[in] numOut number of output 1-D vectors
   */
  void createDataSets(const at::IntArrayRef InShapes,
                      const at::IntArrayRef OutShapes);

  /**
   * @brief Write all the data in the vectors in the respective datasets.
   * @param[in] dsets Vector containing the hdf5-dataset descriptor for every
   * vector to be written
   * @param[in] data vectors containing 1-D vectors of numElements values each
   * to be written in the db.
   * @param[in] numElements The number of elements each vector has
   */

  void writeDataToDataset(ams::MutableArrayRef<hsize_t> currentShape,
                          hid_t& dset,
                          const at::Tensor& tensor_data);

  PERFFASPECT()
  void _store(const at::Tensor& inputs, const at::Tensor& outputs);

public:
  // Delete copy constructors. We do not want to copy the DB around
  hdf5DB(const hdf5DB&) = delete;
  hdf5DB& operator=(const hdf5DB&) = delete;

  /**
   * @brief constructs the class and opens the hdf5 file to write to
   * @param[in] path path to directory to open/create the file 
   * @param[in] domain_name The 'string' handler of the domain we will store data to.
   * @param[in] rId a unique Id for each process taking part in a distributed
   * execution (rank-id)
   */
  hdf5DB(std::string path, std::string domain_name, uint64_t rId);

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
   * @brief Takes an input and an output tensor each holding data,
   * and stores them into a hdf5 file. 
   * @param[in] inputs Tensor containing the inputs to bestored
   * @param[in] outputs Tensor containing the outputs to bestored
   */
  virtual void store(ArrayRef<torch::Tensor> Inputs,
                     ArrayRef<torch::Tensor> Outputs) override;
};

#endif

#ifdef __AMS_ENABLE_RMQ__

enum class ConnectionStatus { FAILED, CONNECTED, CLOSED, ERROR };

/**
  * @brief AMS represents headers as follows:
  * The header is 12 bytes long:
  *   - 1 byte is the size of the header (here 12). Limit max: 255
  *   - 2 bytes are the MPI rank (0 if AMS is not running with MPI). Limit max: 65535
  *   - 2 bytes to store the size of the MSG domain name. Limit max: 65535
  *   - 2 bytes are the number of input tensors. Limit max: 65535
  *   - 2 bytes are the number of output tensors. Limit max: 65535
  *   - 3 bytes for padding. Limit max: 2^16 - 1
  *
  * |_Header_|___Rank___|_DomainSize_|___InDim__|__OutDim__|____Pad_____|.real data.|
  * ^        ^          ^            ^          ^          ^            ^           ^
  * | Byte 1 | Byte 2-3 |  Byte 4-5  | Byte 6-7 | Byte 8-9 | Byte 10-12 | Byte 12-k |
  *
  * The data starts at byte 12, ends at byte k.
  * The data is structured as pairs of input/outputs. Let K be the total number of 
  * elements, then we have K pairs of inputs/outputs:
  *
  *  |__Header_(12B)__|__Input 1__|__Output 1__|...|__Input_K__|__Output_K__|
  */
struct AMSMsgHeader {
  /** @brief Header size (bytes) */
  uint8_t hsize;
  /** @brief MPI rank */
  uint16_t mpi_rank;
  /** @brief Domain Name Size */
  uint16_t domain_size;
  /** @brief Number of input tensors*/
  uint16_t in_dim;
  /** @brief Number of ouput tensors */
  uint16_t out_dim;

  /**
   * @brief Constructor for AMSMsgHeader
   * @param[in]  mpi_rank     MPI rank
   * @param[in]  domain_size  Size of the MSG domain name
   * @param[in]  in_dim       Inputs dimension
   * @param[in]  out_dim      Outputs dimension
   */
  AMSMsgHeader(size_t mpi_rank,
               size_t domain_size,
               size_t in_dim,
               size_t out_dim);

  /**
   * @brief Constructor for AMSMsgHeader
   * @param[in]  mpi_rank     MPI rank
   * @param[in]  domain_size  Size of the MSG domain name
   * @param[in]  in_dim       Inputs dimension
   * @param[in]  out_dim      Outputs dimension
   */
  AMSMsgHeader(uint16_t mpi_rank,
               uint16_t domain_size,
               uint16_t in_dim,
               uint16_t out_dim);

  /**
   * @brief Return the size of a header in the AMS protocol.
   * @return The size of a message header in AMS (in byte)
   */
  static size_t constexpr size()
  {
    return ((sizeof(hsize) + sizeof(mpi_rank) + sizeof(domain_size) +
             sizeof(in_dim) + sizeof(out_dim) + sizeof(float) - 1) /
            sizeof(float)) *
           sizeof(float);
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

template <typename T>
static inline size_t serialize_data(uint8_t* dest, T src)
{
  uint8_t* ptr = reinterpret_cast<uint8_t*>(&src);
  for (int i = 0; i < sizeof(T); i++) {
    dest[i] = ptr[i];
  }

  return sizeof(T);
}

/**
 * @brief Class representing a message for the AMSLib
 */
class AMSMessage
{
private:
  static size_t computeSerializedSize(const torch::Tensor& tensor)
  {
    // First we need to store how many dimensions this tensor has.
    size_t totalBytes = sizeof(size_t);
    // Next we need to get the required bytes to store both shape and strides.
    totalBytes += tensor.sizes().size() * sizeof(size_t) * 2;
    // Next we need to store the number of bytes of this tensor.
    totalBytes += sizeof(size_t);
    // And finally the size of the data themselves.
    return totalBytes + tensor.nbytes();
  }

  static void serializeTensorHeader(const torch::Tensor& tensor, uint8_t*& blob)
  {
    blob += serialize_data(blob, static_cast<uint64_t>(tensor.sizes().size()));
    blob += serialize_data(blob, static_cast<uint64_t>(tensor.nbytes()));
    for (auto& V : tensor.sizes()) {
      blob += serialize_data(blob, static_cast<uint64_t>(V));
    }
    for (auto& V : tensor.strides()) {
      blob += serialize_data(blob, static_cast<uint64_t>(V));
    }
  }

  static void serializeTensor(const torch::Tensor& tensor, uint8_t*& blob)
  {
    serializeTensorHeader(tensor, blob);
    std::memcpy(blob, tensor.data_ptr(), tensor.nbytes());
    blob += tensor.nbytes();
  }

public:
  /** @brief message ID */
  int _id;
  /** @brief The MPI rank (0 if MPI is not used) */
  uint64_t _rank;
  /** @brief The data represented as a binary blob */
  uint8_t* _data;
  /** @brief The total size of the binary blob in bytes */
  size_t _total_size;
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
  AMSMessage(int id,
             uint64_t rId,
             std::string& domain_name,
             ArrayRef<torch::Tensor> Inputs,
             ArrayRef<torch::Tensor> Outputs)
      : _id(id),
        _rank(rId),
        _input_dim(Inputs.size()),
        _output_dim(Outputs.size()),
        _data(nullptr),
        _total_size(0)
  {
    SmallVector<torch::Tensor> _inputs;
    SmallVector<torch::Tensor> _outputs;
    auto tOptions = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(c10::DeviceType::CPU);

    for (auto& tensor : Inputs)
      _inputs.push_back(tensor.contiguous().to(tOptions));

    for (auto& tensor : Outputs)
      _outputs.push_back(tensor.contiguous().to(tOptions));

    AMSMsgHeader header(_rank, domain_name.size(), _input_dim, _output_dim);

    _total_size = AMSMsgHeader::size() + domain_name.size();

    for (auto& tensor : _inputs)
      _total_size += computeSerializedSize(tensor);
    for (auto& tensor : _outputs)
      _total_size += computeSerializedSize(tensor);

    auto& rm = ams::ResourceManager::getInstance();
    _data = rm.allocate<uint8_t>(_total_size, AMSResourceType::AMS_HOST);

    size_t current_offset = header.encode(_data);
    std::memcpy(&_data[current_offset],
                domain_name.c_str(),
                domain_name.size());
    current_offset += domain_name.size();

    uint8_t* blob = _data + current_offset;


    for (auto& tensor : _inputs)
      serializeTensor(tensor, blob);
    for (auto& tensor : _outputs)
      serializeTensor(tensor, blob);
    AMS_DBG(AMSMessage,
            "Allocated message {}: {} with size: {}",
            _id,
            static_cast<void*>(_data),
            reinterpret_cast<uintptr_t>(blob) -
                reinterpret_cast<uintptr_t>(_data));
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
    AMS_DBG(AMSMessage,
            "Copy AMSMessage ({}, {}) <- ({}, {})",
            _id,
            static_cast<void*>(_data),
            other._id,
            static_cast<void*>(other._data));
    swap(other);
  };

  /**
  * @brief Custom destructor for a shared_ptr
  *        (useful for debugging)
  */
  struct AMSMessageDeleter {
    void operator()(void* x)
    {
      AMS_DBG(AMSMessageDeleter, "Deallocating {}", static_cast<void*>(x))
      auto& rm = ams::ResourceManager::getInstance();
      rm.deallocate(x, AMSResourceType::AMS_HOST);
    }
  };

  static AMSMessageDeleter getDeleter() { return AMSMessageDeleter(); }

  /**
   * @brief Internal Method swapping for AMSMessage
   * @param[in]  other         Message to swap
   */
  void swap(const AMSMessage& other);

  AMSMessage& operator=(const AMSMessage&) = delete;

  AMSMessage(AMSMessage&& other) noexcept { *this = std::move(other); }

  AMSMessage& operator=(AMSMessage&& other) noexcept
  {
    AMS_DBG(AMSMessage,
            "Move AMSMessage ({}, {}) <- ({}, {})",
            _id,
            static_cast<void*>(_data),
            other._id,
            static_cast<void*>(other._data));
    if (this != &other) {
      swap(other);
      other._data = nullptr;
    }
    return *this;
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
 * @brief Structure to hold a publish request.
 * 
 * @note PublishMessage should be a templated class (not needed right now)
 */
struct PublishMessage {
  std::shared_ptr<uint8_t> dPtr;
  size_t size;
  int id;
  PublishMessage() : dPtr(nullptr), size(-1), id(-1) {}
  PublishMessage(std::shared_ptr<uint8_t>& dPtr, size_t size, int id)
      : dPtr(dPtr), size(size), id(id)
  {
  }
  // TODO: implement some move semantics to avoid copying shared_ptr (expensive)
};  // struct PublishMessage

/**
 * @brief A simple thread-safe queue for publishing messages.
 */
class MessageQueue
{
private:
  /** @brief The FIFO queue containing messages */
  std::queue<PublishMessage> _queue;
  /** @brief Mutex */
  std::mutex _mutex;

public:
  MessageQueue() = default;
  MessageQueue(MessageQueue&) = delete;
  MessageQueue& operator=(MessageQueue&) = delete;

  MessageQueue(MessageQueue&&) = delete;
  MessageQueue& operator=(MessageQueue&&) = delete;

  /**
   *  @brief Insert a message in the queue
   *  @param[in]  msg The PublishMessage to push
   */
  void push(const PublishMessage& msg);

  /**
   *  @brief Returns true if a message was popped and populate the argument.
   *  @param[out]  msg The PublishMessage that has been popped
   *  @return True if a message was popped, false otherwise
   */
  bool pop(PublishMessage& msg);

  /**
   *  @brief Return size of the queue
   *  @return The number of messages in the queue
   */
  size_t size();
};  // class MessageQueue

/**
 * @brief A thread safe dictionary to store unacknowledged messages.
 * 
 * @note This class is meant to be used as a singleton.
 */
class MessagesBuffer
{
private:
  /** @brief The hashmap containing the messages */
  std::unordered_map<int, PublishMessage> _msgs;
  /** @brief Mutex */
  std::mutex _mutex;

  MessagesBuffer() = default;

public:
  MessagesBuffer(MessagesBuffer&) = delete;
  MessagesBuffer& operator=(MessagesBuffer&) = delete;

  MessagesBuffer(MessagesBuffer&&) = delete;
  MessagesBuffer& operator=(MessagesBuffer&&) = delete;

  /**
   *  @brief Apply a lambda to each element of the map
   *  @param[in]  lambda The lambda to pass, the lambda takes as 
   *                     input a std::pair<int, PublishMessage>
   */
  template <typename Func>
  void forAll(Func&& lambda)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    std::for_each(_msgs.begin(), _msgs.end(), lambda);
  }

  /**
   *  @brief Insert a message in the underlying data structure
   *  @param[in]  msg The PublishMessage to insert
   *  @return True if the message was not present in the record 
              before being inserted (false otherwise)
   */
  bool insert(const PublishMessage& msg);

  /**
   *  @brief Erase a message from the underlying data structure
   *  @param[in]  id The message ID to erase
   */
  void erase(int id);

  /**
   *  @brief Print on stderr the content of the underlying structure
   */
  void print();

  /**
   *  @brief Return size of the underlying data structure
   *  @return The number of messages in the buffer
   */
  size_t size();

  /**
   *  @brief Return the singleton of MessagesBuffer
   *  @return The singleton instance of MessagesBuffer
   */
  static MessagesBuffer& getInstance()
  {
    static MessagesBuffer instance;
    return instance;
  }
};  // class MessagesBuffer

/**
 * @brief Custom handler for RabbitMQ (AMQP) connections based
 *        on libevent that calls the reconnectCallback when
 *        connections errors.
 */
class AMQPHandler : public AMQP::LibEventHandler
{
private:
  /** @brief Path to TLS certificate */
  std::string _cacert;
  /** @brief Callback when reconnecting, set by the connection manager */
  std::function<void()> reconnectCallback;

  // Needed to set the callback
  friend class ConnectionManagerAMQP;

public:
  /**
   *  @brief Constructor
   *  @param[in]  base         Event Loop
   *  @param[in]  cacert       SSL Cacert
   */
  AMQPHandler(struct event_base* base, const std::string& cacert)
      : AMQP::LibEventHandler(base), _cacert(cacert)
  {
  }

  /**
   *  @brief Default destructor
   */
  ~AMQPHandler() = default;

private:
  /**
    *  @brief Final method that is called. This signals that no further calls to your
    *  handler will be made about the connection.
    *  @param  connection      The connection that can be destructed
    */
  virtual void onDetached(AMQP::TcpConnection* connection);

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
  virtual bool onSecuring(AMQP::TcpConnection* connection, SSL* ssl);

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
  virtual bool onSecured(AMQP::TcpConnection* connection, const SSL* ssl);
  /**
    *  Method that is called when the AMQP protocol is ended. This is the
    *  counter-part of a call to connection.close() to graceful shutdown
    *  the connection. Note that the TCP connection is at this time still 
    *  active, and you will also receive calls to onLost() and onDetached()
    *  @param  connection      The connection over which the AMQP protocol ended
    */
  virtual void onClosed(AMQP::TcpConnection* connection);
  /**
   *  @brief Method that is called by the AMQP library when the login attempt
   *  succeeded. After this the connection is ready to use.
   *  @param[in]  connection      The connection that can now be used
   */
  virtual void onReady(AMQP::TcpConnection* connection) override;
};  // class AMQPHandler


/** 
  TODO: THAT CLASS SHOULD MOVE TO DEDICATED HEADER + CPP FILES
**/
/**
 * @brief Manage the connection to an AMQP broker like RabbitMQ.
 */
class ConnectionManagerAMQP
{
private:
  /** @brief MPI rank (0 if no MPI support) */
  uint64_t _rId;
  /** @brief The event loop for sender (usually the default one in libevent) */
  struct event_base* _base;
  /** @brief Event used to initiate message publishing */
  struct event* _sendEvent;
  /** @brief Event used to flush the queue of messsages */
  struct event* _flushEvent;
  /** @brief Event used to simulate connection drop */
  struct event* _dropConnectionEvent;
  /** @brief Handler using Libevent to send messages */
  std::shared_ptr<AMQPHandler> _handler;
  /** @brief AMQP address */
  AMQP::Address _address;
  /** @brief AMQP connection */
  std::unique_ptr<AMQP::TcpConnection> _connection;
  /** @brief AMQP channel */
  std::shared_ptr<AMQP::TcpChannel> _channel;
  /** @brief AMQP reliable channel (wrapper around channel with aautomatic confirmations) */
  std::shared_ptr<AMQP::Reliable<AMQP::Tagger>> _reliableChannel;
  /** @brief A thread-safe queue to hold messages */
  MessageQueue _msgQueue;
  /** @brief Thread that runs the I/O loop */
  std::thread _workerThread;
  /** @brief True if stopped */
  std::atomic<bool> _stop;
  /** @brief True if currently reconnectiong */
  std::atomic<bool> _reconnecting;
  std::string _queue_sender;
  /** @brief name of the exchange */
  std::string _exchange;
  /** @brief name of the routing binded to exchange */
  std::string _routing_key;
  /** @brief True if connection */
  std::atomic<bool> _isConnected;

  /** @brief Number of messages not acked / nacked */
  std::atomic<int> _nbProcessingMsg;

public:
  ConnectionManagerAMQP(uint64_t id,
                        std::string rmq_user,
                        std::string rmq_password,
                        std::string rmq_vhost,
                        std::string service_host,
                        int service_port,
                        std::string rmq_cert,
                        std::string outbound_queue,
                        std::string exchange,
                        std::string routing_key,
                        bool connectionDrop = false)
      : _rId(id),
        _address(service_host,
                 service_port,
                 AMQP::Login(rmq_user, rmq_password),
                 rmq_vhost,
                 rmq_cert.empty() ? false : true),
        _stop(false),
        _queue_sender(outbound_queue),
        _exchange(exchange),
        _routing_key(routing_key),
        _reconnecting(false),
        _isConnected(false),
        _nbProcessingMsg(0)
  {
#ifdef EVTHREAD_USE_PTHREADS_IMPLEMENTED
    evthread_use_pthreads();
#else
    AMS_WARNING(ConnectionManagerAMQP, "Libevent does not support pthreads")
#endif

    AMS_DBG(ConnectionManagerAMQP,
            "{} (OPENSSL_VERSION_NUMBER = {})",
            OPENSSL_VERSION_TEXT,
            OPENSSL_VERSION_NUMBER);

    AMS_DBG(ConnectionManagerAMQP,
            "[{}] RabbitMQ address: {}:{}/{} (exchange = {} / routing key = "
            "{})",
            _rId,
            _address.hostname(),
            _address.port(),
            _address.vhost(),
            _exchange,
            _routing_key)

    _base = event_base_new();
    _handler = std::make_shared<AMQPHandler>(_base, rmq_cert);
    // Set up the reconnection callback.
    _handler->reconnectCallback =
        std::bind(&ConnectionManagerAMQP::scheduleReconnect, this);

    // The main threads uses the 'sendEvent' mechanism to pick messages
    // from the main-application queue and publish them
    // to the rmq broker.
    _sendEvent = event_new(_base,
                           -1,
                           EV_PERSIST,
                           ConnectionManagerAMQP::sendMessageCallback,
                           this);
    event_add(_sendEvent, nullptr);

    // The main thread uses the '_flushEvent' event to notify the event thread to send 'nack' messages
    // to the rmq broker.

    _flushEvent = event_new(_base,
                            -1,
                            EV_PERSIST,
                            ConnectionManagerAMQP::flushNAckMessageCallback,
                            this);
    event_add(_flushEvent, nullptr);

    // 2000ms timer to simulate connection drops
    struct timeval tv = {2, 0};  // Every 2 seconds
    _dropConnectionEvent =
        event_new(_base,
                  -1,
                  EV_PERSIST,
                  ConnectionManagerAMQP::simulateConnectionDrop,
                  this);

    if (connectionDrop) event_add(_dropConnectionEvent, &tv);

    // Start the worker thread.
    createConnection();
    _workerThread = std::thread([this]() { event_base_dispatch(_base); });
  }

  ~ConnectionManagerAMQP()
  {
    if (!_stop) stop();
    if (_workerThread.joinable()) _workerThread.join();
    if (_dropConnectionEvent) {
      event_free(_dropConnectionEvent);
      _dropConnectionEvent = nullptr;
    }
    if (_sendEvent) {
      event_free(_sendEvent);
      _sendEvent = nullptr;
    }
    if (_flushEvent) {
      event_free(_flushEvent);
      _flushEvent = nullptr;
    }
    if (_base) {
      event_base_free(_base);
      _base = nullptr;
    }
  }

  /**
   *  @brief Check if the connection valid
   *  @return True if connected
   */
  bool isConnected() { return !_reconnecting && _isConnected; }

  /**
   *  @brief Publish a message to the broker
   *  @param[in]  msg  The message to publish
   */
  void publish(const PublishMessage& msg)
  {
    AMS_DBG(ConnectionManagerAMQP,
            "Pushing message #{} ({}) to queue",
            msg.id,
            static_cast<void*>(msg.dPtr.get()))
    _msgQueue.push(msg);
    // Counter tracking how many messages we are supposed to publish
    _nbProcessingMsg++;
    event_active(this->_sendEvent, EV_WRITE, 0);
  }

  /**
   * @brief Close the connection abruptly
   *
   * @note Do not use unless you want to simulate connection drops
   */
  void closeConnection()
  {
    if (_connection) {
      close(_connection->fileno());  // Close the connection
      AMS_DBG(ConnectionManagerAMQP, "Connection closed")
    }
    _isConnected = false;
  }

  /**
   * @brief Simulate a drop in connection (useful for debugging)
   */
  static void simulateConnectionDrop(evutil_socket_t, short, void* arg)
  {
    ConnectionManagerAMQP* mgr = reinterpret_cast<ConnectionManagerAMQP*>(arg);
    AMS_WARNING(ConnectionManagerAMQP, "Simulating connection drop...");
    // Close the current connection (simulate a drop)
    mgr->closeConnection();
  }

  /**
   * @brief Trigger the event for flushing messages if
   *        the buffer holding messages contains entries.
   */
  void flush()
  {
    auto reconnecting = this->_reconnecting.load();
    if (!reconnecting) {
      if (pendingMessages() > 0) event_active(this->_flushEvent, EV_WRITE, 0);
    }
  }

  /**
   *  @brief    Total number of messages unacknowledged
   *  @return   Number of messages unacknowledged
   */
  int unacknowledged() const
  {
    if (_reliableChannel) return _reliableChannel->unacknowledged();
    return 0;
  }

  /**
   * @brief Return the number of pending messages. A pending message
   * is defined as neither ack or nack or a message that has to be 
   * resent because an error happened during a prebious trial.
   */
  int pendingMessages() const
  {
    return MessagesBuffer::getInstance().size() + _nbProcessingMsg.load();
  }

  /**
   * @brief Stops the event loop, and closes the TCP connection.
   */
  void stop()
  {
    AMS_DBG(ConnectionManagerAMQP,
            "Stopping connection: {} messages not processed ({} messages not "
            "acked)",
            pendingMessages(),
            unacknowledged())

    _stop = true;
    _connection->close();
    _isConnected = false;
    event_base_loopexit(_base, nullptr);
  }

private:
  ConnectionManagerAMQP(const ConnectionManagerAMQP&) = delete;
  ConnectionManagerAMQP& operator=(const ConnectionManagerAMQP&) = delete;

  ConnectionManagerAMQP(ConnectionManagerAMQP&&) = delete;
  ConnectionManagerAMQP& operator=(ConnectionManagerAMQP&&) = delete;

  /**
   *  @brief Internal method that publishes a message using the reliable channel
   *  @param[in]  msg  The message to publish
   */
  void internalPublish(const PublishMessage& msg)
  {
    // Publish using the reliable channel if available.
    if (_reliableChannel) {
      _reliableChannel
          ->publish("",
                    _queue_sender,
                    reinterpret_cast<char*>(msg.dPtr.get()),
                    msg.size)
          .onAck([this, msg]() {
            AMS_DBG(ConnectionManagerAMQP,
                    "message #{} ({} / {}) got acknowledged "
                    "successfully ",
                    msg.id,
                    static_cast<void*>(msg.dPtr.get()),
                    msg.size)
            // If msg is in the MessagesBuffer, we erase it
            MessagesBuffer::getInstance().erase(msg.id);
            _nbProcessingMsg--;
          })
          .onNack([this, msg]() {
            AMS_DBG(ConnectionManagerAMQP,
                    "message #{} ({} / {}) received negative "
                    "acknowledgment ",
                    msg.id,
                    static_cast<void*>(msg.dPtr.get()),
                    msg.size)
            if (MessagesBuffer::getInstance().insert(msg)) _nbProcessingMsg++;
          })
          .onError([this, msg](const char* errMsg) {
            AMS_DBG(ConnectionManagerAMQP,
                    "message #{} ({} / {}) did not get send: \"{}\"",
                    msg.id,
                    static_cast<void*>(msg.dPtr.get()),
                    msg.size,
                    errMsg)
            // onNack and onError can be both called by AMQP-CPP
            // We make sure that if the message was not in the buffer
            // we do increment _nbProcessingMsg
            if (MessagesBuffer::getInstance().insert(msg)) _nbProcessingMsg++;
          });
    } else {
      AMS_DBG(ConnectionManagerAMQP,
              "No valid channel for publishing message #{}",
              msg.id);
      MessagesBuffer::getInstance().insert(msg);
    }
  }

  /**
   *  @brief Process the messages in the internal queue
   */
  void processMessages()
  {
    // Publishing the current msgs buffered
    PublishMessage msg;
    while (_msgQueue.size() > 0) {
      if (_msgQueue.pop(msg)) {
        AMS_DBG(ConnectionManagerAMQP, "Processing message: {}", msg.id)
        internalPublish(msg);
      }
    }
  }

  /**
   *  @brief    Try to send the unacknowledged messages
   */
  void flushNAckMessages()
  {
    AMS_DBG(ConnectionManagerAMQP, "Flushing messages")
    auto lambda = [this](const std::pair<int, PublishMessage>& item) {
      this->internalPublish(item.second);
    };
    MessagesBuffer::getInstance().forAll(lambda);
  }

  /**
   *  @brief    The send event callback. This is called when _sendEvent is activated.
   *  @param[in]  arg      A pointer to a ConnectionManagerAMQP
   */
  static void sendMessageCallback(evutil_socket_t, short, void* arg)
  {
    ConnectionManagerAMQP* mgr = reinterpret_cast<ConnectionManagerAMQP*>(arg);
    mgr->processMessages();
  }

  /**
   *  @brief    The flush event callback. This is called when _flushEvent is activated.
   *  @param[in]  arg      A pointer to a ConnectionManagerAMQP
   */
  static void flushNAckMessageCallback(evutil_socket_t, short, void* arg)
  {
    AMS_DBG(ConnectionManagerAMQP, "Sending flush message callback")
    ConnectionManagerAMQP* mgr = reinterpret_cast<ConnectionManagerAMQP*>(arg);
    mgr->flushNAckMessages();
  }

  /**
   *  @brief Create connection, channel, and wrap the channel in a reliable channel.
   */
  void createConnection()
  {
    _connection =
        std::make_unique<AMQP::TcpConnection>(_handler.get(), _address);

    _channel = std::make_shared<AMQP::TcpChannel>(_connection.get());
    _channel->onError([&](const char* message) {
      AMS_WARNING(ConnectionManagerAMQP,
                  "Error on channel: "
                  "{}",
                  message)
      _isConnected = false;
    });

    _channel->declareQueue(_queue_sender)
        .onSuccess([](const std::string& name,
                      uint32_t messagecount,
                      uint32_t consumercount) {
          AMS_DBG(ConnectionManagerAMQP,
                  "declared queue: {} (messagecount={}, "
                  "consumercount={})",
                  name,
                  messagecount,
                  consumercount)
        })
        .onError([&](const char* message) {
          AMS_WARNING(ConnectionManagerAMQP,
                      "Error while creating broker queue: "
                      "{}",
                      message)
          _isConnected = false;
        });
    _isConnected = true;
    _reliableChannel =
        std::make_shared<AMQP::Reliable<AMQP::Tagger>>(*_channel);
  }

  /**
   *  @brief Schedule a reconnect if not already in progress.
   */
  void scheduleReconnect()
  {
    if (_stop) return;
    if (_reconnecting.exchange(true)) return;  // Already reconnecting.
    // TODO: Currently we have no delay. We may at some point implement a back off policy here,
    // in which we increase the wait time by some exponential factor... and once we connect
    // we reset the delay
    struct timeval tv = {0, 0};
    event_base_once(_base,
                    -1,
                    EV_TIMEOUT,
                    ConnectionManagerAMQP::reconnectTimerCallback,
                    this,
                    &tv);
  }

  /**
   *  @brief    Static callback wrapper for the timer.
   *  @param[in]  arg      A pointer to a ConnectionManagerAMQP
   */
  static void reconnectTimerCallback(evutil_socket_t, short, void* arg)
  {
    ConnectionManagerAMQP* mgr = static_cast<ConnectionManagerAMQP*>(arg);
    AMS_DBG(ConnectionManagerAMQP, "Reconnecting ...")
    mgr->reconnect();
    AMS_DBG(ConnectionManagerAMQP, "Reconnection complete")
  }

  /**
   *  @brief Reconnect by closing the old connection and re-creating everything.
   */
  void reconnect()
  {
    if (_connection) _connection->close();
    // Clean up the channels.
    _channel.reset();
    _reliableChannel.reset();
    createConnection();
    _reconnecting = false;
  }
};


/**
 * @brief Class that manages a RabbitMQ broker and handles connection, event
 * loop and set up various handlers.
 * @details This class handles a specific type of database backend in AMSLib.
 * Instead of writing inputs/outputs directly to files (HDF5), we
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
 *    "rabbitmq-queue-physics": "test3",
 *    "rabbitmq-exchange-training": "ams-fanout",
 *    "rabbitmq-key-training": "training"
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
 * on the outbound queue (rabbitmq-queue-physics in the JSON configuration).
 * Note that storing data like that is much faster than with writing files as a call to RabbitMQDB::store()
 * is virtually free, the actual data sending part is taking place in a thread and does not slow down
 * the main simulation (MPI).
 *
 * 2. Consuming data: The exchange and the training key (rabbitmq-exchange-training and rabbitmq-key-training
 * in the JSON configuration) are for incoming data. The RMQConsumer is listening on that exchange/key for messages.
 * In the AMSLib approach, that queue is used to communicate updates to rank regarding the ML surrrogate model.
 * RMQConsumer will automatically populate a std::vector with all messages received since the execution of AMS started.
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
  /** @brief TLS certificate path */
  std::string _cacert;
  /** @brief Represent the ID of the last message sent */
  int _msg_tag;
  /** @brief True if we support surrogate update */
  bool _updateSurrogate;
  /** @brief Object in charge of managing the connection to RabbitMQ */
  std::unique_ptr<ConnectionManagerAMQP> _publishingManager;

public:
  RMQInterface() : _rId(0), _updateSurrogate(false) {}

  /**
   * @brief Connect to a RabbitMQ server
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
   */
  void connect(std::string rmq_user,
               std::string rmq_password,
               std::string rmq_vhost,
               std::string service_host,
               int service_port,
               std::string rmq_cert,
               std::string outbound_queue,
               std::string exchange,
               std::string routing_key,
               bool updateSurrogate)
  {
    bool amsRMQFailure = checkEnvVariable("AMS_SIMULATE_RMQ_FAILURE");
    AMS_CWARNING(RMQInterface, amsRMQFailure, "Simulating connetion drops")

    _publishingManager = std::make_unique<ConnectionManagerAMQP>(_rId,
                                                                 rmq_user,
                                                                 rmq_password,
                                                                 rmq_vhost,
                                                                 service_host,
                                                                 service_port,
                                                                 rmq_cert,
                                                                 outbound_queue,
                                                                 exchange,
                                                                 routing_key,
                                                                 amsRMQFailure);
    _updateSurrogate = updateSurrogate;
  }

  /**
   * @brief Check if a environment variable is set to 1
   * @return True if envVar is set 1, false otherwise
   */
  bool checkEnvVariable(const std::string& envVar)
  {
    if (const char* env_p = std::getenv(envVar.c_str()))
      return strcmp(env_p, "1") == 0 ? true : false;
    return false;
  }

  /**
   * @brief Check if the RabbitMQ connection is connected for the publisher.
   * @return True if connected
   */
  bool isPublisherConnected() const
  {
    return _publishingManager && _publishingManager->isConnected();
  }

  /**
   * @brief Check the RabbitMQ connection is connected.
   * @return True if connected
   */
  bool isConnected() const
  {
    // TODO: Add back the consumer here
    return isPublisherConnected();
  }

  /**
   * @brief Set the internal ID of the interface (usually MPI rank).
   * @param[in] id The ID
   */
  void setId(uint64_t id) { _rId = id; }

  /**
   * @brief Return the latest model and, by default, delete the corresponding message from the Consumer
   * @param[in] domain_name The name of the domain
   * @param[in] num_elements The number of elements for inputs/outputs
   * @param[in] inputs A vector containing arrays of inputs, each array has num_elements elements
   * @param[in] outputs A vector containing arrays of outputs, each array has num_elements elements
   */
  void publish(std::string& domain_name,
               ArrayRef<torch::Tensor> Inputs,
               ArrayRef<torch::Tensor> Outputs)
  {
    CALIPER(CALI_MARK_BEGIN("STORE_RMQ");)
    AMS_DBG(RMQInterface,
            "[tag={}] stores {} elements of input/output "
            "dimensions ({}, {})",
            _msg_tag,
            Inputs.size(),
            Outputs.size())

    AMSMessage msg(_msg_tag, _rId, domain_name, Inputs, Outputs);

    // TODO: we could simplify the logic here
    // AMSMessage could directly produce a shared ptr
    std::shared_ptr<uint8_t> ptr(msg.data(), AMSMessage::getDeleter());
    PublishMessage record(ptr, msg.size(), _msg_tag);
    _publishingManager->publish(record);

    _msg_tag++;
    CALIPER(CALI_MARK_END("STORE_RMQ");)
  }

  /**
   * @brief Flush messages from the RMQ connection
   * @param[in] repeat Number of times to repeat flushing
   * @param[in] ms     Number of ms to wait before each trial
   */
  void flush(int repeat, int ms)
  {
    _publishingManager->flush();
    int iters = 0;
    while (_publishingManager->pendingMessages() > 0 && (iters++ < repeat)) {
      AMS_DBG(RMQInterface,
              "[r{}] Flushing messages {} messages ...",
              _rId,
              _publishingManager->pendingMessages())
      std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
  }

  /**
   * @brief Close the underlying connection
   */
  void close()
  {
    flush(100, 100);
    _publishingManager->stop();
    auto size = MessagesBuffer::getInstance().size();
    if (size != 0) {
      AMS_DBG(RMQInterface, "Rank {} did not ack {} messages", _rId, size)
    }
  }

  ~RMQInterface()
  {
    if (_publishingManager) _publishingManager->stop();
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
  virtual void store(ArrayRef<torch::Tensor> Inputs,
                     ArrayRef<torch::Tensor> Outputs)
  {
    interface.publish(appDomain, Inputs, Outputs);
  }


  /**
   * @brief Return the type of this broker
   * @return The type of the broker
   */
  std::string type() override { return "rabbitmq"; }

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
    AMS_FATAL(RMQInterface, "RMQ Disabled yet we are requesting to connect")
    return false;
  }

  bool isConnected() const { return false; }

  void close() {}
};

#endif  // __AMS_ENABLE_RMQ__

class FilesystemInterface
{
  std::string dbPath;
  bool connected;

public:
  FilesystemInterface() : connected(false) {}

  bool connect(std::string& path)
  {
    connected = true;
    std::experimental::filesystem::path Path(path);
    std::error_code ec;

    if (!std::experimental::filesystem::exists(Path, ec)) {
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

#ifdef __AMS_ENABLE_RMQ__
  friend RabbitMQDB;
#endif

private:
  std::unordered_map<std::string, std::shared_ptr<BaseDB>> db_instances;
  AMSDBType dbType;
  uint64_t rId;
  /** @brief If True, the DB is allowed to update the surrogate model */
  bool updateSurrogate;

  DBManager() : dbType(AMSDBType::AMS_NONE), updateSurrogate(false) {};

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
      AMS_DBG(DBManager,
              "Closing DB {} {} (#client={})",
              e.first,
              static_cast<void*>(e.second.get()),
              e.second.use_count() - 1);
      if (e.second.use_count() > 0) e.second->close();
    }

    if (rmq_interface.isConnected()) {
      AMS_DBG(DBManager, "Closing RMQ Connection");
      rmq_interface.close();
    }
  }

  DBManager(const DBManager&) = delete;
  DBManager(DBManager&&) = delete;
  DBManager& operator=(const DBManager&) = delete;
  DBManager& operator=(DBManager&&) = delete;

  bool isInitialized() const
  {
    return fs_interface.isConnected();  // || rmq_interface.isConnected();
  }

  /**
  * @brief Create an object of the respective database.
  * This should never be used for large scale simulations as txt/csv format will
  * be extremely slow.
  * @param[in] domainName name of the domain model to store data for
  * @param[in] dbType Type of the database to create
  * @param[in] rId a unique Id for each process taking part in a distributed
  * execution (rank-id)
  */
  // Declared here, implemented in basedb.cpp to avoid including jsondb.hpp in header
  std::shared_ptr<BaseDB> createDB(std::string& domainName,
                                   AMSDBType dbType,
                                   uint64_t rId = 0);

  /**
  * @brief get a data base object referred by this string.
  * This should never be used for large scale simulations as txt/csv format will
  * be extremely slow.
  * @param[in] domainName name of the domain model to store data for. 
  * @param[in] rId a unique Id for each process taking part in a distributed
  * execution (rank-id)
  */
  std::shared_ptr<BaseDB> getDB(std::string& domainName, uint64_t rId = 0)
  {
    AMS_DBG(DBManager,
            "Requested DB for domain: '{}' DB Configured to "
            "operate with '{}'",
            domainName,
            getDBTypeAsStr(dbType))

    if (dbType == AMSDBType::AMS_NONE) return nullptr;

    std::string key = domainName;

    auto db_iter = db_instances.find(std::string(key));
    if (db_iter == db_instances.end()) {
      auto db = createDB(domainName, dbType, rId);
      db_instances.insert(std::make_pair(std::string(domainName), db));
      AMS_DBG(DBManager,
              "Creating new Database writting to file: {}",
              domainName);
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
    AMS_DBG(DBManager,
            "Using existing Database writting to file: {}",
            domainName);

    return db;
  }

  void dropDB(std::string& domainName, uint64_t rId = 0)
  {
    AMS_DBG(DBManager,
            "Requested DB for domain: '{}' DB Configured to "
            "operate with '{}'",
            domainName,
            getDBTypeAsStr(dbType))

    if (dbType == AMSDBType::AMS_NONE) return;

    std::string key = domainName;
    auto db_iter = db_instances.find(std::string(key));
    if (db_iter == db_instances.end()) return;

    if (db_iter->second == nullptr) {
      return;
    }

    if (db_iter->second->getId() != rId) {
      THROW(std::runtime_error, "Deleting databases from different ranks");
    }

    if (db_iter->second.use_count() == 2) {
      AMS_DBG(DBManager, "Removing element");
      db_instances.erase(db_iter);
    }
  }

  void instantiate_fs_db(AMSDBType type,
                         std::string db_path,
                         bool is_debug = false)
  {
    AMS_CWARNING(DBManager,
                 isInitialized(),
                 "Data Base is already initialized. Reconfiguring can result "
                 "into "
                 "issues")

    AMS_CWARNING(DBManager,
                 dbType != AMSDBType::AMS_NONE,
                 "Setting DBManager default DB when already set")
    dbType = type;

    AMS_CWARNING(DBManager,
                 (is_debug && dbType != AMSDBType::AMS_HDF5),
                 "Only HDF5 supports debug")

    if (dbType != AMSDBType::AMS_NONE) fs_interface.connect(db_path);
  }

  void instantiate_rmq_db(int port,
                          std::string& host,
                          std::string& rmq_pass,
                          std::string& rmq_user,
                          std::string& rmq_vhost,
                          std::string& rmq_cert,
                          std::string& outbound_queue,
                          std::string& exchange,
                          std::string& routing_key,
                          bool update_surrogate)
  {
    std::experimental::filesystem::path Path(rmq_cert);
    std::error_code ec;
    AMS_CWARNING(AMS,
                 !std::experimental::filesystem::exists(Path, ec),
                 "Certificate file '{}' for RMQ server does not exist. AMS "
                 "will "
                 "try to connect without it.",
                 rmq_cert);
    dbType = AMSDBType::AMS_RMQ;
    updateSurrogate = update_surrogate;
#ifdef __AMS_ENABLE_RMQ__
    rmq_interface.connect(rmq_user,
                          rmq_pass,
                          rmq_vhost,
                          host,
                          port,
                          rmq_cert,
                          outbound_queue,
                          exchange,
                          routing_key,
                          update_surrogate);
#else
    AMS_FATAL(DBManager,
              "Requsted RMQ database but AMS is not built with such support "
              "enabled")
#endif
  }

  size_t getNumInstances() const { return db_instances.size(); }
  void clean() { db_instances.clear(); }
};

}  // namespace db
}  // namespace ams

#endif  // __AMS_BASE_DB__
