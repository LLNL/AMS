/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "wf/basedb.hpp"

using namespace ams::db;

/**
 * AMSMsgHeader
 */

size_t AMSMsgHeader::encode(uint8_t* data_blob)
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
  // Domain Size (should be 2 bytes)
  DBG(AMSMsgHeader,
      "Generating domain name of size %d --- %d offset %d",
      domain_size,
      sizeof(domain_size),
      current_offset);
  std::memcpy(data_blob + current_offset, &(domain_size), sizeof(domain_size));
  current_offset += sizeof(domain_size);
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


AMSMsgHeader AMSMsgHeader::decode(uint8_t* data_blob)
{
  size_t current_offset = 0;
  // Header size (should be 1 bytes)
  uint8_t new_hsize = data_blob[current_offset];
  CWARNING(AMSMsgHeader,
           new_hsize != AMSMsgHeader::size(),
           "buffer is likely not a valid AMSMessage (%d / %ld)",
           new_hsize,
           current_offset)

  current_offset += sizeof(uint8_t);
  // Data type (should be 1 bytes)
  uint8_t new_dtype = data_blob[current_offset];
  current_offset += sizeof(uint8_t);
  // MPI rank (should be 2 bytes)
  uint16_t new_mpirank =
      (reinterpret_cast<uint16_t*>(data_blob + current_offset))[0];
  current_offset += sizeof(uint16_t);

  // Domain Size (should be 2 bytes)
  uint16_t new_domain_size =
      (reinterpret_cast<uint16_t*>(data_blob + current_offset))[0];
  current_offset += sizeof(uint16_t);

  // Num elem (should be 4 bytes)
  uint32_t new_num_elem;
  std::memcpy(&new_num_elem, data_blob + current_offset, sizeof(uint32_t));
  current_offset += sizeof(uint32_t);
  // Input dim (should be 2 bytes)
  uint16_t new_in_dim;
  std::memcpy(&new_in_dim, data_blob + current_offset, sizeof(uint16_t));
  current_offset += sizeof(uint16_t);
  // Output dim (should be 2 bytes)
  uint16_t new_out_dim;
  std::memcpy(&new_out_dim, data_blob + current_offset, sizeof(uint16_t));

  return AMSMsgHeader(new_mpirank,
                      new_domain_size,
                      new_num_elem,
                      new_in_dim,
                      new_out_dim,
                      new_dtype);
}

/**
 * AMSMessage
 */

void AMSMessage::swap(const AMSMessage& other)
{
  _id = other._id;
  _rank = other._rank;
  _num_elements = other._num_elements;
  _input_dim = other._input_dim;
  _output_dim = other._output_dim;
  _total_size = other._total_size;
  _data = other._data;
}

AMSMessage::AMSMessage(int id, uint8_t* data)
    : _id(id),
      _num_elements(0),
      _input_dim(0),
      _output_dim(0),
      _data(data),
      _total_size(0)
{
  auto header = AMSMsgHeader::decode(data);

  int current_rank = 0;
#ifdef __ENABLE_MPI__
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &current_rank));
#endif
  _rank = header.mpi_rank;
  CWARNING(AMSMessage,
           _rank != current_rank,
           "MPI rank are not matching (using %d)",
           _rank)

  _num_elements = header.num_elem;
  _input_dim = header.in_dim;
  _output_dim = header.out_dim;
  _data = data;
  auto type_value = header.dtype;

  _total_size = AMSMsgHeader::size() + getTotalElements() * type_value;

  DBG(AMSMessage, "Allocated message %d: %p", _id, _data);
}

/**
 * RMQConsumerHandler
 */

bool RMQConsumerHandler::onSecuring(AMQP::TcpConnection* connection, SSL* ssl)
{
  ERR_clear_error();
  unsigned long err;
#if OPENSSL_VERSION_NUMBER < 0x10100000L
  int ret = SSL_use_certificate_file(ssl, _cacert.c_str(), SSL_FILETYPE_PEM);
#else
  int ret = SSL_use_certificate_chain_file(ssl, _cacert.c_str());
#endif
  // FIXME: with openssl 3.0
  // Set => SSL_set_options(ssl, SSL_OP_IGNORE_UNEXPECTED_EOF);

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
    DBG(RMQConsumerHandler, "Success logged with ca-chain %s", _cacert.c_str())
    return true;
  }
}

void RMQConsumerHandler::onReady(AMQP::TcpConnection* connection)
{
  DBG(RMQConsumerHandler,
      "[rank=%d] Sucessfuly logged in. Connection ready to use.",
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
              "message received [tag=%lu] : '%s' of size %lu B from "
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
 * RMQConsumer
 */

RMQConsumer::RMQConsumer(const AMQP::Address& address,
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

inbound_msg RMQConsumer::pop_messages()
{
  if (!_messages.empty()) {
    inbound_msg msg = _messages.back();
    _messages.pop_back();
    return msg;
  }
  return std::make_tuple("", "", "", -1, false);
}

inbound_msg RMQConsumer::get_messages(uint64_t delivery_tag)
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

/**
 * RMQPublisherHandler
 */

RMQPublisherHandler::RMQPublisherHandler(
    std::shared_ptr<struct event_base> loop,
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
  _ftr_error = _error_connection.get_future();
}

void RMQPublisherHandler::publish(AMSMessage&& msg)
{
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    _messages.push_back(msg);
  }
  if (_rchannel) {
    // publish a message via the reliable-channel
    //    onAck   : message has been explicitly ack'ed by RabbitMQ
    //    onNack  : message has been explicitly nack'ed by RabbitMQ
    //    onError : error occurred before any ack or nack was received
    //    onLost  : messages that have either been nack'ed, or lost
    _rchannel
        ->publish("", _queue, reinterpret_cast<char*>(msg.data()), msg.size())
        .onAck([this,
                &_nb_msg_ack = _nb_msg_ack,
                id = msg.id(),
                data = msg.data(),
                &_messages = this->_messages]() mutable {
          DBG(RMQPublisherHandler,
              "[rank=%d] message #%d (Addr:%p) got acknowledged "
              "successfully "
              "by "
              "RMQ "
              "server",
              _rank,
              id,
              data)
          this->free_ams_message(id, _messages);
          _nb_msg_ack++;
        })
        .onNack([this, id = msg.id(), data = msg.data()]() mutable {
          WARNING(RMQPublisherHandler,
                  "[rank=%d] message #%d (%p) received negative "
                  "acknowledged "
                  "by "
                  "RMQ "
                  "server",
                  _rank,
                  id,
                  data)
        })
        .onError([this, id = msg.id(), data = msg.data()](
                     const char* err_message) mutable {
          WARNING(RMQPublisherHandler,
                  "[rank=%d] message #%d (%p) did not get send: %s",
                  _rank,
                  id,
                  data,
                  err_message)
        });
  } else {
    WARNING(RMQPublisherHandler,
            "[rank=%d] The reliable channel was not ready for message #%d.",
            _rank,
            msg.id())
  }
  _nb_msg++;
}

/**
   *  @brief  Wait (blocking call) until connection has been established or that ms * repeat is over.
   *  @param[in]  ms            Number of milliseconds the function will wait on the future
   *  @param[in]  repeat        Number of times the function will wait
   *  @return     True if connection has been established
   */
bool RMQPublisherHandler::waitToEstablish(unsigned ms, int repeat)
{
  if (waitFuture(established, ms, repeat)) {
    auto status = established.get();
    DBG(RMQPublisherHandler, "Connection Status: %d", status);
    return status == CONNECTED;
  }
  return false;
}

/**
   *  @brief  Wait (blocking call) until connection has been closed or that ms * repeat is over.
   *  @param[in]  ms            Number of milliseconds the function will wait on the future
   *  @param[in]  repeat        Number of times the function will wait
   *  @return     True if connection has been closed
   */
bool RMQPublisherHandler::waitToClose(unsigned ms, int repeat)
{
  if (waitFuture(closed, ms, repeat)) {
    return closed.get() == CLOSED;
  }
  return false;
}

/**
   *  @brief  Check if the connection can be used to send messages.
   *  @return     True if connection is valid (i.e., can send messages)
   */
bool RMQPublisherHandler::connection_valid()
{
  std::chrono::milliseconds span(1);
  return _ftr_error.wait_for(span) != std::future_status::ready;
}

void RMQPublisherHandler::flush()
{
  uint32_t tries = 0;
  while (auto unAck = unacknowledged()) {
    DBG(RMQPublisherHandler,
        "Waiting for %lu messages to be acknowledged",
        unAck);

    if (++tries > 10) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50 * tries));
  }
  free_all_messages(_messages);
}

// TODO: Code duplication between consummer and producer, create common base class
bool RMQPublisherHandler::onSecuring(AMQP::TcpConnection* connection, SSL* ssl)
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
    DBG(RMQPublisherHandler, "Success logged with ca-chain %s", _cacert.c_str())
    return true;
  }
}

/**
   *  @brief Method that is called by the AMQP library when the login attempt
   *  succeeded. After this the connection is ready to use.
   *  @param[in]  connection      The connection that can now be used
   */
void RMQPublisherHandler::onReady(AMQP::TcpConnection* connection)
{
  DBG(RMQPublisherHandler,
      "[rank=%d] Sucessfuly logged in (connection %p). Connection ready to "
      "use.",
      _rank,
      connection)

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

void RMQPublisherHandler::onError(AMQP::TcpConnection* connection,
                                  const char* message)
{
  WARNING(RMQPublisherHandler,
          "[rank=%d] fatal error on TCP connection: %s",
          _rank,
          message)
  try {
    _error_connection.set_value(ERROR);
  } catch (const std::future_error& e) {
    DBG(RMQPublisherHandler, "[rank=%d] future already set.", _rank)
  }
}

/**
    *  @brief Final method that is called. This signals that no further calls to your
    *  handler will be made about the connection.
    *  @param  connection      The connection that can be destructed
    */
void RMQPublisherHandler::onDetached(AMQP::TcpConnection* connection)
{
  //  add your own implementation, like cleanup resources or exit the application
  DBG(RMQPublisherHandler, "[rank=%d] Connection is detached.", _rank)
  close_connection.set_value(CLOSED);
}

bool RMQPublisherHandler::waitFuture(std::future<RMQConnectionStatus>& future,
                                     unsigned ms,
                                     int repeat)
{
  std::chrono::milliseconds span(ms);
  int iters = 0;
  std::future_status status;
  while ((status = future.wait_for(span)) == std::future_status::timeout &&
         (iters++ < repeat))
    std::future<RMQConnectionStatus> established;
  return status == std::future_status::ready;
}

/**
   *  @brief  Free the data pointed pointer in a vector and update vector.
   *  @param[in]  addr            Address of memory to free.
   *  @param[in]  buffer          The vector containing memory buffers
   */
void RMQPublisherHandler::free_ams_message(int msg_id,
                                           std::vector<AMSMessage>& buf)
{
  const std::lock_guard<std::mutex> lock(_mutex);
  auto it =
      std::find_if(buf.begin(), buf.end(), [&msg_id](const AMSMessage& obj) {
        return obj.id() == msg_id;
      });
  CFATAL(RMQPublisherHandler,
         it == buf.end(),
         "Failed to deallocate msg #%d: not found",
         msg_id)
  auto& msg = *it;
  auto& rm = ams::ResourceManager::getInstance();
  rm.deallocate(msg.data(), AMSResourceType::AMS_HOST);

  DBG(RMQPublisherHandler, "Deallocated msg #%d (%p)", msg.id(), msg.data())
  buf.erase(it);
}

/**
   *  @brief  Free the data pointed by each pointer in a vector.
   *  @param[in]  buffer            The vector containing memory buffers
   */
void RMQPublisherHandler::free_all_messages(std::vector<AMSMessage>& buffer)
{
  const std::lock_guard<std::mutex> lock(_mutex);
  auto& rm = ams::ResourceManager::getInstance();
  for (auto& dp : buffer) {
    DBG(RMQPublisherHandler, "deallocate msg #%d (%p)", dp.id(), dp.data())
    rm.deallocate(dp.data(), AMSResourceType::AMS_HOST);
  }
  buffer.clear();
}

/**
 * RMQPublisher
 */

RMQPublisher::RMQPublisher(const AMQP::Address& address,
                           std::string cacert,
                           std::string queue,
                           std::vector<AMSMessage>&& msgs_to_send)
    : _rank(0),
      _queue(queue),
      _cacert(cacert),
      _handler(nullptr),
      _buffer_msg(std::move(msgs_to_send))
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

void RMQPublisher::publish(AMSMessage&& message)
{
  // We have some messages to send first (from a potential restart)
  if (_buffer_msg.size() > 0) {
    for (auto& msg : _buffer_msg) {
      DBG(RMQPublisher,
          "Publishing backed up message %d: %p",
          msg.id(),
          msg.data())
      _handler->publish(std::move(msg));
    }
    _buffer_msg.clear();
  }

  DBG(RMQPublisher, "Publishing message %d: %p", message.id(), message.data())
  _handler->publish(std::move(message));
}

/**
   *  @brief    Total number of messages successfully acknowledged
   *  @return   Number of messages
   */
bool RMQPublisher::close(unsigned ms, int repeat)
{
  _handler->flush();
  _connection->close(false);
  return _handler->waitToClose(ms, repeat);
}

/**
 * RMQInterface
 */

bool RMQInterface::connect(std::string rmq_name,
                           std::string rmq_password,
                           std::string rmq_user,
                           std::string rmq_vhost,
                           int service_port,
                           std::string service_host,
                           std::string rmq_cert,
                           std::string inbouund_queue,
                           std::string outbound_queue)
{
  _queue_sender = outbound_queue;
  _queue_receiver = inbouund_queue;
  _cacert = rmq_cert;

  AMQP::Login login(rmq_user, rmq_password);
  _address = std::make_shared<AMQP::Address>(service_host,
                                             service_port,
                                             login,
                                             rmq_vhost,
                                             /*is_secure*/ true);
  _publisher =
      std::make_shared<RMQPublisher>(*_address, _cacert, _queue_sender);

  _publisher_thread = std::thread([&]() { _publisher->start(); });

  if (!_publisher->waitToEstablish(100, 10)) {
    _publisher->stop();
    _publisher_thread.join();
    FATAL(RabbitMQInterface, "Could not establish connection");
  }

  connected = true;
  return connected;
}

void RMQInterface::restart(int rank)
{
  std::vector<AMSMessage> messages = _publisher->get_buffer_msgs();

  AMSMessage& msg_min =
      *(std::min_element(messages.begin(),
                         messages.end(),
                         [](const AMSMessage& a, const AMSMessage& b) {
                           return a.id() < b.id();
                         }));

  DBG(RMQPublisher,
      "[rank=%d] we have %lu buffered messages that will get re-send "
      "(starting from msg #%d).",
      rank,
      messages.size(),
      msg_min.id())

  // Stop the faulty publisher
  _publisher->stop();
  _publisher_thread.join();
  _publisher.reset();
  connected = false;

  _publisher = std::make_shared<RMQPublisher>(*_address,
                                              _cacert,
                                              _queue_sender,
                                              std::move(messages));
  _publisher_thread = std::thread([&]() { _publisher->start(); });
  connected = true;
}

void RMQInterface::close()
{
  if (!_publisher_thread.joinable()) {
    return;
  }
  bool status = _publisher->close(100, 10);
  CWARNING(RabbitMQDB, !status, "Could not gracefully close TCP connection")
  DBG(RabbitMQInterface, "Number of messages sent: %d", _msg_tag)
  DBG(RabbitMQInterface,
      "Number of unacknowledged messages are %d",
      _publisher->unacknowledged())
  _publisher->stop();
  //_consumer->stop();
  _publisher_thread.join();
  //_consumer_thread.join();
  connected = false;
}