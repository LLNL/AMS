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

AMSMsgHeader::AMSMsgHeader(size_t mpi_rank,
                           size_t domain_size,
                           size_t num_elem,
                           size_t in_dim,
                           size_t out_dim,
                           size_t type_size)
    : hsize(static_cast<uint8_t>(AMSMsgHeader::size())),
      dtype(static_cast<uint8_t>(type_size)),
      mpi_rank(static_cast<uint16_t>(mpi_rank)),
      domain_size(static_cast<uint16_t>(domain_size)),
      num_elem(static_cast<uint32_t>(num_elem)),
      in_dim(static_cast<uint16_t>(in_dim)),
      out_dim(static_cast<uint16_t>(out_dim))
{
}

AMSMsgHeader::AMSMsgHeader(uint16_t mpi_rank,
                           uint16_t domain_size,
                           uint32_t num_elem,
                           uint16_t in_dim,
                           uint16_t out_dim,
                           uint8_t type_size)
    : hsize(static_cast<uint8_t>(AMSMsgHeader::size())),
      dtype(type_size),
      mpi_rank(mpi_rank),
      domain_size(domain_size),
      num_elem(num_elem),
      in_dim(in_dim),
      out_dim(out_dim)
{
}

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

AMSMessage::AMSMessage(int id, uint64_t rId, uint8_t* data)
    : _id(id),
      _num_elements(0),
      _input_dim(0),
      _output_dim(0),
      _data(data),
      _total_size(0)
{
  auto header = AMSMsgHeader::decode(data);

  int current_rank = rId;
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
 * AMSMessageInbound
 */

AMSMessageInbound::AMSMessageInbound(uint64_t id,
                                     uint64_t rId,
                                     std::string body,
                                     std::string exchange,
                                     std::string routing_key,
                                     bool redelivered)
    : id(id),
      rId(rId),
      body(std::move(body)),
      exchange(std::move(exchange)),
      routing_key(std::move(routing_key)),
      redelivered(redelivered){};


bool AMSMessageInbound::empty() { return body.empty() || routing_key.empty(); }

bool AMSMessageInbound::isTraining()
{
  auto split = splitString(body, ":");
  return split[0] == "UPDATE";
}

std::string AMSMessageInbound::getModelPath()
{
  auto split = splitString(body, ":");
  if (split[0] == "UPDATE") {
    return split[1];
  }
  return {};
}

std::vector<std::string> AMSMessageInbound::splitString(std::string str,
                                                        std::string delimiter)
{
  size_t pos = 0;
  std::string token;
  std::vector<std::string> res;
  while ((pos = str.find(delimiter)) != std::string::npos) {
    token = str.substr(0, pos);
    res.push_back(token);
    str.erase(0, pos + delimiter.length());
  }
  res.push_back(str);
  return res;
}

/**
 * RMQHandler
 */

RMQHandler::RMQHandler(uint64_t rId,
                       std::shared_ptr<struct event_base> loop,
                       std::string cacert)
    : AMQP::LibEventHandler(loop.get()),
      _rId(rId),
      _loop(loop),
      _cacert(std::move(cacert))
{
  established = establish_connection.get_future();
  closed = close_connection.get_future();
  ftr_error = error_connection.get_future();
}

bool RMQHandler::waitToEstablish(unsigned ms, int repeat)
{
  if (waitFuture(established, ms, repeat)) {
    auto status = established.get();
    DBG(RMQHandler, "Connection Status: %d", status);
    return status == CONNECTED;
  }
  return false;
}

bool RMQHandler::waitToClose(unsigned ms, int repeat)
{
  if (waitFuture(closed, ms, repeat)) {
    return closed.get() == CLOSED;
  }
  return false;
}

bool RMQHandler::connectionValid()
{
  std::chrono::milliseconds span(1);
  return ftr_error.wait_for(span) != std::future_status::ready;
}

bool RMQHandler::onSecuring(AMQP::TcpConnection* connection, SSL* ssl)
{
  // No TLS certificate provided
  if (_cacert.empty()) {
    CFATAL(RMQHandler, false, "No TLS certificate. Bypassing.")
    return true;
  }

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
    DBG(RMQHandler, "Success logged with ca-chain %s", _cacert.c_str())
    return true;
  }
}

bool RMQHandler::onSecured(AMQP::TcpConnection* connection, const SSL* ssl)
{
  DBG(RMQHandler, "[r%d] Secured TLS connection has been established.", _rId)
  return true;
}

void RMQHandler::onClosed(AMQP::TcpConnection* connection)
{
  DBG(RMQHandler, "[r%d] Connection is closed.", _rId)
}

void RMQHandler::onError(AMQP::TcpConnection* connection, const char* message)
{
  CFATAL(RMQHandler, false, "In onError %s", message)
  WARNING(RMQHandler, "[r%d] fatal error on TCP connection: %s", _rId, message)
  try {
    error_connection.set_value(ERROR);
  } catch (const std::future_error& e) {
    DBG(RMQHandler, "[r%d] future already set.", _rId)
  }
}

void RMQHandler::onDetached(AMQP::TcpConnection* connection)
{
  CFATAL(RMQHandler, false, "[r%d] Connection is detached.", _rId)
  close_connection.set_value(CLOSED);
}

bool RMQHandler::waitFuture(std::future<RMQConnectionStatus>& future,
                            unsigned ms,
                            int repeat)
{
  std::chrono::milliseconds span(ms);
  int iters = 0;
  std::future_status status;
  CFATAL(RMQHandler, false, "in waitFuture")

  while ((status = future.wait_for(span)) == std::future_status::timeout &&
         (iters++ < repeat))
    std::future<RMQConnectionStatus> established;
  return status == std::future_status::ready;
}

/**
 * RMQConsumerHandler
 */

RMQConsumerHandler::RMQConsumerHandler(uint64_t rId,
                                       std::shared_ptr<struct event_base> loop,
                                       std::string cacert,
                                       std::string exchange,
                                       std::string routing_key,
                                       AMQP::ExchangeType extype)
    : RMQHandler(rId, loop, cacert),
      _exchange(exchange),
      _extype(extype),
      _routing_key(routing_key),
      _messages(std::make_shared<std::vector<AMSMessageInbound>>()),
      _channel(nullptr)
{
}

std::tuple<uint64_t, std::string> RMQConsumerHandler::getLatestModel()
{
  std::string model = "";
  uint64_t latest_tag = 0;
  for (AMSMessageInbound& e : *_messages) {
    if (latest_tag < e.id) {
      model = e.getModelPath();
      latest_tag = e.id;
    }
  }
  return std::make_tuple(latest_tag, model);
}

AMSMessageInbound RMQConsumerHandler::popMessages()
{
  if (!_messages->empty()) {
    AMSMessageInbound msg = _messages->back();
    _messages->pop_back();
    return msg;
  }
  return AMSMessageInbound();
}

AMSMessageInbound RMQConsumerHandler::getMessages(uint64_t delivery_tag,
                                                  bool erase)
{
  if (!_messages->empty()) {
    auto it = std::find_if(_messages->begin(),
                           _messages->end(),
                           [&delivery_tag](const AMSMessageInbound& e) {
                             return e.id == delivery_tag;
                           });
    if (it != _messages->end()) {
      AMSMessageInbound msg(std::move(*it));
      if (erase) _messages->erase(it);
      return msg;
    }
  }
  return AMSMessageInbound();
}

void RMQConsumerHandler::onReady(AMQP::TcpConnection* connection)
{
  DBG(RMQConsumerHandler,
      "[r%d] Sucessfuly logged in. Connection ready to use.",
      _rId)

  _channel = std::make_shared<AMQP::TcpChannel>(connection);
  _channel->onError([&](const char* message) {
    WARNING(RMQConsumerHandler, "[r%d] Error on channel: %s", _rId, message)
    establish_connection.set_value(FAILED);
  });

  // The exchange will be deleted once all bound queues are removed
  _channel->declareExchange(_exchange, _extype, AMQP::autodelete)
      .onSuccess([&, this]() {
        DBG(RMQConsumerHandler,
            "[r%d] declared exchange %s (type: %d)",
            _rId,
            _exchange.c_str(),
            _extype)
        establish_connection.set_value(CONNECTED);
        _channel->declareQueue(AMQP::exclusive)
            .onSuccess([&, this](const std::string& name,
                                 uint32_t messagecount,
                                 uint32_t consumercount) {
              DBG(RMQConsumerHandler,
                  "[r%d] declared queue: %s (messagecount=%d, "
                  "consumercount=%d)",
                  _rId,
                  name.c_str(),
                  messagecount,
                  consumercount)
              _channel->bindQueue(_exchange, name, _routing_key)
                  .onSuccess([&, name, this]() {
                    DBG(RMQConsumerHandler,
                        "[r%d] Bounded queue %s to exchange %s with "
                        "routing key = %s",
                        _rId,
                        name.c_str(),
                        _exchange.c_str(),
                        _routing_key.c_str())

                    // We can now install callback functions for when we will consumme messages
                    // callback function that is called when the consume operation starts
                    auto startCb = [&](const std::string& consumertag) {
                      DBG(RMQConsumerHandler,
                          "[r%d] consume operation started with tag: %s",
                          _rId,
                          consumertag.c_str())
                    };

                    // callback function that is called when the consume operation failed
                    auto errorCb = [&](const char* message) {
                      WARNING(RMQConsumerHandler,
                              "[r%d] consume operation failed: %s",
                              _rId,
                              message);
                    };
                    // callback operation when a message was received
                    auto messageCb = [&](const AMQP::Message& message,
                                         uint64_t deliveryTag,
                                         bool redelivered) {
                      // acknowledge the message
                      _channel->ack(deliveryTag);
                      // _on_message_received(message, deliveryTag, redelivered);
                      std::string msg(message.body(), message.bodySize());
                      DBG(RMQConsumerHandler,
                          "[r%d] message received [tag=%d] : '%s' of size "
                          "%d B from "
                          "'%s'/'%s'",
                          _rId,
                          deliveryTag,
                          msg.c_str(),
                          message.bodySize(),
                          message.exchange().c_str(),
                          message.routingkey().c_str())
                      _messages->push_back(
                          AMSMessageInbound(deliveryTag,
                                            _rId,
                                            msg,
                                            message.exchange(),
                                            message.routingkey(),
                                            redelivered));
                    };

                    /* callback that is called when the consumer is cancelled by RabbitMQ (this
                    * only happens in rare situations, for example when someone removes the queue
                    * that you are consuming from)
                    */
                    auto cancelledCb = [&](const std::string& consumertag) {
                      WARNING(RMQConsumerHandler,
                              "[r%d] consume operation cancelled by the "
                              "RabbitMQ server: %s",
                              _rId,
                              consumertag.c_str())
                    };

                    DBG(RMQConsumerHandler,
                        "[r%d] starting consume operation",
                        _rId)

                    // start consuming from the queue, and install the callbacks
                    _channel->consume(name)
                        .onReceived(std::move(messageCb))
                        .onSuccess(std::move(startCb))
                        .onCancelled(std::move(cancelledCb))
                        .onError(std::move(errorCb));
                  })  //consume
                  .onError([&](const char* message) {
                    WARNING(RMQConsumerHandler,
                            "[r%d] creating queue: %s",
                            _rId,
                            message)
                    establish_connection.set_value(FAILED);
                  });  //consume
            })         //bindQueue
            .onError([&](const char* message) {
              WARNING(RMQConsumerHandler,
                      "[r%d] failed to bind queue to exchange: %s",
                      _rId,
                      message)
            });  //bindQueue
      })         //declareExchange
      .onError([&](const char* message) {
        WARNING(RMQConsumerHandler,
                "[r%d] failed to create exchange: %s",
                _rId,
                message)
        establish_connection.set_value(FAILED);
      });  //declareExchange
}

/**
 * RMQConsumer
 */

RMQConsumer::RMQConsumer(uint64_t rId,
                         const AMQP::Address& address,
                         std::string cacert,
                         std::string exchange,
                         std::string routing_key)
    : _rId(rId),
      _cacert(cacert),
      _routing_key(routing_key),
      _exchange(exchange),
      _handler(nullptr)
{
#ifdef EVTHREAD_USE_PTHREADS_IMPLEMENTED
  evthread_use_pthreads();
#endif
  DBG(RMQConsumer,
      "Libevent %s (LIBEVENT_VERSION_NUMBER = %#010x)",
      event_get_version(),
      event_get_version_number());
  DBG(RMQConsumer,
      "%s (OPENSSL_VERSION_NUMBER = %#010x)",
      OPENSSL_VERSION_TEXT,
      OPENSSL_VERSION_NUMBER);
#if OPENSSL_VERSION_NUMBER < 0x10100000L
  SSL_library_init();
#else
  OPENSSL_init_ssl(0, NULL);
#endif
  DBG(RMQConsumer,
      "RabbitMQ address: %s:%d/%s (exchange = %s / routing key = %s)",
      address.hostname().c_str(),
      address.port(),
      address.vhost().c_str(),
      _exchange.c_str(),
      _routing_key.c_str())

  _loop = std::shared_ptr<struct event_base>(event_base_new(),
                                             [](struct event_base* event) {
                                               event_base_free(event);
                                             });
  _handler = std::make_shared<RMQConsumerHandler>(
      rId, _loop, _cacert, _exchange, _routing_key, AMQP::fanout);
  _connection = new AMQP::TcpConnection(_handler.get(), address);
}

void RMQConsumer::start() { event_base_dispatch(_loop.get()); }

void RMQConsumer::stop() { event_base_loopexit(_loop.get(), NULL); }

bool RMQConsumer::ready()
{
  return _connection->ready() && _connection->usable();
}

bool RMQConsumer::waitToEstablish(unsigned ms, int repeat)
{
  return _handler->waitToEstablish(ms, repeat);
}

AMSMessageInbound RMQConsumer::popMessages()
{
  return _handler->popMessages();
};

void RMQConsumer::delMessage(uint64_t delivery_tag)
{
  _handler->delMessage(delivery_tag);
}

AMSMessageInbound RMQConsumer::getMessages(uint64_t delivery_tag, bool erase)
{
  return _handler->getMessages(delivery_tag, erase);
}

std::tuple<uint64_t, std::string> RMQConsumer::getLatestModel()
{
  return _handler->getLatestModel();
}

bool RMQConsumer::close(unsigned ms, int repeat)
{
  _connection->close(false);
  return _handler->waitToClose(ms, repeat);
}

RMQConsumer::~RMQConsumer()
{
  _connection->close(false);
  delete _connection;
}

/**
 * RMQPublisherHandler
 */

RMQPublisherHandler::RMQPublisherHandler(
    uint64_t rId,
    std::shared_ptr<struct event_base> loop,
    std::string cacert,
    std::string queue)
    : RMQHandler(rId, loop, cacert),
      _queue(queue),
      _nb_msg_ack(0),
      _nb_msg(0),
      _channel(nullptr),
      _rchannel(nullptr)
{
}

/**
 *  @brief  Return the messages that have NOT been acknowledged by the RabbitMQ server. 
 *  @return     A vector of AMSMessage
 */
std::vector<AMSMessage>& RMQPublisherHandler::msgBuffer() { return _messages; }

/**
 *  @brief    Free AMSMessages held by the handler
 */
void RMQPublisherHandler::cleanup() { freeAllMessages(_messages); }

/**
 *  @brief    Total number of messages sent
 *  @return   Number of messages
 */
int RMQPublisherHandler::msgSent() const { return _nb_msg; }

/**
 *  @brief    Total number of messages successfully acknowledged
 *  @return   Number of messages
 */
int RMQPublisherHandler::msgAcknowledged() const { return _nb_msg_ack; }

/**
 *  @brief    Total number of messages unacknowledged
 *  @return   Number of messages unacknowledged
 */
unsigned RMQPublisherHandler::unacknowledged() const
{
  return _rchannel->unacknowledged();
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
              "[r%d] message #%d (Addr:%p) got acknowledged "
              "successfully "
              "by "
              "RMQ "
              "server",
              _rId,
              id,
              data)
          this->freeMessage(id, _messages);
          _nb_msg_ack++;
        })
        .onNack([this, id = msg.id(), data = msg.data()]() mutable {
          WARNING(RMQPublisherHandler,
                  "[r%d] message #%d (%p) received negative "
                  "acknowledged "
                  "by "
                  "RMQ "
                  "server",
                  _rId,
                  id,
                  data)
        })
        .onError([this, id = msg.id(), data = msg.data()](
                     const char* err_message) mutable {
          WARNING(RMQPublisherHandler,
                  "[r%d] message #%d (%p) did not get send: %s",
                  _rId,
                  id,
                  data,
                  err_message)
        });
  } else {
    WARNING(RMQPublisherHandler,
            "[r%d] The reliable channel was not ready for message #%d.",
            _rId,
            msg.id())
  }
  _nb_msg++;
}

void RMQPublisherHandler::onReady(AMQP::TcpConnection* connection)
{
  DBG(RMQPublisherHandler,
      "[r%d] Sucessfuly logged in (connection %p). Connection ready to "
      "use.",
      _rId,
      connection)

  _channel = std::make_shared<AMQP::TcpChannel>(connection);
  _channel->onError([&](const char* message) {
    CFATAL(
        RMQPublisherHandler, false, "[r%d] Error on channel: %s", _rId, message)
  });

  _channel->declareQueue(_queue)
      .onSuccess([&](const std::string& name,
                     uint32_t messagecount,
                     uint32_t consumercount) {
        DBG(RMQPublisherHandler,
            "[r%d] declared queue: %s (messagecount=%d, "
            "consumercount=%d)",
            _rId,
            _queue.c_str(),
            messagecount,
            consumercount)
        // We can now instantiate the shared buffer between AMS and RMQ
        _rchannel =
            std::make_shared<AMQP::Reliable<AMQP::Tagger>>(*_channel.get());
        establish_connection.set_value(CONNECTED);
      })
      .onError([&](const char* message) {
        CFATAL(RMQPublisherHandler,
               false,
               "[r%d] Error while creating broker queue (%s): "
               "%s",
               _rId,
               _queue.c_str(),
               message)
        establish_connection.set_value(FAILED);
      });
}

void RMQPublisherHandler::freeMessage(int msg_id, std::vector<AMSMessage>& buf)
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

void RMQPublisherHandler::freeAllMessages(std::vector<AMSMessage>& buffer)
{
  const std::lock_guard<std::mutex> lock(_mutex);
  auto& rm = ams::ResourceManager::getInstance();
  for (auto& dp : buffer) {
    DBG(RMQPublisherHandler, "deallocate msg #%d (%p)", dp.id(), dp.data())
    rm.deallocate(dp.data(), AMSResourceType::AMS_HOST);
  }
  buffer.clear();
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
  freeAllMessages(_messages);
}

/**
 * RMQPublisher
 */

RMQPublisher::RMQPublisher(uint64_t rId,
                           const AMQP::Address& address,
                           std::string cacert,
                           std::string queue,
                           std::vector<AMSMessage>&& msgs_to_send)
    : _rId(rId),
      _queue(queue),
      _cacert(cacert),
      _handler(nullptr),
      _buffer_msg(std::move(msgs_to_send))
{
#ifdef EVTHREAD_USE_PTHREADS_IMPLEMENTED
  evthread_use_pthreads();
#endif
  DBG(RMQPublisher,
      "Libevent %s (LIBEVENT_VERSION_NUMBER = %#010x)",
      event_get_version(),
      event_get_version_number());
  DBG(RMQPublisher,
      "%s (OPENSSL_VERSION_NUMBER = %#010x)",
      OPENSSL_VERSION_TEXT,
      OPENSSL_VERSION_NUMBER);
#if OPENSSL_VERSION_NUMBER < 0x10100000L
  SSL_library_init();
#else
  OPENSSL_init_ssl(0, NULL);
#endif
  DBG(RMQPublisher,
      "RabbitMQ address: %s:%d/%s (queue = %s)",
      address.hostname().c_str(),
      address.port(),
      address.vhost().c_str(),
      _queue.c_str())

  _loop = std::shared_ptr<struct event_base>(event_base_new(),
                                             [](struct event_base* event) {
                                               event_base_free(event);
                                             });

  _handler =
      std::make_shared<RMQPublisherHandler>(_rId, _loop, _cacert, _queue);
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

bool RMQPublisher::ready_publish()
{
  return _connection->ready() && _connection->usable();
}

bool RMQPublisher::waitToEstablish(unsigned ms, int repeat)
{
  return _handler->waitToEstablish(ms, repeat);
}

unsigned RMQPublisher::unacknowledged() const
{
  return _handler->unacknowledged();
}

void RMQPublisher::start() { event_base_dispatch(_loop.get()); }

void RMQPublisher::stop() { event_base_loopexit(_loop.get(), NULL); }

bool RMQPublisher::connectionValid() { return _handler->connectionValid(); }

std::vector<AMSMessage>& RMQPublisher::getMsgBuffer()
{
  return _handler->msgBuffer();
}

void RMQPublisher::cleanup() { _handler->cleanup(); }

int RMQPublisher::msgSent() const { return _handler->msgSent(); }

int RMQPublisher::msgAcknowledged() const
{
  return _handler->msgAcknowledged();
}

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
                           std::string outbound_queue,
                           std::string exchange,
                           std::string routing_key)
{
  _queue_sender = outbound_queue;
  _exchange = exchange;
  _routing_key = routing_key;
  _cacert = rmq_cert;

  // Here we generate 64-bits wide random numbers to have a unique distributed ID
  // WARNING: there is no guarantee of uniqueness here as each MPI rank will have its own generator
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distrib(0,
                                             std::numeric_limits<int>::max());
  _rId = static_cast<uint64_t>(distrib(generator));

  AMQP::Login login(rmq_user, rmq_password);
  _address = std::make_shared<AMQP::Address>(service_host,
                                             service_port,
                                             login,
                                             rmq_vhost,
                                             /*is_secure*/ true);
  _publisher =
      std::make_shared<RMQPublisher>(_rId, *_address, _cacert, _queue_sender);

  _publisher_thread = std::thread([&]() { _publisher->start(); });

  if (!_publisher->waitToEstablish(100, 10)) {
    _publisher->stop();
    _publisher_thread.join();
    FATAL(RabbitMQInterface, "Could not establish connection");
  }

  _consumer = std::make_shared<RMQConsumer>(
      _rId, *_address, _cacert, _exchange, _routing_key);
  _consumer_thread = std::thread([&]() { _consumer->start(); });

  if (!_consumer->waitToEstablish(100, 10)) {
    _consumer->stop();
    _consumer_thread.join();
    FATAL(RabbitMQDB, "Could not establish consumer connection");
  }

  connected = true;
  return connected;
}

void RMQInterface::restartPublisher()
{
  std::vector<AMSMessage> messages = _publisher->getMsgBuffer();

  AMSMessage& msg_min =
      *(std::min_element(messages.begin(),
                         messages.end(),
                         [](const AMSMessage& a, const AMSMessage& b) {
                           return a.id() < b.id();
                         }));

  DBG(RMQPublisher,
      "[r%d] we have %lu buffered messages that will get re-send "
      "(starting from msg #%d).",
      _rId,
      messages.size(),
      msg_min.id())

  // Stop the faulty publisher
  _publisher->stop();
  _publisher_thread.join();
  _publisher.reset();
  connected = false;

  _publisher = std::make_shared<RMQPublisher>(
      _rId, *_address, _cacert, _queue_sender, std::move(messages));
  _publisher_thread = std::thread([&]() { _publisher->start(); });
  connected = true;
}

void RMQInterface::close()
{
  if (!_publisher_thread.joinable() || !_consumer_thread.joinable()) {
    DBG(RMQInterface, "Threads are not joinable")
    return;
  }
  bool status = _publisher->close(100, 10);
  CWARNING(RabbitMQDB,
           !status,
           "Could not gracefully close publisher TCP connection")

  DBG(RabbitMQInterface, "Number of messages sent: %d", _msg_tag)
  DBG(RabbitMQInterface,
      "Number of unacknowledged messages are %d",
      _publisher->unacknowledged())
  _publisher->stop();
  _publisher_thread.join();

  status = _consumer->close(100, 10);
  CWARNING(RabbitMQDB,
           !status,
           "Could not gracefully close consumer TCP connection")
  _consumer->stop();
  _consumer_thread.join();

  connected = false;
}
