/**
 * @brief Class that represents the state of RMQPublisher, how many messages
 * has been sent, acknowledged, lost etc.
 * This class helps us keep a precise view of what has been done and can be
 * used to restart the publishing process from a coherent starting point in
 * case of failures.
 */
class RMQPublisherState {
private:
  std::mutex _mutex;
  /** @brief Total number of messages sent */
  int _nb_msg;
  /** @brief Number of messages successfully acknowledged */
  int _nb_msg_ack;
  /** @brief Messages that have not yet been ack'ed by RabbitMQ (by default all messages until they are ack'ed) */
  std::vector<AMSMessage> _buffer;

  void _update(const AMSMessage& msg, std::vector<AMSMessage>& buf) {
    const std::lock_guard<std::mutex> lock(_mutex);
    buf.push_back(msg);
  }

  /**
   *  @brief  Free the data pointed pointer in a vector and update vector.
   *  @param[in]  addr            Address of memory to free.
   *  @param[in]  buffer          The vector containing memory buffers
   */
  void _free_ams_message(int msg_id, std::vector<AMSMessage>& buf)
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    auto it =
        std::find_if(buf.begin(), buf.end(), [&msg_id](const AMSMessage& obj) {
          return obj.id() == msg_id;
        });
    if (it == buf.end()) {
      WARNING(RMQPublisherState,
              "Failed to deallocate msg #%d: not found",
              msg_id)
      return;
    }
    auto& msg = *it;
    auto& rm = ams::ResourceManager::getInstance();
    try {
      rm.deallocate(msg.data(), AMSResourceType::HOST);
    } catch (const umpire::util::Exception& e) {
      WARNING(RMQPublisherState,
              "Failed to deallocate #%d (%p)",
              msg.id(),
              msg.data());
    }
    DBG(RMQPublisherState, "Deallocated msg #%d (%p)", msg.id(), msg.data())
    it = std::remove_if(buf.begin(),
                        buf.end(),
                        [&msg_id](const AMSMessage& obj) {
                          return obj.id() == msg_id;
                        });
    CWARNING(RMQPublisherState,
             it == buf.end(),
             "Failed to erase %p from buffer",
             msg.data());
    buf.erase(it, buf.end());
  }

  /**
   *  @brief  Free the data pointed by each pointer in a vector.
   *  @param[in]  buffer            The vector containing memory buffers
   */
  void _free_all_messages(std::vector<AMSMessage>& buffer)
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    auto& rm = ams::ResourceManager::getInstance();
    for (auto& dp : buffer) {
      DBG(RMQPublisherState, "deallocate msg #%d (%p)", dp.id(), dp.data())
      try {
        rm.deallocate(dp.data(), AMSResourceType::HOST);
      } catch (const umpire::util::Exception& e) {
        WARNING(RMQPublisherState,
                "Failed to deallocate msg #%d (%p)",
                dp.id(),
                dp.data());
      }
    }
    buffer.erase(buffer.begin(), buffer.end());
  }

public:
  RMQPublisherState() : _nb_msg(0), _nb_msg_ack(0) {}
  ~RMQPublisherState() {
    DBG(RMQPublisherState, "in ~RMQPublisherState");
    // cleanup();
  };

  RMQPublisherState(const RMQPublisherState&) = delete;
  RMQPublisherState& operator=(const RMQPublisherState&) = delete;

  RMQPublisherState(RMQPublisherState&& other) noexcept { *this = std::move(other); }

  RMQPublisherState& operator=(RMQPublisherState&& other) noexcept
  {
    DBG(RMQPublisherState, "Move RMQPublisherState : %d / %d", other._nb_msg, other._nb_msg_ack)
    if (this != &other) {
      _nb_msg = other._nb_msg;
      _nb_msg_ack = other._nb_msg_ack;
      _buffer = std::move(other._buffer);
    }
    return *this;
  }

  int msg_sent() const { return _nb_msg; }
  int msg_acknowledged() const { return _nb_msg_ack; }
  size_t size() const { return _buffer.size(); }

  void update(const AMSMessage& msg) {
    _update(msg, _buffer);
    _nb_msg++;
  }

  AMSMessage&& pop_back() {
    AMSMessage&& msg = std::move(*_buffer.back());
    _buffer.pop_back();
    return msg;
  }

  void acknowledge(AMSMessage) {
    DBG(RMQPublisherState, "acknowledged messages : %d", msg.id())
    _free_ams_message(msg.id(), _buffer)
    _nb_msg_ack++;
  }

  int min_msg_id() {
    AMSMessage& msg_min =
        *(std::min_element(messages.begin(),
                           messages.end(),
                           [](const AMSMessage& a, const AMSMessage& b) {
                             return a.id() < b.id();
                           }));
    return msg_min.id();
  }

  void cleanup() { _free_all_messages(_buffer); }
}; // class RMQPublisherState