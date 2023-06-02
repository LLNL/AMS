#ifndef __AMS_DATA_BROKER_HPP__
#define __AMS_DATA_BROKER_HPP__

#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <algorithm> // for std::remove

#include "AMS.h"
#include "wf/debug.h"

namespace ams {
/**
* \brief Abstract class that defines what AMS expect from a message broker.
*/
template <typename TypeValue>
class DataBroker {
protected:
    /** \brief unique id of the process running this simulation */
    uint64_t _id;
    /** \brief Config file path (JSON) */
    std::string _config;
public:
    DataBroker(uint64_t id, std::string config) : _id(id),  _config(config) {}
    // We disallow copying for safety reasons
    DataBroker(const DataBroker&) = delete;
    DataBroker& operator=(const DataBroker&) = delete;

    /** \brief Define the type of the broker (RabbitMQ etc) */
    virtual std::string type() const = 0;

    /**
     * Takes a data buffer and push it onto the libevent buffer.
     * @param[in] num_elements Number of elements of each 1-D vector
     * @param[in] data Array containing 'num_elements' values to be sent
     */
    virtual void push(ssize_t num_elements, TypeValue* data) const = 0;
    
    /**
     * Takes an input and an output vector each holding 1-D vectors data, and push it onto the libevent buffer.
     * @param[in] num_elements Number of elements of each 1-D vector
     * @param[in] inputs Vector of 1-D vectors containing the inputs to be sent
     * @param[in] inputs Vector of 1-D vectors, each 1-D vectors contains 'num_elements' values to be sent
     * @param[in] outputs Vector of 1-D vectors, each 1-D vectors contains 'num_elements' values to be sent
     */
    virtual void push(ssize_t num_elements, std::vector<TypeValue*>& inputs, std::vector<TypeValue*>& outputs) const = 0;

    /**
     * Make sure the buffer is being drained. This function blocks until the 
     * buffer is empty (every byte has been sent to the broker).
     * @param[in] sleep_time Number of seconds between two checking (active pooling)
     */
    virtual void drain(int sleep_time = 1) = 0;
    virtual ~DataBroker() {}
};
} // namespace ams

#ifdef __ENABLE_RMQ__

#include <pthread.h>
#include <amqpcpp.h>
#include <amqpcpp/linux_tcp.h>
#include <event2/event.h>
#include <event2/event-config.h>
#include <amqpcpp/libevent.h>
#include <openssl/ssl.h>
#include <openssl/opensslv.h>

#include "rabbitmq_handler.hpp"

typedef std::unordered_map<std::string, std::string> json;

namespace ams {

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
* \brief Class that manages a RabbitMQ broker and handles connection, event loop and
* set up various handlers.
*/
template <typename TypeValue>
class RabbitMQBroker final : public DataBroker<TypeValue> {
private:
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
    EventBuffer* _evbuffer;
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
                // std::cerr << "key=" << key << " val=" << line << std::endl;
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
    RabbitMQBroker(const RabbitMQBroker&) = delete;
    RabbitMQBroker& operator=(const RabbitMQBroker&) = delete;

    RabbitMQBroker(uint64_t id, std::string config) : DataBroker<TypeValue>(id, config),
            _rank(0), _handler(nullptr), _evbuffer(nullptr), _address(nullptr), _worker(nullptr) {

        auto rmq_config = _read_config(this->_config);
#ifdef __ENABLE_MPI__
        MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &_rank));
#endif
#ifdef EVTHREAD_USE_PTHREADS_IMPLEMENTED
        evthread_use_pthreads();
#endif
        _loop = event_base_new();
        // fprintf(stderr, "[rank=%d][ info ] Libevent %s\n", _rank, event_get_version());
        // fprintf(stderr,
        //     "[rank=%d][ info ] %s (OPENSSL_VERSION_NUMBER = %#010x)\n",
        //     _rank, OPENSSL_VERSION_TEXT, OPENSSL_VERSION_NUMBER
        // );
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
            fprintf(stderr, "something is wrong\n");
            throw std::runtime_error("address is NULL");
        }

        fprintf(stderr, "RabbitMQ address: %s:%d/%s (queue = %s)\n", 
                    _address->hostname().c_str(),
                    _address->port(),
                    _address->vhost().c_str(),
                    _queue_name.c_str()
        );

        start(*_address, _queue_name);
        // mandatory to give some time to OpenSSL and RMQ to set things up, otherwise it will fail
        // TODO: find a way to remove that magic sleep
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
                fprintf(stderr,
                    "[WARNING][rank=%d] declared queue: %s (messagecount=%d, consumercount=%d)\n",
                    _rank, queue.c_str(), messagecount, consumercount
                );
            }
        }).onError([queue, &_rank = _rank](const char *message) {
            fprintf(stderr,
                "[ERROR][rank=%d] Error while creating broker queue (%s): %s\n",
                _rank, queue.c_str(), message
            );
            // TODO: throw dedicated excpetion or/and try to recover
            // from it (re-trying to open a queue, testing if the RM server is alive etc)
            throw std::runtime_error(message);
        });

        _worker = new struct worker;
        _worker->loop = _loop;
        _evbuffer = new EventBuffer(_rank, _loop, _channel, queue);
        
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
    void drain(int sleep_time = 1) override {
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

    /**
     * Takes a data buffer and push it onto the libevent buffer.
     * @param[in] num_elements Number of elements of each 1-D vector
     * @param[in] data Array containing 'num_elements' values to be sent
     */
    void push(ssize_t num_elements, TypeValue* data) const override {
        ssize_t datlen = num_elements * sizeof(TypeValue);
        fprintf(stderr, "push() send inputs: %d elements, size %d B\n", num_elements, datlen);
        _evbuffer->push(static_cast<void*>(data), datlen);
        // Necessary for some reasons, other the event buffer overheat and potentially segfault
        // TODO: investigate, error -> "[err] buffer.c:1066: Assertion chain || datlen==0 failed in evbuffer_copyout"
        sleep(1);
    }

    /**
     * Takes an input and an output vector each holding 1-D vectors data, and push it onto the libevent buffer.
     * @param[in] num_elements Number of elements of each 1-D vector
     * @param[in] inputs Vector of 1-D vectors containing the inputs to be sent
     * @param[in] inputs Vector of 1-D vectors, each 1-D vectors contains 'num_elements'  values to be sent
     * @param[in] outputs Vector of 1-D vectors, each 1-D vectors contains 'num_elements'  values to be sent
     */
    void push(ssize_t num_elements, std::vector<TypeValue*>& inputs, std::vector<TypeValue*>& outputs) const override {
        ssize_t datlen = num_elements * sizeof(TypeValue);
        for (int i = 0; i < inputs.size(); ++i) {
            // fprintf(stderr, "[i=%d] send inputs: %d elements, size %d B\n", i, num_elements, datlen);
            _evbuffer->push(static_cast<void*>(inputs[i]), datlen);
            // Necessary for some reasons, other the event buffer overheat and potentially segfault
            sleep(1);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            // fprintf(stderr, "[i=%d] send outputs: %d elements, size %d B\n", i, num_elements, datlen);
            _evbuffer->push(static_cast<void*>(outputs[i]), datlen);
            sleep(1);
        }
    }

    ~RabbitMQBroker() {
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
        free(_connection); //Segfault with delete? -> might be heap corruption
    }

    /** \brief Return the type of this broker */
    inline std::string type() const override {
        return "RabbitMQ";
    }
}; // class RabbitMQBroker

} //namespace ams

#endif // __ENABLE_RMQ__

/**
* Create an object of the respective data broker.
* @param[in] config_path path to the config file containing logging info (JSON)
* @param[in] type Type of the broker to create (right now only RabbitMQ is supported)
* @param[in] rId a unique Id for each process taking part in a distributed execution (rank-id)
* @return DataBroker pointer for the new broker
*/
template <typename TypeValue>
ams::DataBroker<TypeValue> *createDataBroker(char *config_path, AMSBrokerType brokerType, uint64_t rId = 0) {
    if (config_path == nullptr) {
        std::cerr << " [WARNING] Path of broker configuration is NULL, Please provide a valid path to enable a broker\n";
        std::cerr << " [WARNING] Continueing\n";
        return nullptr;
    }

    switch (brokerType) {
#ifdef __ENABLE_RMQ__
    case AMSBrokerType::RMQ:
        std::cout << "Broker config path " << config_path << "\n";
        return new ams::RabbitMQBroker<TypeValue>(rId, config_path);
#endif
    default:
        return nullptr;
    }
}

#endif // __AMS_DATA_BROKER_HPP__
