#ifndef __AMS_DATA_BROKER_HPP__
#define __AMS_DATA_BROKER_HPP__

#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>
#include <algorithm> // for std::remove

#include "AMS.h"

namespace ams {
/**
* @brief Abstract class that defines what AMS expect from a message broker.
*/
class DataBroker {
protected:
    /** @brief unique id of the process running this simulation */
    uint64_t _id;
    /** @brief Config file path (JSON) */
    std::string _config;
public:
    DataBroker(uint64_t id, std::string config) : _id(id),  _config(config) {}
    // We disallow copying for safety reasons
    DataBroker(const DataBroker&) = delete;
    DataBroker& operator= (const DataBroker&) = delete;

    /** @brief Define the type of the broker (RabbitMQ etc) */
    virtual std::string type() const = 0;
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

struct worker {
    AMQP::TcpConnection* connection;
    AMQP::TcpChannel* channel;
    struct event_base* loop;
    pthread_t id;
};

void* thread_func(void* arg) {
    struct worker* w = (struct worker*) arg;
    event_base_dispatch(w->loop);
    return NULL;
}

class RabbitMQBroker : public DataBroker {
private:
    /** @brief MPI rank (if MPI is used, otherwise 0) */
    int _rank;
    /** @brief The event loop (usually the default one in libevent) */
    struct event_base* _loop;
    /** @brief The handler which contains various callbacks */
    RabbitMQHandler* _handler;
    /** @brief evbuffer that is responsible to offload data to RabbitMQ*/
    EventBuffer* _evbuffer;
    /** @brief Signal Handler, important to exit the event loop */
    SignalHandler* _sighandler;

    /** 
     * @brief Read a JSON and create a hashmap
     * @param[in] fn Path of the RabbitMQ JSON config file
     * @return a hashmap (std::unordered_map) of the JSON file
    */
    json read_json(std::string fn) {
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
        };

        config.open(fn, std::ifstream::in);
        
        if (config.is_open()) {
            std::string line;
            while (std::getline(config, line)) {
              if (line.find("{") != std::string::npos || line.find("}") != std::string::npos) { continue; }
              line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
              line.erase(std::remove(line.begin(), line.end(), ','), line.end());
              line.erase(std::remove(line.begin(), line.end(), '"'), line.end());
      
              std::string key = line.substr(0, line.find(':'));
              line.erase(0, line.find(":") + 1);
              connection_info[key] = line;
              //std::cerr << "key=" << key << " and value=" << line << std::endl;
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
    // RabbitMQBroker(const RabbitMQBroker&) = delete;
    // RabbitMQBroker& operator= (const RabbitMQBroker&) = delete;
    
    RabbitMQBroker(uint64_t id, std::string config) : DataBroker(id, config) {
        _rank = 0;
        _handler = nullptr;
        _evbuffer = nullptr;
        _sighandler = nullptr;

        json rmq_config = read_json(_config);
#ifdef __ENABLE_MPI__
        MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &_rank));
#endif
#ifdef EVTHREAD_USE_PTHREADS_IMPLEMENTED
        evthread_use_pthreads();
#endif
        _loop = event_base_new();
        event_base_dump_events(_loop, stderr);

        fprintf(stderr, "%s = (OPENSSL_VERSION_NUMBER=%#010x)\n", OPENSSL_VERSION_TEXT, OPENSSL_VERSION_NUMBER);
#if OPENSSL_VERSION_NUMBER < 0x10100000L
        SSL_library_init();
#else
        OPENSSL_init_ssl(0, NULL);
#endif
        _handler = new RabbitMQHandler(_rank, _loop, rmq_config["rabbitmq-cert"]);
        fprintf(stderr, "[rank=%d][ info ] creating handler.\n", _rank);
        std::string _queue_name = "test3";

        uint16_t port = std::stoi(rmq_config["rabbitmq-port"]);
        bool is_secure = true;

        AMQP::Login login(rmq_config["rabbitmq-user"], rmq_config["rabbitmq-password"]);
        AMQP::Address address(rmq_config["rabbitmq-host"], port, login, rmq_config["rabbitmq-vhost"], is_secure); 

        fprintf(stderr, "[rank=%d][ info ] RabbitMQ address: %s:%d/%s\n", _rank, address.hostname().c_str(), address.port(), address.vhost().c_str());

        AMQP::TcpConnection connection(_handler, address);
        fprintf(stderr, "[rank=%d][ info ] created connection.\n", _rank);

        AMQP::TcpChannel channel(&connection);
        channel.onError([&_rank = _rank](const char* message) {
            fprintf(stderr, "[rank=%d][error] while creating channel: %s\n", _rank, message);
        });

        fprintf(stderr, "[rank=%d][ info ] creating channel.\n", _rank);
        channel.onReady([&_rank = _rank]() {
            fprintf(stderr, "[rank=%d][ info ] channel is ready to use!\n", _rank);
        });

        //channel.declareExchange("my-exchange", AMQP::fanout);
        channel.declareQueue(_queue_name).onSuccess([&_queue_name = _queue_name, &_rank = _rank](const std::string &name, uint32_t messagecount, uint32_t consumercount) {
            fprintf(stderr, "[rank=%d][ info ] declared queue: %s (message_count=%d, consumer_count=%d)\n", _rank, _queue_name.c_str(), messagecount, consumercount);
        }).onError([&_queue_name = _queue_name, &_rank = _rank](const char *message) {
            fprintf(stderr, "[rank=%d][ fail ] while creating queue (%s): %s\n", _rank, _queue_name.c_str(), message);
        });

        int k = 10;
        size_t datlen = k*sizeof(double);
        double* data = (double*)malloc(datlen);

        _sighandler = new SignalHandler(_rank, _loop, SIGUSR1);
        fprintf(stderr, "[rank=%d][ info ] SignalHandler is ready to use!\n", _rank);

        _evbuffer = new EventBuffer(_rank, _loop, &channel, _queue_name);
        fprintf(stderr, "[rank=%d][ info ] EventBuffer is ready to use!\n", _rank);

        struct worker* w = (struct worker*) malloc(sizeof(struct worker));
        w->loop = _loop;
        if (pthread_create(&w->id, NULL, thread_func, w)) {
            perror("error pthread_create");
            exit(-1);
        }

        for (int i = 0; i < k; i++) {
            data[i] = (double) i+1;
        }
        _evbuffer->push(data, k);

        // we stop the loop in the worker thread if we drained all bytes that have
        // been committed to that shared buffer
        while(true) {
            if (_evbuffer->is_drained()) {
                pthread_kill(w->id, SIGUSR1);
                break;
            }
            sleep(1);
            // fprintf(stderr, "[rank=%d][ info ] buffer size = %d, byte_sent = %d.\n", _rank, _evbuffer->size(), _evbuffer->get_byte_to_send());
        }

        pthread_join(w->id, NULL);
        // close the channel
        channel.close().onSuccess([&connection, &channel, &_rank = _rank]() {
            fprintf(stderr, "[rank=%d][ info ] channel closed.\n", _rank);
            // close the connection
            connection.close();
        });

        free(w);
        free(data);
    }

    ~RabbitMQBroker() {
        event_base_free(_loop);
        delete _handler;
        delete _evbuffer;
        delete _sighandler;
    }

    /** @brief Return the type of this broker */
    inline std::string type() const {
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
* @return a DataBroker pointer for the new broker
*/
ams::DataBroker *createDataBroker(char *config_path, AMSBrokerType brokerType, uint64_t rId = 0) {
    if (config_path == nullptr) {
        std::cerr << " [WARNING] Path of broker configuration is NULL, Please provide a valid path to enable a broker\n";
        std::cerr << " [WARNING] Continueing\n";
        return nullptr;
    }

    switch (brokerType) {
#ifdef __ENABLE_RMQ__
    case AMSBrokerType::RMQ:
        return new ams::RabbitMQBroker(rId, config_path);
#endif
    default:
        return nullptr;
    }
}

#endif // __AMS_DATA_BROKER_HPP__
