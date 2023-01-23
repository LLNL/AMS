#ifndef __AMS_RMQ_BROKER_HPP__
#define __AMS_RMQ_BROKER_HPP__

#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>

#include <amqpcpp.h>
#include <amqpcpp/linux_tcp.h>
#include <ev.h>
#include <amqpcpp/libev.h>
#include <openssl/ssl.h>
#include <openssl/opensslv.h>

#include "data_broker.hpp"
#include "rmq_handler.hpp"

typedef std::unordered_map<std::string, std::string> json;

namespace ams {

class BrokerRMQ : public DataBroker {
private:
    std::string _config;
    HandlerRMQ* _handler;
    struct ev_loop* _loop; // Default event loop
public:
    BrokerRMQ(std::string config) : _config(config), _handler(nullptr) {
        // access to the event loop
        _loop = EV_DEFAULT;
        json rmq_config = read_json(_config);

        // init the SSL library
#if OPENSSL_VERSION_NUMBER < 0x10100000L
        SSL_library_init();
#else
        OPENSSL_init_ssl(0, NULL);
#endif

        _handler = new HandlerRMQ(_loop, rmq_config["cert"]);
        std::cout << "[ info ] creating handler." << std::endl;

        uint16_t port = std::stoi(rmq_config["service-port"]);
        bool is_secure = true;
        
        AMQP::Login login(rmq_config["rabbitmq-user"], rmq_config["rabbitmq-password"]);
        AMQP::Address address(rmq_config["host"], port, login, rmq_config["rabbitmq-vhost"], is_secure); 
        
        std::cout << "[ info ] RabbitMQ address: " 
            << address.hostname() << ":" 
            << address.port() << "/" 
            << address.vhost() 
            << std::endl;
        
        AMQP::TcpConnection connection(_handler, address);
        std::cout << "[ info ] created connection." << std::endl;
        
        AMQP::TcpChannel channel(&connection);
        channel.onError([](const char* message) {
            std::cerr << "[error] while creating channel: " << message << std::endl;
        });
        std::cout << "[ info ] creating channel." << std::endl;
        std::string queue_name = "ams";

        //channel.declareExchange("my-exchange", AMQP::fanout);
        channel.declareQueue(queue_name).onSuccess([&queue_name]() {
            std::cout << "[ info ] declared queue: " << queue_name << std::endl;
        }).onError([&queue_name](const char *message) {
            std::cerr << "[ fail ] while creating queue (" << queue_name << "): " << message << std::endl;
        });

        channel.onReady([&channel, &queue_name]() {
            std::cout << "[ info ] channel is ready to use!" << std::endl;
            // send the first instructions (like publishing messages)
        
            // start a transaction
            channel.startTransaction();

            // publish a number of messages
            channel.publish("", queue_name, "hello world from AMS");
            //channel.publish("my-exchange", "my-key", "another message");

            // commit the transactions, and set up callbacks that are called when
            // the transaction was successful or not
            channel.commitTransaction().onSuccess([]() {
                // all messages were successfully published
                std::cout << "[ info ] messages were sucessfuly published!" << std::endl;
            })
            .onError([](const char *message) {
                // none of the messages were published
                // now we have to do it all over again
                std::cerr << "[ fail ] messages did not get send: " << message << std::endl;
          });
        });

        // run the event loop
        ev_run(_loop, 0);

        // close the channel
        channel.close().onSuccess([&connection, &channel]() {
            // report that channel was closed
            std::cout << "[ info ] channel closed." << std::endl;       
            // close the connection
            connection.close();
        });
    }

    ~BrokerRMQ() {
        delete _handler;
    }

    inline std::string type() const {
        return  "RabbitMQ";
    }

    json read_json(std::string fn) {
        std::ifstream config;
        json connection_info = {
            {"rabbitmq-erlang-cookie", ""},
            {"rabbitmq-name", ""},
            {"rabbitmq-password", ""},
            {"rabbitmq-user", ""},
            {"rabbitmq-vhost", ""},
            {"service-port", ""},
            {"host", ""},
            {"cert", ""},
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
}; // class BrokerRMQ

} //namespace ams

#endif
