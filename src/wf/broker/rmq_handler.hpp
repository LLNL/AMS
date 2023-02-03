#ifndef __AMS_RMQ_HANDLER_HPP__
#define __AMS_RMQ_HANDLER_HPP__

#include <iostream>
#include <event2/event.h>
#include <amqpcpp.h>
#include <amqpcpp/libevent.h>
#include <openssl/ssl.h>
#include <openssl/opensslv.h>
#include <openssl/err.h>

namespace ams {

class HandlerRMQ : public AMQP::LibEventHandler {
private:
    const char* _cacert;
public:
    /**
     *  Constructor
     *  @param  ev_loop         EV Loop
     *  @param  cacert          SSL Cacert
     */
    HandlerRMQ(struct event_base *loop, std::string cacert) : AMQP::LibEventHandler(loop), _cacert(cacert.c_str()) {}
    virtual ~HandlerRMQ() {
        delete _cacert;
    }
private:
   /**
     *  Method that is called by the AMQP library when a new connection
     *  is associated with the handler. This is the first call to your handler
     *  @param  connection      The connection that is attached to the handler
     */
    virtual void onAttached(AMQP::TcpConnection *connection) override {
        // @todo
        //  add your own implementation, for example initialize things
        //  to handle the connection.
        //std::cout << "Connection attached to the handler!" << std::endl;
    }

    /**
     *  Method that is called by the AMQP library when the TCP connection 
     *  has been established. After this method has been called, the library
     *  still has take care of setting up the optional TLS layer and of
     *  setting up the AMQP connection on top of the TCP layer., This method 
     *  is always paired with a later call to onLost().
     *  @param  connection      The connection that can now be used
     */
    virtual void onConnected(AMQP::TcpConnection *connection) override {
        // @todo
        //  add your own implementation (probably not needed)
        std::cout << "[ info ] connected!" << std::endl;
    }

    /**
     *  Method that is called after a TCP connection has been set up, and right before
     *  the SSL handshake is going to be performed to secure the connection (only for
     *  amqps:// connections). This method can be overridden in user space to load
     *  client side certificates.
     *  @param  connection      The connection for which TLS was just started
     *  @param  ssl             Pointer to the SSL structure that can be modified
     *  @return bool            True to proceed / accept the connection, false to break up
     */
    virtual bool onSecuring(AMQP::TcpConnection *connection, SSL *ssl) {
        ERR_clear_error();
        unsigned long err;
        int ret = SSL_use_certificate_chain_file(ssl, _cacert);

        if (ret != 1) {
            std::string error("openssl: error loading ca-chain from [");
            SSL_get_error(ssl, ret);
            if ((err = ERR_get_error())) {
                error += std::string(ERR_reason_error_string(err));
            }
            error += "]";
            throw std::runtime_error(error);
        } else {
            std::cout << "[  ok  ] sucess logged with ca-chain " << _cacert << std::endl;
            return true;
        }
    }

    /**
     *  Method that is called when the secure TLS connection has been established. 
     *  This is only called for amqps:// connections. It allows you to inspect
     *  whether the connection is secure enough for your liking (you can
     *  for example check the server certificate). The AMQP protocol still has
     *  to be started.
     *  @param  connection      The connection that has been secured
     *  @param  ssl             SSL structure from openssl library
     *  @return bool            True if connection can be used
     */
    virtual bool onSecured(AMQP::TcpConnection *connection, const SSL *ssl) override {
        // @todo
        //  add your own implementation, for example by reading out the
        //  certificate and check if it is indeed yours        
        std::cout << "[ info ] Secured TLS connection has been established" << std::endl;
        return true;
    }

    /**
     *  Method that is called by the AMQP library when the login attempt
     *  succeeded. After this the connection is ready to use.
     *  @param  connection      The connection that can now be used
     */
    virtual void onReady(AMQP::TcpConnection *connection) override {
        // @todo
        //  add your own implementation, for example by creating a channel
        //  instance, and start publishing or consuming
        std::cout << "[  ok  ] Sucessfuly logged in. Connection ready to use!" << std::endl;
    }

    /**
     *  Method that is called by the AMQP library when a fatal error occurs
     *  on the connection, for example because data received from RabbitMQ
     *  could not be recognized, or the underlying connection is lost. This
     *  call is normally followed by a call to onLost() (if the error occurred
     *  after the TCP connection was established) and onDetached().
     *  @param  connection      The connection on which the error occurred
     *  @param  message         A human readable error message
     */
    virtual void onError(AMQP::TcpConnection *connection, const char *message) override {
        // @todo
        //  add your own implementation, for example by reporting the error
        //  to the user of your program and logging the error
        std::cerr << "[ fail ] TCP connection: " << message << std::endl;
    }

    /**
     *  Method that is called when the AMQP protocol is ended. This is the
     *  counter-part of a call to connection.close() to graceful shutdown
     *  the connection. Note that the TCP connection is at this time still 
     *  active, and you will also receive calls to onLost() and onDetached()
     *  @param  connection      The connection over which the AMQP protocol ended
     */
    virtual void onClosed(AMQP::TcpConnection *connection) override {
        // @todo
        //  add your own implementation (probably not necessary, but it could
        //  be useful if you want to do some something immediately after the
        //  amqp connection is over, but do not want to wait for the tcp 
        //  connection to shut down
        std::cout << "[ info ] AMQP connection is now closed." << std::endl;
    }

    /**
     *  Method that is called when the TCP connection was closed or lost.
     *  This method is always called if there was also a call to onConnected()
     *  @param  connection      The connection that was closed and that is now unusable
     */
    virtual void onLost(AMQP::TcpConnection *connection) override {
        // @todo
        //  add your own implementation (probably not necessary)
        std::cout << "[ info ] TCP connection is now closed (or lost)." << std::endl;
    }

    /**
     *  Final method that is called. This signals that no further calls to your
     *  handler will be made about the connection.
     *  @param  connection      The connection that can be destructed
     */
    virtual void onDetached(AMQP::TcpConnection *connection) override 
    {
        // @todo
        //  add your own implementation, like cleanup resources or exit the application
        // std::cout << "[info] Handler is detached from connections." << std::endl;
    }

};

} // namespace ams
#endif
