#ifndef __AMS_RMQ_HANDLER_HPP__
#define __AMS_RMQ_HANDLER_HPP__

#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <signal.h>

#include <event2/event.h>
#include <event2/thread.h>
#include <event2/buffer.h>

#include <amqpcpp.h>
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

namespace ams {

/**
 * @brief TODO
 */
class RabbitMQHandler : public AMQP::LibEventHandler {
private:
    /** @brief Path to TLS certificate */
    const char* _cacert;
    /** @brief The MPI rank (0 if MPI is not used) */
    int _rank;

public:
    /**
     *  @brief Constructor
     *  @param[in]  loop         Event Loop
     *  @param[in]  cacert       SSL Cacert
     *  @param[in]  rank         MPI rank
     */
    RabbitMQHandler(int rank, struct event_base *loop, std::string cacert) : AMQP::LibEventHandler(loop), _rank(rank), _cacert(cacert.c_str()) {}
    virtual ~RabbitMQHandler() = default;

private:
    /**
     *  @brief Method that is called after a TCP connection has been set up, and right before
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
            fprintf(stderr, "[rank=%d][  ok  ] sucess logged with ca-chain %s\n", _rank, _cacert);
            return true;
        }
    }

    /**
     *  @brief Method that is called when the secure TLS connection has been established. 
     *  This is only called for amqps:// connections. It allows you to inspect
     *  whether the connection is secure enough for your liking (you can
     *  for example check the server certificate). The AMQP protocol still has
     *  to be started.
     *  @param[in]  connection      The connection that has been secured
     *  @param[in]  ssl             SSL structure from openssl library
     *  @return     bool            True if connection can be used
     */
    virtual bool onSecured(AMQP::TcpConnection *connection, const SSL *ssl) override {
        fprintf(stderr, "[rank=%d][ info ] Secured TLS connection has been established\n", _rank);
        return true;
    }

    /**
     *  @brief Method that is called by the AMQP library when the login attempt
     *  succeeded. After this the connection is ready to use.
     *  @param[in]  connection      The connection that can now be used
     */
    virtual void onReady(AMQP::TcpConnection *connection) override {
        fprintf(stderr, "[rank=%d][  ok  ] Sucessfuly logged in. Connection ready to use!\n", _rank);
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
    virtual void onError(AMQP::TcpConnection *connection, const char *message) override {
        fprintf(stderr, "[rank=%d][ fail ] fatal error when establishing TCP connection: %s\n", _rank, message);
    }
};

/**
 * @brief TODO
 */
class EventBuffer {
public:
    /**
     *  @brief Constructor
     *  @param[in]  loop
     *  @param[in]  channel
     *  @param[in]  queue
     */
    EventBuffer(int rank, struct event_base *loop, AMQP::TcpChannel *channel, std::string queue) : 
        _rank(rank), _buffer(nullptr), _byte_to_send(0), _channel(channel), _queue(std::move(queue)) {
        // initialize the libev buff event structure
        _buffer = evbuffer_new();
        // evbuffer_enable_locking(_buffer, NULL);
        evbuffer_add_cb(_buffer, callback, this);
        /*
         * Force all the callbacks on an evbuffer to be run, not immediately after
         * the evbuffer is altered, but instead from inside the event loop
         * Without that the call to callback() would block the main
         * thread
         */
        evbuffer_defer_callbacks(_buffer, loop);
    }

    size_t size() {
        return evbuffer_get_length(_buffer);
    }
    
    bool is_drained() {
        return _byte_to_send == 0;
    }
    
    size_t get_byte_to_send() {
        return _byte_to_send;
    }

    void push(double* data, ssize_t num_elem) {
        evbuffer_lock(_buffer);
        evbuffer_add(_buffer, data, num_elem*sizeof(data));
        _byte_to_send = _byte_to_send + num_elem*sizeof(data);
        evbuffer_unlock(_buffer);
    }

    char* encode64(char* input) {
        size_t unencoded_length = strlen(input);
        size_t encoded_length = base64_encoded_length(unencoded_length);
        char *base64_encoded_string = (char *)malloc(encoded_length);
        ssize_t encoded_size = base64_encode(base64_encoded_string, encoded_length, input, unencoded_length);
        // printf("%s -> %s\n", input, base64_encoded_string);
        // free(base64_encoded_string);
        return base64_encoded_string;
    }

    virtual ~EventBuffer() {
        evbuffer_free(_buffer);
    }
private:
    /** @brief The actual event structure */
    struct evbuffer* _buffer;
    /** @brief Pointer towards the AMQP channel */
    AMQP::TcpChannel *_channel;
    /** @brief Name of the RabbitMQ queue */
    std::string _queue;
    /** @brief Total number of bytes that must be send */
    size_t _byte_to_send;
    /** @brief MPI rank */
    int _rank;

    /**
     *  @brief  Callback method that is called by libevent when the timer expires
     *  @param  fd          The loop in which the event was triggered
     *  @param  event       Internal timer object
     *  @param  argc        The events that triggered this call
     */
    static void callback(struct evbuffer *buffer, const struct evbuffer_cb_info *info, void *arg) {
        // retrieve the this pointer
        EventBuffer *self = static_cast<EventBuffer*>(arg);
        int rank = self->_rank;
        fprintf(stderr, "[rank=%d][thread] before event_buffer_cb: orig_size=%d added=%d deleted=%d\n", rank, info->orig_size, info->n_added, info->n_deleted);

        // we remove only if some byte got added (this callback will get
        // trigger when data is added AND removed from the buffer
        if (info->n_added > 0) {
            // We read one double
            size_t datlen = info->n_added;
            int k = datlen/sizeof(double);
            double* data = (double*)malloc(datlen);

            evbuffer_lock(buffer);
            int nbyte_drained = evbuffer_remove(buffer, data, datlen);
            if (nbyte_drained < 0) perror("event_buffer_cb: cannot drain buffer");
            evbuffer_unlock(buffer);

            std::string result = "["+std::to_string(rank)+"] ";
            fprintf(stderr, "[rank=%d][thread] buffer contains %d bytes:\n[thread] ", rank, nbyte_drained);
            for (int i = 0; i < k-1; i++) {
                fprintf(stderr, "%.2f ", data[i]);
                result.append(std::to_string(data[i])+" ");
            }
            fprintf(stderr, "%.2f\n", data[k-1]);
            result.append(std::to_string(data[k-1]));

            // publish the data in the buffer
            self->_channel->startTransaction();
            self->_channel->publish("", self->_queue, result);
            self->_channel->commitTransaction().onSuccess([self, rank, nbyte_drained]() {
                fprintf(stderr, "[rank=%d][ info ] messages were sucessfuly published on queue=%s\n", rank, self->_queue.c_str());
                self->_byte_to_send = self->_byte_to_send - nbyte_drained;
            }).onError([self, rank, nbyte_drained](const char *message) {
                fprintf(stderr, "[rank=%d][ fail ] messages did not get send: %s\n", rank, message);
                self->_byte_to_send = self->_byte_to_send - nbyte_drained;
            });

            free(data);
        }
        fprintf(stderr, "[rank=%d][thread] after event_buffer_cb: orig_size=%d added=%d deleted=%d\n", rank, info->orig_size, info->n_added, info->n_deleted);
    }
};

/**
 * @brief Signal Handler that triggers a callback when a given signal
 * is caught. We use it to gracefully exit the event loop.
 */
class SignalHandler {
public:
    /**
     *  @brief Constructor
     *  @param[in]  rank MPI rank
     *  @param[in]  loop event loop used (must exist before)
     *  @param[in]  sig signal code that will be intercepted
     */
    SignalHandler(int rank, struct event_base *loop, int sig) : 
        _rank(rank), _event(nullptr), _loop(loop), _sig(sig) {
        _event = evsignal_new(loop, sig, callback, this);
        event_add(_event, NULL);
    }
    
    virtual ~SignalHandler() {
        event_free(_event);
    }

private:
    /** @brief Signal event */
    struct event* _event;
    /** @brief Event loop */
    struct event_base* _loop;
    /** @brief Signal code that will be intercepted */
    int _sig;
    /** @brief MPI rank */
    int _rank;

    /**
     *  @brief Callback method that is called by libevent when the signal sig is intercepted
     *  @param  fd          The loop in which the event was triggered
     *  @param  event       Internal event object (evsignal in this case)
     *  @param  argc        The events that triggered this call
     */
    static void callback(int fd, short event, void* argc) {
        SignalHandler *self = static_cast<SignalHandler*>(argc);
        fprintf(stderr,"[rank=%d][thread] caught an interrupt signal; exiting cleanly event loop...\n", self->_rank);
        event_base_loopbreak(self->_loop);
    }
};

} // namespace ams
#endif
