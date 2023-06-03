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

#include "wf/debug.h"

namespace ams {

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
            fprintf(stderr, "[rank=%d][  ok  ] sucess logged with ca-chain %s\n", _rank, _cacert);
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
        fprintf(stderr, "[rank=%d][ info ] Secured TLS connection has been established\n", _rank);
        return true;
    }

    /**
     *  \brief Method that is called by the AMQP library when the login attempt
     *  succeeded. After this the connection is ready to use.
     *  @param[in]  connection      The connection that can now be used
     */
    virtual void onReady(AMQP::TcpConnection *connection) override {
        fprintf(stderr, "[rank=%d][  ok  ] Sucessfuly logged in. Connection ready to use!\n", _rank);
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
        fprintf(stderr, "[rank=%d][ fail ] fatal error when establishing TCP connection: %s\n", _rank, message);
    }
};

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
            std::cout << k << ":" << result_b64.size() << std::endl;
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
};

} // namespace ams
#endif
