# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pika
import ssl
import sys
import os
import logging
import json

class RMQChannel:
    """
        A wrapper around RMQ channel
    """

    def __init__(self, connection, q_name):
        self.connection = connection
        self.q_name = q_name

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def callback(method, properties, body):
        return body.decode("utf-8")

    def open(self):
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue = self.q_name)

    def close(self):
        self.channel.close()

    def receive(self, n_msg: int = None, accum_msg = list()):
        """
        Consume a message on the queue and post processing by calling the callback.
        @param n_msg The number of messages to receive.
            - if n_msg is None, this call will block for ever and will process all messages that arrives
            - if n_msg = 1 for example, this function will block until one message has been processed.
        @return a list containing all received messages
        """

        if self.channel and self.channel.is_open:
            self.logger.info(
                f"Starting to consume messages from queue={self.q_name}, routing_key={self.routing_key} ..."
            )
            # we will consume only n_msg and requeue all other messages
            # if there are more messages in the queue.
            # It will block as long as n_msg did not get read
            if n_msg:
                n_msg = max(n_msg, 0)
                message_consumed = 0
                # Comsume n_msg messages and break out
                for method_frame, properties, body in self.channel.consume(self.q_name):
                    # Call the call on the message parts
                    try:
                        accum_msg.append(
                            RMQClient.callback(
                                method_frame,
                                properties,
                                body,
                            )
                        )
                    except Exception as e:
                        self.logger.error(f"Exception {type(e)}: {e}")
                        self.logger.debug(traceback.format_exc())
                    finally:
                        # Acknowledge the message even on failure
                        self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)
                    self.logger.warning(
                        f"Consumed message {message_consumed+1}/{method_frame.delivery_tag} (exchange={method_frame.exchange}, routing_key={method_frame.routing_key})"
                    )
                    message_consumed += 1
                    # Escape out of the loop after nb_msg messages
                    if message_consumed == n_msg:
                        # Cancel the consumer and return any pending messages
                        self.channel.cancel()
                        break
        return accum_msg

    def send(self, text: str):
        """
        Send a message
        @param text The text to send
        """
        self.channel.basic_publish(exchange="", routing_key=self.q_name, body=text)
        return

    def get_messages(self):
        return  # messages

    def purge(self):
        """Removes all the messages from the queue."""
        if self.channel and self.channel.is_open:
            self.channel.queue_purge(self.q_name)


class RMQClient:
    """
    RMQClient is a class that manages the RMQ client lifecycle.
    """
    def __init__(self, host, port, vhost, user, password, cert, logger: logging.Logger = None):
        # CA Cert, can be generated with (where $REMOTE_HOST and $REMOTE_PORT can be found in the JSON file):
        # openssl s_client -connect $REMOTE_HOST:$REMOTE_PORT -showcerts < /dev/null 2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' rmq-pds.crt
        self.logger = logger if logger else logging.getLogger(__name__)
        self.cert = cert
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        self.context.verify_mode = ssl.CERT_REQUIRED
        self.context.load_verify_locations(self.cert)
        self.host = host
        self.vhost = vhost
        self.port = port
        self.user = user
        self.password = password

        self.credentials = pika.PlainCredentials(
            self.user, self.password)

        self.connection_params = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=self.credentials,
            ssl_options=pika.SSLOptions(self.context),
        )

    def __enter__(self):
        self.connection = pika.BlockingConnection(self.connection_params)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

    def connect(self, queue):
        """Connect to the queue"""
        return RMQChannel(self.connection, queue)

