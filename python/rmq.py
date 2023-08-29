#!/usr/bin/env python3
import pika
import ssl
import sys
import os
import logging
import json

LOGGER = logging.getLogger(__name__)


class RMQClient:
    """
    RMQClient is a class that manages the RMQ client lifecycle.
    """

    def __init__(self, json_config, cert, logger: logging.Logger = LOGGER):
        # JSON file containing host, port, password etc..
        with open(json_config, "r") as f:
            self.config = json.load(f)
        # CA Cert, can be generated with (where $REMOTE_HOST and $REMOTE_PORT can be found in the JSON file):
        # openssl s_client -connect $REMOTE_HOST:$REMOTE_PORT -showcerts < /dev/null 2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' rmq-pds.crt
        self.logger = logger if logger else logging.getLogger(__name__)
        self.cert = cert
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        self.context.verify_mode = ssl.CERT_REQUIRED
        self.context.load_verify_locations(self.cert)

        self.credentials = pika.PlainCredentials(
            self.config["rabbitmq-user"], self.config["rabbitmq-password"]
        )
        self.connection_params = pika.ConnectionParameters(
            host=self.config["service-host"],
            port=self.config["service-port"],
            virtual_host=self.config["rabbitmq-vhost"],
            credentials=self.credentials,
            ssl_options=pika.SSLOptions(self.context),
        )

    @staticmethod
    def callback(method, properties, body):
        return body

    def connect(self, queue):
        self.connection = pika.BlockingConnection(self.connection_params)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=queue)

    def receive(self, queue, n_msg: int = None):
        """
        Consume a message on the queue and call the callback.
            - if n_msg is None, this call will block for ever and will process all messages that arrives
            - if n_msg = 1 for example, this function will block until one message has been processed.
        """
        accum_msg = []
        if self.channel and self.channel.is_open:
            self.logger.info(
                f"Starting to consume messages from queue={self.queue}, routing_key={self.routing_key} ..."
            )
            # we will consume only n_msg and requeue all other messages
            # if there are more messages in the queue.
            # It will block as long as n_msg did not get read
            if n_msg:
                n_msg = max(n_msg, 0)
                message_consumed = 0
                # Comsume n_msg messages and break out
                for method_frame, properties, body in self.channel.consume(queue):
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

    def send(self, queue, text):
        self.channel.basic_publish(exchange="", routing_key=queue, body=text)
        print(f" [x] Sent '{text}")
        # connection.close()

    def get_messages(self):
        return  # messages

    def purge(self, queue):
        """Remove all the messages from the queue (be careful!)."""
        if self.channel and self.channel.is_open:
            self.channel.queue_purge(queue)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
