# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ssl
import sys
import os
import json
import logging
import re
import copy
import functools
import pika
from pika.exchange_type import ExchangeType
from typing import Callable
import numpy as np

class RMQConsumer(object):
    """
    Asynchronous RMQ consumer.
    RMQConsumer handles unexpected interactions
    with RabbitMQ such as channel and connection closures.
    """

    def __init__(self,
        credentials: str,
        cacert: str,
        queue: str,
        on_message_cb: Callable = None,
        on_close_cb: Callable = None,
        prefetch_count: int = 1):
        """Create a new instance of the consumer class, passing in the AMQP
        URL used to connect to RabbitMQ.

        :param str credentials: The credentials file in JSON
        :param str cacert: The TLS certificate
        :param str queue: The queue to listen to
        :param Callable: on_message_cb this function will be called each time Pika receive a message
        :param Callable: on_message_cb this function will be called when Pika will close the connection
        :param int: prefetch_count Define consumer throughput, should be relative to resource and number of messages expected 

        """
        self.should_reconnect = False
        # Holds the latest error/reason to reconnect
        # Could be a Tuple like (200, 'Normal shutdown') or an exception from pika.AMQPError
        self.reconnect_reason = None 
        self.was_consuming = False

        self._connection = None
        self._connection_parameters = None
        self._channel = None
        self._closing = False
        self._consumer_tag = None
        self._consuming = False
        self._prefetch_count = prefetch_count
        self._on_message_cb = on_message_cb 
        self._on_close_cb = on_close_cb 

        self._credentials = self._parse_credentials(credentials)
        self._cacert = cacert
        self._queue = queue 

    def __enter__(self):
        self.run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _parse_credentials(self, json_file: str) -> dict:
        """ Internal method to parse the credentials file"""
        data = {}
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def create_credentials(self):
        """
        Create the pika credentials using TLS needed to connect to RabbitMQ.

        :rtype: pika.ConnectionParameters

        """
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.load_verify_locations(self._cacert)

        pika_credentials = pika.PlainCredentials(self._credentials["rabbitmq-user"], self._credentials["rabbitmq-password"])
        return pika.ConnectionParameters(
            host=self._credentials["service-host"],
            port=self._credentials["service-port"],
            virtual_host=self._credentials["rabbitmq-vhost"],
            credentials=pika_credentials,
            ssl_options=pika.SSLOptions(ssl_context)
        )

    def connect(self):
        """This method connects to RabbitMQ, returning the connection handle.
        When the connection is established, the on_connection_open method
        will be invoked by pika.

        :rtype: pika.SelectConnection

        """
        self._connection_parameters = self.create_credentials()
        print(f"Connecting to {self._credentials['service-host']}")

        return pika.SelectConnection(
            parameters = self._connection_parameters,
            on_open_callback = self.on_connection_open,
            on_open_error_callback = self.on_connection_open_error,
            on_close_callback = self.on_connection_closed)

    def close_connection(self):
        self._consuming = False
        if self._connection.is_closing or self._connection.is_closed:
            print("Connection is closing or already closed")
        else:
            print("Closing connection")
            self._connection.close()

    def on_connection_open(self, _unused_connection):
        """This method is called by pika once the connection to RabbitMQ has
        been established. It passes the handle to the connection object in
        case we need it, but in this case, we'll just mark it unused.

        :param pika.SelectConnection _unused_connection: The connection

        """
        print("Connection opened")
        self.open_channel()

    def on_connection_open_error(self, _unused_connection, err):
        """This method is called by pika if the connection to RabbitMQ
        can't be established.

        :param pika.SelectConnection _unused_connection: The connection
        :param Exception err: The error

        """
        print(f"Error: Connection open failed: {err}")
        self.reconnect_reason = err
        self.reconnect()

    def on_connection_closed(self, _unused_connection, reason):
        """This method is invoked by pika when the connection to RabbitMQ is
        closed unexpectedly. Since it is unexpected, we will reconnect to
        RabbitMQ if it disconnects.

        :param pika.connection.Connection connection: The closed connection obj
        :param Exception reason: exception representing reason for loss of
            connection.

        """
        self._channel = None
        if self._closing:
            self._connection.ioloop.stop()
        else:
            print(f"warning: Connection closed, reconnect necessary: {reason}")
            self.reconnect_reason = reason
            self.reconnect()

    def reconnect(self):
        """Will be invoked if the connection can't be opened or is
        closed. Indicates that a reconnect is necessary then stops the
        ioloop.

        """
        self.should_reconnect = True
        self.stop()

    def open_channel(self):
        """Open a new channel with RabbitMQ by issuing the Channel.Open RPC
        command. When RabbitMQ responds that the channel is open, the
        on_channel_open callback will be invoked by pika.

        """
        print("Creating a new channel")
        self._connection.channel(on_open_callback = self.on_channel_open)

    def on_channel_open(self, channel):
        """This method is invoked by pika when the channel has been opened.
        The channel object is passed in so we can make use of it.

        Since the channel is now open, we'll declare the exchange to use.

        :param pika.channel.Channel channel: The channel object

        """
        self._channel = channel
        print(f"Channel opened {self._channel}")
        self.add_on_channel_close_callback()
        # we do not set up exchange first here, we use the default exchange ''
        self.setup_queue(self._queue)

    def add_on_channel_close_callback(self):
        """This method tells pika to call the on_channel_closed method if
        RabbitMQ unexpectedly closes the channel.

        """
        print("Adding channel close callback")
        self._channel.add_on_close_callback(self.on_channel_closed)

    def on_channel_closed(self, channel, reason):
        """Invoked by pika when RabbitMQ unexpectedly closes the channel.
        Channels are usually closed if you attempt to do something that
        violates the protocol, such as re-declare an exchange or queue with
        different parameters. In this case, we'll close the connection
        to shutdown the object.

        :param pika.channel.Channel: The closed channel
        :param Exception reason: why the channel was closed

        """
        print(f"warning: Channel {channel} was closed: {reason}")
        if isinstance(self._on_close_cb, Callable):
            self._on_close_cb() # running user callback
        self.close_connection()

    def setup_queue(self, queue_name):
        """Setup the queue on RabbitMQ by invoking the Queue.Declare RPC
        command. When it is complete, the on_queue_declareok method will
        be invoked by pika.

        :param str|unicode queue_name: The name of the queue to declare.

        """
        print(f"Declaring queue \"{queue_name}\"")
        cb = functools.partial(self.on_queue_declareok, userdata = queue_name)
        self._channel.queue_declare(queue = queue_name, exclusive=False, callback = cb)

    def on_queue_declareok(self, _unused_frame, userdata):
        """Method invoked by pika when the Queue.Declare RPC call made in
        setup_queue has completed. In this method we will bind the queue
        and exchange together with the routing key by issuing the Queue.Bind
        RPC command. When this command is complete, the on_bindok method will
        be invoked by pika.

        :param pika.frame.Method _unused_frame: The Queue.DeclareOk frame
        :param str|unicode userdata: Extra user data (queue name)

        """
        queue_name = userdata
        print(f"Queue \"{queue_name}\" declared")
        self.set_qos()

    def set_qos(self):
        """This method sets up the consumer prefetch to only be delivered
        one message at a time. The consumer must acknowledge this message
        before RabbitMQ will deliver another one. You should experiment
        with different prefetch values to achieve desired performance.

        """
        self._channel.basic_qos(
            prefetch_count = self._prefetch_count,
            callback = self.on_basic_qos_ok
        )

    def on_basic_qos_ok(self, _unused_frame):
        """Invoked by pika when the Basic.QoS method has completed. At this
        point we will start consuming messages by calling start_consuming
        which will invoke the needed RPC commands to start the process.

        :param pika.frame.Method _unused_frame: The Basic.QosOk response frame

        """
        print(f"QOS set to: {self._prefetch_count}")
        self.start_consuming()

    def start_consuming(self):
        """This method sets up the consumer by first calling
        add_on_cancel_callback so that the object is notified if RabbitMQ
        cancels the consumer. It then issues the Basic.Consume RPC command
        which returns the consumer tag that is used to uniquely identify the
        consumer with RabbitMQ. We keep the value to use it when we want to
        cancel consuming. The on_message method is passed in as a callback pika
        will invoke when a message is fully received.

        """
        print("Issuing consumer related RPC commands")
        self.add_on_cancel_callback()
        self._consumer_tag = self._channel.basic_consume(
            self._queue, self.on_message, auto_ack=False)
        self.was_consuming = True
        self._consuming = True
        print(" [*] Waiting for messages. To exit press CTRL+C")

    def add_on_cancel_callback(self):
        """Add a callback that will be invoked if RabbitMQ cancels the consumer
        for some reason. If RabbitMQ does cancel the consumer,
        on_consumer_cancelled will be invoked by pika.

        """
        print("Adding consumer cancellation callback")
        self._channel.add_on_cancel_callback(self.on_consumer_cancelled)

    def on_consumer_cancelled(self, method_frame):
        """Invoked by pika when RabbitMQ sends a Basic.Cancel for a consumer
        receiving messages.

        :param pika.frame.Method method_frame: The Basic.Cancel frame

        """
        print(f"Consumer was cancelled remotely, shutting down: {method_frame}")
        if self._channel:
            self._channel.close()

    def on_message(self, _unused_channel, basic_deliver, properties, body):
        """Invoked by pika when a message is delivered from RabbitMQ. The
        channel is passed for your convenience. The basic_deliver object that
        is passed in carries the exchange, routing key, delivery tag and
        a redelivered flag for the message. The properties passed in is an
        instance of BasicProperties with the message properties and the body
        is the message that was sent.

        :param pika.channel.Channel _unused_channel: The channel object
        :param pika.Spec.Basic.Deliver: basic_deliver method
        :param pika.Spec.BasicProperties: properties
        :param bytes body: The message body

        """
        print(f"Received message #{basic_deliver.delivery_tag} from {properties}")
        if isinstance(self._on_message_cb, Callable):
            self._on_message_cb(_unused_channel, basic_deliver, properties, body)
        self.acknowledge_message(basic_deliver.delivery_tag)

    def acknowledge_message(self, delivery_tag):
        """Acknowledge the message delivery from RabbitMQ by sending a
        Basic.Ack RPC method for the delivery tag.

        :param int delivery_tag: The delivery tag from the Basic.Deliver frame

        """
        print(f"Acknowledging message {delivery_tag}")
        self._channel.basic_ack(delivery_tag)

    def stop_consuming(self):
        """Tell RabbitMQ that you would like to stop consuming by sending the
        Basic.Cancel RPC command.

        """
        if self._channel:
            print(f"Sending a Basic.Cancel RPC command to RabbitMQ")
            cb = functools.partial(
                self.on_cancelok, userdata = self._consumer_tag)
            self._channel.basic_cancel(self._consumer_tag, cb)

    def on_cancelok(self, _unused_frame, userdata):
        """This method is invoked by pika when RabbitMQ acknowledges the
        cancellation of a consumer. At this point we will close the channel.
        This will invoke the on_channel_closed method once the channel has been
        closed, which will in-turn close the connection.

        :param pika.frame.Method _unused_frame: The Basic.CancelOk frame
        :param str|unicode userdata: Extra user data (consumer tag)

        """
        self._consuming = False
        print(f"RabbitMQ acknowledged the cancellation of the consumer: {userdata}")
        self.close_channel()

    def close_channel(self):
        """Call to close the channel with RabbitMQ cleanly by issuing the
        Channel.Close RPC command.

        """
        print("Closing the channel")
        self._channel.close()

    def run(self):
        """Run the example consumer by connecting to RabbitMQ and then
        starting the IOLoop to block and allow the SelectConnection to operate.

        """
        self._connection = self.connect()
        self._connection.ioloop.start()

    def stop(self):
        """Cleanly shutdown the connection to RabbitMQ by stopping the consumer
        with RabbitMQ. When RabbitMQ confirms the cancellation, on_cancelok
        will be invoked by pika, which will then closing the channel and
        connection. The IOLoop is started again because this method is invoked
        when CTRL-C is pressed raising a KeyboardInterrupt exception. This
        exception stops the IOLoop which needs to be running for pika to
        communicate with RabbitMQ. All of the commands issued prior to starting
        the IOLoop will be buffered but not processed.

        """
        if not self._closing:
            self._closing = True
            print("Stopping RabbitMQ connection")
            if self._consuming:
                self.stop_consuming()
                self._connection.ioloop.start()
            else:
                if self._connection:
                    self._connection.ioloop.stop()
            print("Stopped RabbitMQ connection")
        else:
            print("Already closed?")
