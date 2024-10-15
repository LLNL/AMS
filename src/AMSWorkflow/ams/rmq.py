# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
import functools
import logging
import ssl
import struct
import traceback
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import json
import pika

class AMSMessage(object):
    """
    Represents a RabbitMQ incoming message from AMSLib.

    Attributes:
        body: The body of the message as received from RabbitMQ
    """

    def __init__(self, body: str):
        self.body = body

        self.num_elements = None
        self.hsize = None
        self.dtype_byte = None
        self.mpi_rank = None
        self.domain_name_size = None
        self.domain_names = []
        self.input_dim = None
        self.output_dim = None

    def __str__(self):
        dt = "float" if self.dtype_byte == 4 else 8
        if not self.dtype_byte:
            dt = None
        return f"AMSMessage(domain={self.domain_names}, #mpi={self.mpi_rank}, num_elements={self.num_elements}, datatype={dt}, input_dim={self.input_dim}, output_dim={self.output_dim})"

    def __repr__(self):
        return self.__str__()

    def header_format(self) -> str:
        """
        This string represents the AMS format in Python pack format:
        See https://docs.python.org/3/library/struct.html#format-characters
        - 1 byte is the size of the header (here 12). Limit max: 255
        - 1 byte is the precision (4 for float, 8 for double). Limit max: 255
        - 2 bytes are the MPI rank (0 if AMS is not running with MPI). Limit max: 65535
        - 2 bytes to store the size of the MSG domain name. Limit max: 65535
        - 4 bytes are the number of elements in the message. Limit max: 2^32 - 1
        - 2 bytes are the input dimension. Limit max: 65535
        - 2 bytes are the output dimension. Limit max: 65535
        - 2 bytes are for aligning memory to 8

            |_Header_|_Datatype_|_Rank_|_DomainSize_|_#elems_|_InDim_|_OutDim_|_Pad_|_DomainName_|.Real_Data.|

        Then the data starts at byte 16 with the domain name, then the real data and
        is structured as pairs of input/outputs. Let K be the total number of elements,
        then we have K pairs of inputs/outputs (either float or double):

            |__Header_(16B)__|_Domain_Name_|__Input 1__|__Output 1__|...|__Input_K__|__Output_K__|

        """
        return "BBHHIHHH"

    def endianness(self) -> str:
        """
        '=' means native endianness in standart size (system).
        See https://docs.python.org/3/library/struct.html#format-characters
        """
        return "="

    def encode(self, num_elem: int, domain_name: str, input_dim: int, output_dim: int, dtype_byte: int = 4) -> bytes:
        """
        For debugging and testing purposes, this function encode a message identical to what AMS would send
        """
        header_format = self.ams_endianness() + self.ams_header_format()
        hsize = struct.calcsize(header_format)
        assert dtype_byte in [4, 8]
        dt = "f" if dtype_byte == 4 else "d"
        mpi_rank = 0
        data = np.random.rand(num_elem * (input_dim + output_dim))
        domain_name_size = len(domain_name)
        domain_name = bytes(domain_name, "utf-8")
        padding = 0
        header_content = (hsize, dtype_byte, mpi_rank, domain_name_size, data.size, input_dim, output_dim, padding)
        # float or double
        msg_format = f"{header_format}{domain_name_size}s{data.size}{dt}"
        return struct.pack(msg_format, *header_content, domain_name, *data)

    def _parse_header(self, body: str) -> dict:
        """
        Parse the header to extract information about data.
        """
        fmt = self.endianness() + self.header_format()
        if len(body) == 0:
            print("Empty message. skipping")
            return {}

        hsize = struct.calcsize(fmt)
        res = {}
        # Parse header
        (
            res["hsize"],
            res["datatype"],
            res["mpirank"],
            res["domain_size"],
            res["num_element"],
            res["input_dim"],
            res["output_dim"],
            res["padding"],
        ) = struct.unpack(fmt, body[:hsize])
        assert hsize == res["hsize"]
        assert res["datatype"] in [4, 8]
        if len(body) < hsize:
            print(f"Incomplete message of size {len(body)}. Header should be of size {hsize}. skipping")
            return {}

        # Theoritical size in Bytes for the incoming message (without the header)
        # Int() is needed otherwise we might overflow here (because of uint16 / uint8)
        res["dsize"] = int(res["datatype"]) * int(res["num_element"]) * (int(res["input_dim"]) + int(res["output_dim"]))
        res["msg_size"] = hsize + res["dsize"]
        res["multiple_msg"] = len(body) != res["msg_size"]

        self.num_elements = int(res["num_element"])
        self.hsize = int(res["hsize"])
        self.dtype_byte = int(res["datatype"])
        self.mpi_rank = int(res["mpirank"])
        self.domain_name_size = int(res["domain_size"])
        self.input_dim = int(res["input_dim"])
        self.output_dim = int(res["output_dim"])

        return res

    def _parse_data(self, body: str, header_info: dict) -> Tuple[str, np.array, np.array]:
        data = np.array([])
        if len(body) == 0:
            return data
        hsize = header_info["hsize"]
        dsize = header_info["dsize"]
        domain_name_size = header_info["domain_size"]
        domain_name = body[hsize : hsize + domain_name_size]
        domain_name = domain_name.decode("utf-8")
        try:
            if header_info["datatype"] == 4:  # if datatype takes 4 bytes (float)
                data = np.frombuffer(
                    body[hsize + domain_name_size : hsize + domain_name_size + dsize], dtype=np.float32
                )
            else:
                data = np.frombuffer(
                    body[hsize + domain_name_size : hsize + domain_name_size + dsize], dtype=np.float64
                )
        except ValueError as e:
            print(f"Error: {e} => {header_info}")
            return np.array([])

        idim = header_info["input_dim"]
        odim = header_info["output_dim"]
        data = data.reshape((-1, idim + odim))
        # Return input, output
        return (domain_name, data[:, :idim], data[:, idim:])

    def _decode(self, body: str) -> Tuple[np.array]:
        input = []
        output = []
        # Multiple AMS messages could be packed in one RMQ message
        # TODO: we should manage potential mutliple messages per AMSMessage better
        while body:
            header_info = self._parse_header(body)
            domain_name, temp_input, temp_output = self._parse_data(body, header_info)
            # print(f"MSG: {domain_name} input shape {temp_input.shape} outpute shape {temp_output.shape}")
            # total size of byte we read for that message
            chunk_size = header_info["hsize"] + header_info["dsize"] + header_info["domain_size"]
            input.append(temp_input)
            output.append(temp_output)
            # We remove the current message and keep going
            body = body[chunk_size:]
            self.domain_names.append(domain_name)
        return domain_name, np.concatenate(input), np.concatenate(output)

    def decode(self) -> Tuple[str, np.array, np.array]:
        return self._decode(self.body)

def default_ams_callback(method, properties, body):
    """Simple callback that decode incoming message assuming they are AMS binary messages"""
    return AMSMessage(body)

class AMSChannel:
    """
    A wrapper around Pika RabbitMQ channel
    """

    def __init__(self, connection, q_name, callback: Optional[Callable] = None, logger: Optional[logging.Logger] = None):
        self.connection = connection
        self.q_name = q_name
        self.logger = logger if logger else logging.getLogger(__name__)
        self.callback = callback if callback else self.default_callback

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def default_callback(self, method, properties, body):
        """ Simple callback that return the message received"""
        return body

    def open(self):
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.q_name)

    def close(self):
        self.channel.close()

    def receive(self, n_msg: int = None, timeout: int = None, accum_msg = list()):
        """
        Consume a message on the queue and post processing by calling the callback.
        @param n_msg The number of messages to receive.
            - if n_msg is None, this call will block for ever and will process all messages that arrives
            - if n_msg = 1 for example, this function will block until one message has been processed.
        @param timeout If None, timout infinite, otherwise timeout in seconds
        @return a list containing all received messages
        """

        if self.channel and self.channel.is_open:
            self.logger.info(
                f"Starting to consume messages from queue={self.q_name} ..."
            )
            # we will consume only n_msg and requeue all other messages
            # if there are more messages in the queue.
            # It will block as long as n_msg did not get read
            if n_msg:
                n_msg = max(n_msg, 0)
                message_consumed = 0
                # Comsume n_msg messages and break out
                for method_frame, properties, body in self.channel.consume(self.q_name, inactivity_timeout=timeout):
                    if (method_frame, properties, body) == (None, None, None):
                        self.logger.info(f"Timeout after {timeout} seconds")
                        self.channel.cancel()
                        break
                    # Call the call on the message parts
                    try:
                        accum_msg.append(
                            self.callback(
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
                        message_consumed += 1
                    self.logger.warning(
                        f"Consumed message {message_consumed}/{method_frame.delivery_tag} (exchange=\'{method_frame.exchange}\', routing_key={method_frame.routing_key})"
                    )
                    # Escape out of the loop after nb_msg messages
                    if message_consumed == n_msg:
                        # Cancel the consumer and return any pending messages
                        self.channel.cancel()
                        break
        return accum_msg

    def send(self, text: str, exchange : str = ""):
        """
        Send a message
        @param text The text to send
        @param exchange Exchange to use
        """
        self.channel.basic_publish(exchange=exchange, routing_key=self.q_name, body=text)
        return

    def get_messages(self):
        return  # messages

    def purge(self):
        """Removes all the messages from the queue."""
        if self.channel and self.channel.is_open:
            self.channel.queue_purge(self.q_name)

class BlockingClient:
    """
    BlockingClient is a class that manages a simple blocking RMQ client lifecycle.
    """

    def __init__(
        self,
        host: str,
        port: int,
        vhost: str,
        user: str,
        password: str,
        cert: Optional[str] = None,
        callback: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None
    ):
        # CA Cert, can be generated with (where $REMOTE_HOST and $REMOTE_PORT can be found in the JSON file):
        # openssl s_client -connect $REMOTE_HOST:$REMOTE_PORT -showcerts < /dev/null 2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' rmq-pds.crt
        self.logger = logger if logger else logging.getLogger(__name__)
        self.cert = cert

        if self.cert is None or self.cert == "":
            ssl_options = None
        else:
            self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            self.context.verify_mode = ssl.CERT_REQUIRED
            self.context.check_hostname = False
            self.context.load_verify_locations(self.cert)
            ssl_options = pika.SSLOptions(self.context)

        self.host = host
        self.vhost = vhost
        self.port = port
        self.user = user
        self.password = password
        self.callback = callback

        self.credentials = pika.PlainCredentials(self.user, self.password)

        self.connection_params = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=self.credentials,
            ssl_options=ssl_options,
        )

    def __enter__(self):
        self.connection = pika.BlockingConnection(self.connection_params)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()

    def connect(self, queue):
        """Connect to the queue"""
        return AMSChannel(self.connection, queue, self.callback)


class AsyncConsumer(object):
    """
    Asynchronous RMQ consumer. AsyncConsumer handles unexpected interactions
    with RabbitMQ such as channel and connection closures. AsyncConsumer can
    receive messages but cannot send messages.
    """

    def __init__(
        self,
        host: str,
        port: int,
        vhost: str,
        user: str,
        password: str,
        cert: str,
        queue: str,
        prefetch_count: int = 1,
        on_message_cb: Optional[Callable] = None,
        on_close_cb: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Create a new instance of the consumer class, passing in the AMQP
        URL used to connect to RabbitMQ.

        :param str credentials: The credentials file in JSON
        :param str cacert: The TLS certificate
        :param str queue: The queue to listen to
        :param Callable: on_message_cb this function will be called each time Pika receive a message
        :param Callable: on_close_cb this function will be called when Pika will close the connection
        :param int: prefetch_count Define consumer throughput, should be relative to resource and number of messages expected

        """
        self._user = user
        self._passwd = password
        self._host = host
        self._port = port
        self._vhost = vhost
        self._cacert = cert
        self._queue = queue

        self.should_reconnect = False
        # Holds the latest error/reason to reconnect
        # Could be a Tuple like (200, 'Normal shutdown') or an exception from pika.AMQPError
        self.reconnect_reason = None
        self.was_consuming = False
        self.logger = logger if logger else logging.getLogger(__name__)

        self._connection = None
        self._connection_parameters = None
        self._channel = None
        self._closing = False
        self._consumer_tag = None
        self._consuming = False
        self._prefetch_count = prefetch_count
        self._on_message_cb = on_message_cb
        self._on_close_cb = on_close_cb

    def __enter__(self):
        self.run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def connection_params(self):
        """
        Create the pika credentials using TLS needed to connect to RabbitMQ.

        :rtype: pika.ConnectionParameters
        """
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        ssl_context.check_hostname = False
        ssl_context.load_verify_locations(self._cacert)

        pika_credentials = pika.PlainCredentials(self._user, self._passwd)
        return pika.ConnectionParameters(
            host=self._host,
            port=self._port,
            virtual_host=self._vhost,
            credentials=pika_credentials,
            ssl_options=pika.SSLOptions(ssl_context),
        )

    def connect(self):
        """This method connects to RabbitMQ, returning the connection handle.
        When the connection is established, the on_connection_open method
        will be invoked by pika.

        :rtype: pika.SelectConnection

        """
        self._connection_parameters = self.connection_params()
        self.logger.debug(f"Connecting to {self._host}")

        return pika.SelectConnection(
            parameters=self._connection_parameters,
            on_open_callback=self.on_connection_open,
            on_open_error_callback=self.on_connection_open_error,
            on_close_callback=self.on_connection_closed,
        )

    def close_connection(self):
        self._consuming = False
        if self._connection.is_closing or self._connection.is_closed:
            self.logger.debug("Connection is closing or already closed")
        else:
            self.logger.debug("Closing connection")
            self._connection.close()

    def on_connection_open(self, connection):
        """This method is called by pika once the connection to RabbitMQ has
        been established. It passes the handle to the connection object in
        case we need it, but in this case, we'll just mark it unused.

        :param pika.SelectConnection _unused_connection: The connection

        """
        assert self._connection is connection
        self.logger.debug("Connection opened")
        self.open_channel()

    def on_connection_open_error(self, _unused_connection, err):
        """This method is called by pika if the connection to RabbitMQ
        can't be established.

        :param pika.SelectConnection _unused_connection: The connection
        :param Exception err: The error

        """
        self.logger.error(f"Connection open failed: {err}")
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
            self.logger.warning(f"Connection closed, reconnect necessary: {reason}")
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
        self.logger.debug("Creating a new channel")
        self._connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        """This method is invoked by pika when the channel has been opened.
        The channel object is passed in so we can make use of it.

        Since the channel is now open, we'll declare the exchange to use.

        :param pika.channel.Channel channel: The channel object

        """
        self._channel = channel
        self.logger.debug("Channel opened")
        self.add_on_channel_close_callback()
        # we do not set up exchange first here, we use the default exchange ''
        self.setup_queue(self._queue)

    def add_on_channel_close_callback(self):
        """This method tells pika to call the on_channel_closed method if
        RabbitMQ unexpectedly closes the channel.

        """
        self.logger.debug("Adding channel close callback")
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
        self.logger.debug(f"Channel was closed. {reason}")
        if isinstance(self._on_close_cb, Callable):
            self._on_close_cb()  # running user callback
        self.close_connection()

    def setup_queue(self, queue_name):
        """Setup the queue on RabbitMQ by invoking the Queue.Declare RPC
        command. When it is complete, the on_queue_declareok method will
        be invoked by pika.

        :param str|unicode queue_name: The name of the queue to declare.

        """
        self.logger.debug(f'Declaring queue "{queue_name}"')
        cb = functools.partial(self.on_queue_declareok, userdata=queue_name)
        self._channel.queue_declare(queue=queue_name, exclusive=False, callback=cb)

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
        self.logger.debug(f'Queue "{queue_name}" declared')
        self.set_qos()

    def set_qos(self):
        """This method sets up the consumer prefetch to only be delivered
        one message at a time. The consumer must acknowledge this message
        before RabbitMQ will deliver another one. You should experiment
        with different prefetch values to achieve desired performance.

        """
        self._channel.basic_qos(prefetch_count=self._prefetch_count, callback=self.on_basic_qos_ok)

    def on_basic_qos_ok(self, _unused_frame):
        """Invoked by pika when the Basic.QoS method has completed. At this
        point we will start consuming messages by calling start_consuming
        which will invoke the needed RPC commands to start the process.

        :param pika.frame.Method _unused_frame: The Basic.QosOk response frame

        """
        self.logger.debug(f"QOS set to: {self._prefetch_count}")
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
        self.logger.debug("Issuing consumer related RPC commands")
        self.add_on_cancel_callback()
        self._consumer_tag = self._channel.basic_consume(self._queue, self.on_message, auto_ack=False)
        self.was_consuming = True
        self._consuming = True
        self.logger.info(f"Waiting for messages (tag: {self._consumer_tag}). To exit press CTRL+C")

    def add_on_cancel_callback(self):
        """Add a callback that will be invoked if RabbitMQ cancels the consumer
        for some reason. If RabbitMQ does cancel the consumer,
        on_consumer_cancelled will be invoked by pika.

        """
        self.logger.debug("Adding consumer cancellation callback")
        self._channel.add_on_cancel_callback(self.on_consumer_cancelled)

    def on_consumer_cancelled(self, method_frame):
        """Invoked by pika when RabbitMQ sends a Basic.Cancel for a consumer
        receiving messages.

        :param pika.frame.Method method_frame: The Basic.Cancel frame

        """
        self.logger.debug(f"Consumer was cancelled remotely, shutting down: {method_frame}")
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
        self.logger.info(f"Received message #{basic_deliver.delivery_tag} from {properties}")
        if isinstance(self._on_message_cb, Callable):
            self._on_message_cb(_unused_channel, basic_deliver, properties, body)
        self.acknowledge_message(basic_deliver.delivery_tag)

    def acknowledge_message(self, delivery_tag):
        """Acknowledge the message delivery from RabbitMQ by sending a
        Basic.Ack RPC method for the delivery tag.

        :param int delivery_tag: The delivery tag from the Basic.Deliver frame

        """
        self.logger.debug(f"Acknowledging message {delivery_tag}")
        self._channel.basic_ack(delivery_tag)

    def stop_consuming(self):
        """Tell RabbitMQ that you would like to stop consuming by sending the
        Basic.Cancel RPC command.

        """
        if self._channel:
            self.logger.debug("Sending a Basic.Cancel RPC command to RabbitMQ")
            cb = functools.partial(self.on_cancelok, userdata=self._consumer_tag)
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
        self.logger.debug(f"RabbitMQ acknowledged the cancellation of the consumer: {userdata}")
        self.close_channel()

    def close_channel(self):
        """Call to close the channel with RabbitMQ cleanly by issuing the
        Channel.Close RPC command.
        """
        self.logger.debug("Closing the channel")
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
        connection.
        """
        if not self._closing:
            self._closing = True
            self.logger.debug(" Stopping RabbitMQ connection")
            if self._consuming:
                self.stop_consuming()
            else:
                if self._connection:
                    self._connection.ioloop.stop()
            self.logger.debug("Stopped RabbitMQ connection")


class AsyncFanOutConsumer(AsyncConsumer):
    def __init__(
        self,
        host: str,
        port: int,
        vhost: str,
        user: str,
        password: str,
        cert: str,
        queue: str,
        prefetch_count: int = 1,
        on_message_cb: Optional[Callable] = None,
        on_close_cb: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(
            host, port, vhost, user, password, cert, queue, prefetch_count, on_message_cb, on_close_cb, logger
        )

    # Callback when the channel is open
    def on_channel_open(self, channel):
        self._channel = channel
        self.logger.debug("Channel opened")
        self.add_on_channel_close_callback()
        self._channel.exchange_declare(
            exchange="control-panel", exchange_type="fanout", callback=self.on_exchange_declared
        )

    # Callback when the exchange is declared
    def on_exchange_declared(self, frame):
        self._channel.queue_declare(queue="", exclusive=True, callback=self.on_queue_declared)

    # Callback when the queue is declared
    def on_queue_declared(self, queue_result):
        self._queue = queue_result.method.queue
        self._channel.queue_bind(exchange="control-panel", queue=self._queue, callback=self.on_queue_bound)

    # Callback when the queue is bound to the exchange
    def on_queue_bound(self, frame):
        self.set_qos()


class AMSSyncProducer:

    def __init__(
        self,
        host: str,
        port: int,
        vhost: str,
        user: str,
        password: str,
        cert: str,
        publish_queue: str,
        logger: Optional[logging.Logger] = None,
    ):

        self.host = host
        self.port = port
        self.vhost = vhost
        self.user = user
        self.password = password
        self.cert = cert
        self._connected = False
        self._publish_queue = publish_queue
        self._num_sent_messages = 0
        self._num_confirmed_messages = 0

    def open(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = False
        context.load_verify_locations(self.cert)
        credentials = pika.PlainCredentials(self.user, self.password)
        self.connection_parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=credentials,
            ssl_options=pika.SSLOptions(context),
        )

        self.connection = pika.BlockingConnection(self.connection_parameters)
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue=self._publish_queue, exclusive=False)
        # TODO: assert if publish_queue is different than method.queue.
        # Verify if this is guaranteed by the RMQ specification.
        self._publish_queue = result.method.queue
        self._connected = True
        return self

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def send_message(self, message):
        self._num_sent_messages += 1
        try:
            self.channel.basic_publish(exchange="", routing_key=self._publish_queue, body=message)
        except pika.exceptions.UnroutableError:
            print(f" [{self._num_sent_messages}] Message could not be confirmed")
        else:
            self._num_confirmed_messages += 1


class AMSFanOutProducer(AMSSyncProducer):

    def __init__(
        self,
        host: str,
        port: int,
        vhost: str,
        user: str,
        password: str,
        cert: str,
        logger: logging.Logger = None,
    ):
        super().__init__(host, port, vhost, user, password, cert, "control-panel", logger)

    def open(self):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = False
        context.load_verify_locations(self.cert)
        credentials = pika.PlainCredentials(self.user, self.password)
        self.connection_parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=credentials,
            ssl_options=pika.SSLOptions(context),
        )

        self.connection = pika.BlockingConnection(self.connection_parameters)
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange="control-panel", exchange_type="fanout")
        self._connected = True
        return self

    def close(self):
        self.channel.exchange_delete(exchange=self._publish_queue)

    def broadcast(self, message):
        self._num_sent_messages += 1
        try:
            self.channel.basic_publish(exchange="control-panel", routing_key="", body=message)
            print(f" [x] Sent '{message}'")
        except pika.exceptions.UnroutableError:
            print(f" [{self._num_sent_messages}] Message could not be confirmed")
        else:
            self._num_confirmed_messages += 1


@dataclass
class AMSRMQConfiguration:
    service_port: int
    service_host: str
    rabbitmq_erlang_cookie: str
    rabbitmq_name: str
    rabbitmq_password: str
    rabbitmq_user: str
    rabbitmq_vhost: str
    rabbitmq_cert: str
    rabbitmq_inbound_queue: str
    rabbitmq_outbound_queue: str
    rabbitmq_ml_submit_queue: str
    rabbitmq_ml_status_queue: str

    def __post_init__(self):
        if not Path(self.rabbitmq_cert).exists():
            raise RuntimeError(f"Certificate rmq path: {self.rabbitmq_cert} does not exist")

    @classmethod
    def from_json(cls, json_file):
        if not Path(json_file).exists():
            raise RuntimeError(f"Certificate rmq path: {json_file} does not exist")

        with open(json_file, "r") as fd:
            data = json.load(fd)
        data = {key.replace("-", "_"): value for key, value in data.items()}

        return cls(**data)

    def to_dict(self, AMSlib=False):
        assert AMSlib, "AMSRMQConfiguration cannot convert class to non amslib dictionary"
        if AMSlib:
            return {
                "service-port": self.service_port,
                "service-host": self.service_host,
                "rabbitmq-erlang-cookie": self.rabbitmq_erlang_cookie,
                "rabbitmq-name": self.rabbitmq_name,
                "rabbitmq-password": self.rabbitmq_password,
                "rabbitmq-user": self.rabbitmq_user,
                "rabbitmq-vhost": self.rabbitmq_vhost,
                "rabbitmq-cert": self.rabbitmq_cert,
                "rabbitmq-outbound-queue": self.rabbitmq_outbound_queue,
                "rabbitmq-exchange": "not-used",
                "rabbitmq-routing-key": "",
            }
        raise
