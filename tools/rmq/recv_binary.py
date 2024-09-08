#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import pika
import ssl
import sys
import os
import json
import copy
import numpy as np

from typing import Tuple

# CA Cert, can be generated with (where $REMOTE_HOST and $REMOTE_PORT can be found in the JSON file):
#   openssl s_client -connect $REMOTE_HOST:$REMOTE_PORT -showcerts < /dev/null
#   2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' > rabbitmq-credentials.cacert

nbmsg = 0
all_messages = []
byte_received = 0

def get_rmq_connection(json_file):
    data = {}
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def parse_header(body: str) -> dict:
    """
    We encode the message as follow:
      - 1 byte is the size of the header (here 16). Limit max: 255
      - 1 byte is the precision (4 for float, 8 for double). Limit max: 255
      - 2 bytes are the MPI rank (0 if AMS is not running with MPI). Limit max: 65535
      - 2 bytes to store the size of the MSG domain name. Limit max: 65535
      - 4 bytes are the number of elements in the message. Limit max: 2^32 - 1
      - 2 bytes are the input dimension. Limit max: 65535
      - 2 bytes are the output dimension. Limit max: 65535

        |_Header_|_Datatype_|___Rank___|__DomainSize__|__#elems__|___InDim____|___OutDim___|.real data
     
    Then the data starts at 16 and is structered as pairs of input/outputs.
    Let K be the total number of elements, then we have K pairs of inputs/outputs (either float or double):

        |__Header_(16B)__|__Input 1__|__Output 1__|...|__Input_K__|__Output_K__|
    """

    if len(body) == 0:
        print(f"Empty message. skipping")
        return {}

    header_size = np.frombuffer(body[0:1], dtype=np.uint8)[0]
    res = {}

    if header_size != 16:
        print(f"Incomplete message of size {len(body)}. Header size is {header_size}, it should be of size 16. skipping ({body})")
        return {}

    try:
        res["header_size"] = header_size
        res["datatype"] = np.frombuffer(body[1:2], dtype=np.uint8)[0]
        res["mpirank"] = np.frombuffer(body[2:4], dtype=np.uint16)[0]
        res["domain_size"] = np.frombuffer(body[4:6], dtype=np.uint16)[0]
        res["num_element"] = np.frombuffer(body[6:10], dtype=np.uint32)[0]
        res["input_dim"] = np.frombuffer(body[10:12], dtype=np.uint16)[0]
        res["output_dim"] = np.frombuffer(body[12:14], dtype=np.uint16)[0]
        res["padding"] = np.frombuffer(body[14:16], dtype=np.uint16)[0]
        # Theoritical size in Bytes for the incoming message (without the header)
        # Int() is needed otherwise we might overflow here (because of uint16 / uint8)
        res["data_size"] = int(res["datatype"]) * res["num_element"] * (int(res["input_dim"]) + int(res["output_dim"])+res["padding"])
        res["multiple_msg"] = len(body) != (header_size + res["data_size"])
    except ValueError as e:
        return {}
    return res

def multiple_messages(body: str) -> bool:
    return parse_header(body)["multiple_msg"]

def parse_data(body: str, header_info: dict) -> Tuple[str, np.array, np.array]:
    data = np.array([])
    if len(body) == 0:
        return data

    header_size = header_info["header_size"]
    data_size = header_info["data_size"]
    domain_name_size = header_info["domain_size"]
    domain_name = body[header_size : header_size + domain_name_size]
    domain_name = domain_name.decode("utf-8")

    try:
        if data_size == 4: #if datatype takes 4 bytes
            data = np.frombuffer(body[header_size+domain_name_size:header_size+domain_name_size+data_size], dtype=np.float32)
        else:
            data = np.frombuffer(body[header_size+domain_name_size:header_size+domain_name_size+data_size], dtype=np.float64)
    except ValueError as e:
        print(f"Error: {e} => {header_info}")

    idim = header_info["input_dim"]
    odim = header_info["output_dim"]
    data = data.reshape((-1, idim + odim))
    # Return input, output
    return (domain_name, data[:, :idim], data[:, idim:])

def callback(ch, method, properties, body, args = None):
    global nbmsg
    global all_messages
    global byte_received
    nbmsg += 1
    byte_received += len(body)

    i = 1
    stream = copy.deepcopy(body)
    while stream:
        header_info = parse_header(stream)
        if not header_info:
            break
        domain_name, data_input, data_output = parse_data(stream, header_info)
        num_element = header_info["num_element"]
        mpirank = header_info["mpirank"]
        # total size of byte we read for that message
        chunk_size = header_info["header_size"] + header_info["domain_size"] + header_info["data_size"]

        print(
            f" [{nbmsg}/{i}] Received from exchange=\"{method.exchange}\" routing_key=\"{method.routing_key}\"\n"
            f"        > [r{mpirank}] ({domain_name})   : {len(stream)/(1024*1024)} MB / {num_element} elements\n")

        if data_input.size > 0:
            all_messages.append(data_input)
        # We remove the current message and keep going
        stream = stream[chunk_size:]
        i += 1

def main(credentials: str, cacert: str, queue: str):
    conn = get_rmq_connection(credentials)
    if cacert is None:
        ssl_options = None
    else:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = False
        context.load_verify_locations(cacert)
        ssl_options = pika.SSLOptions(context)

    credentials = pika.PlainCredentials(conn["rabbitmq-user"], conn["rabbitmq-password"])
    cp = pika.ConnectionParameters(
        host=conn["service-host"],
        port=conn["service-port"],
        virtual_host=conn["rabbitmq-vhost"],
        credentials=credentials,
        ssl_options=ssl_options
    )

    connection = pika.BlockingConnection(cp)
    channel = connection.channel()

    print(f"Connecting to {conn['service-host']} ...")

    # Warning:
    #   if no queue is specified then RabbitMQ will NOT hold messages that are not routed to queues.
    #   So in order to receive the message, the receiver will have to be started BEFORE the sender
    #   Otherwise the message will be lost.

    result = channel.queue_declare(queue=queue, exclusive=False)
    queue_name = result.method.queue
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    print(f"Listening on queue = {queue_name}")

    print(" [*] Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()

def parse_args():
    parser = argparse.ArgumentParser(description="Tools that consumes AMS-encoded messages from RabbitMQ queue")
    parser.add_argument('-c', '--creds', help="Credentials file (JSON)", required=True)
    parser.add_argument('-t', '--tls-cert', help="TLS certificate file", required=False)
    parser.add_argument('-q', '--queue', help="Queue to listen to", required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    try:
        args = parse_args()
        main(credentials = args.creds, cacert = args.tls_cert, queue = args.queue)
    except KeyboardInterrupt:
        print("")
        print("Done")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
