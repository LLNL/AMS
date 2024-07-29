#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import argparse
import pika
import sys
import ssl
import json
import struct
from typing import Tuple

import numpy as np

def ams_header_format() -> str:
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

def ams_endianness() -> str:
    """
    '=' means native endianness in standart size (system).
    See https://docs.python.org/3/library/struct.html#format-characters
    """
    return "="

def ams_encode_message(num_elem: int, domain_name: str, input_dim: int, output_dim: int, dtype_byte: int = 4) -> bytes:
    """
    For debugging and testing purposes, this function encode a message identical to what AMS would send
    """
    header_format = ams_endianness() + ams_header_format()
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

def ams_parse_creds(json_file: str) -> dict:
    """
    Parse the credentials to retrieve connection informations
    """
    data = {}
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def main(args: dict):
    conn = ams_parse_creds(args.creds)
    if args.tls_cert is None:
        ssl_context = None
    else:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_verify_locations(args.tls_cert)
        ssl_context = pika.SSLOptions(context)

    credentials = pika.PlainCredentials(conn["rabbitmq-user"], conn["rabbitmq-password"])
    cp = pika.ConnectionParameters(
        host=conn["service-host"],
        port=conn["service-port"],
        virtual_host=conn["rabbitmq-vhost"],
        credentials=credentials,
        ssl_options=ssl_context
    )

    connection = pika.BlockingConnection(cp)
    channel = connection.channel()
    result = channel.queue_declare(queue = args.queue, exclusive = False)
    queue_name = result.method.queue

    encoded_msg = ams_encode_message(
        num_elem = args.num_elem,
        domain_name = args.domain_name,
        input_dim = args.input_dim,
        output_dim = args.output_dim,
        dtype_byte = args.data_type
    )
    for i in range(1, args.num_msg+1):
        channel.basic_publish(exchange='', routing_key = args.routing_key, body = encoded_msg)
        print(f"[{i}/{args.num_msg}] Sent message with {args.num_elem} elements of dim=({args.input_dim},{args.output_dim}) elements on queue='{queue_name}'/routing_key='{args.routing_key}'")
    connection.close()

def parse_args() -> dict:

    parser = argparse.ArgumentParser(description="Tool that sends AMS-compatible messages to a RabbitMQ server")
    parser.add_argument('-c', '--creds', help="Credentials file (JSON)", required=True)
    parser.add_argument('-t', '--tls-cert', help="TLS certificate file", required=False)
    parser.add_argument('-q', '--queue', help="On which queue to send messages (default = '')", default='', required=False)
    parser.add_argument('-r', '--routing-key', help="Routing key for the messages", required=True)

    parser.add_argument('-m', '--num-msg', type=int, help="Number of messages that will get sent (default: 1)", default=1)
    parser.add_argument('-n', '--num-elem', type=int, help="Number of elements per message", required=True)
    parser.add_argument('-i', '--input-dim', type=int, help="Input dimensions (default: 2)", default=2)
    parser.add_argument('-o', '--output-dim', type=int, help="Output dimensions (default: 4)",  default=4)
    parser.add_argument('-d', '--data-type', type=int, help="Data size in bytes: float (4) or double (8) (default: 4)", choices=[4, 8], default=4)
    parser.add_argument('-x', '--domain-name', type=str, help="Domain name", default="domain_test")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
