#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pika
import sys
import ssl
import json
import argparse

# JSON file containing host, port, password etc..
# PDS_JSON = "creds.json"
PDS_JSON = "rmq-pds.json"

# CA Cert, can be generated with (where $REMOTE_HOST and $REMOTE_PORT can be found in the JSON file):
#   openssl s_client -connect $REMOTE_HOST:$REMOTE_PORT -showcerts < /dev/null 2>/dev/null | sed -ne '/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p' > rmq-pds.crt

# CA_CERT  = "creds.pem"
CA_CERT  = "rmq-pds.crt"

def get_rmq_connection(json_file):
    data = {}
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def callback(ch, method, properties, body, args):
    data = body.decode()
    print(properties)
    print(f"Received \"{data}\" from exchange=\"{method.exchange}\" routing_key=\"{method.routing_key}\" args={args}")


def main(args):
    conn = get_rmq_connection(args.creds)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = False
    context.load_verify_locations(args.tls_cert)

    print(f"[send.py] Connecting to {conn['service-host']} ...")

    credentials = pika.PlainCredentials(conn["rabbitmq-user"], conn["rabbitmq-password"])
    cp = pika.ConnectionParameters(
        host=conn["service-host"],
        port=conn["service-port"],
        virtual_host=conn["rabbitmq-vhost"],
        credentials=credentials,
        ssl_options=pika.SSLOptions(context)
    )

    connection = pika.BlockingConnection(cp)
    channel = connection.channel()

    
    # Turn on delivery confirmations
    channel.confirm_delivery()

    result = channel.queue_declare(queue='', exclusive=False)

    queue_name = result.method.queue
    for i in range(args.num_msg):
      try:
        channel.basic_publish(exchange=args.exchange, routing_key=args.routing_key, body=args.msg)
        print(f" [{i}] Sent '{args.msg}' on exchange='{args.exchange}'/routing_key='{args.routing_key}'")
      except pika.exceptions.UnroutableError:
        print(f" [{i}] Message could not be confirmed")
    connection.close()

def parse_args():

    parser = argparse.ArgumentParser(description="Tool that sends AMS-compatible messages to a RabbitMQ server")
    parser.add_argument('-c', '--creds', help="Credentials file (JSON)", required=True)
    parser.add_argument('-t', '--tls-cert', help="TLS certificate file", required=True)
    parser.add_argument('-e', '--exchange', help="On which exchange to send messages (default = '')", default='', required=False)
    parser.add_argument('-r', '--routing-key', help="Routing key for the messages", required=True)
    parser.add_argument('-n', '--num-msg', type=int, help="Number of messages that will get sent (default: 1)", default=1)
    parser.add_argument('-m', '--msg', type=str, help="Content of the message")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
