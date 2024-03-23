#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import os
import sys

from ams.rmq import BlockingClient


def main():
    parser = argparse.ArgumentParser(description="AMS Broker interface to send/receive messages.")

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        help="Path to broker configuration file",
        required=True,
    )

    parser.add_argument(
        "-t",
        "--certificate",
        dest="certificate",
        help="Path to TLS certificate file",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--send",
        dest="msg_send",
        type=str,
        help="Message to send",
        required=True,
    )

    parser.add_argument(
        "-q",
        "--queue",
        dest="queue",
        type=str,
        help="Queue to which the message will be sent",
        required=True,
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"Error: config file {args.config} does not exist")
        sys.exit(1)

    if not os.path.isfile(args.certificate):
        print(f"Error: certificate file {args.certificate} does not exist")
        sys.exit(1)

    with open(args.config, "r") as fd:
        config = json.load(fd)

    host = config["service-host"]
    vhost = config["rabbitmq-vhost"]
    port = config["service-port"]
    user = config["rabbitmq-user"]
    password = config["rabbitmq-password"]

    with BlockingClient(host, port, vhost, user, password, args.certificate) as client:
        with client.connect(args.queue) as channel:
            channel.send(args.msg_send)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
