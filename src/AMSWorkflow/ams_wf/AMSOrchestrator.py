# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import os
from ams.rmq import RMQClient
from ams.orchestrator import AMSDaemon
from ams.orchestrator import FluxDaemonWrapper

import argparse


def main():
    daemon_actions = ["start", "wrap"]
    parser = argparse.ArgumentParser(
        description="AMS Machine Learning Daemon running on Training allocation"
    )
    parser.add_argument(
        "-a",
        "--action",
        dest='action',
        choices=daemon_actions,
        help="Decide whether to start daemon process directly or through flux wrap script",
        required=True,
    )

    parser.add_argument(
        "-c",
        "--certificate",
        dest="certificate",
        help="Path to certificate file to establish connection",
        required=True,
    )

    parser.add_argument(
        "-cfg",
        "--config",
        dest="config",
        help="Path to AMS configuration file",
        required=True,
    )

    args = parser.parse_args()
    if args.action == "start":
        daemon = AMSDaemon(args.config, args.certificate)
        # Busy wait for messages and spawn ML training jobs
        daemon()
    elif args.action == "wrap":
        daemon_cmd = [
            "python",
            __file__,
            "--action",
            "start",
            "-c",
            args.certificate,
            "-cfg",
            args.config,
        ]
        daemon = FluxDaemonWrapper(args.config, args.certificate)
        daemon(daemon_cmd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
