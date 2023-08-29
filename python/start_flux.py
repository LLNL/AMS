#!/usr/bin/env python3
import sys
import os
import subprocess
import json
from rmq import RMQClient


class StartFlux:
    """
    Class to manage Flux instance
    """

    def __init__(self, config=None):
        self.config = config
        self.rmq_client = RMQClient("rmq/rmq-pds.json", "rmq-pds.crt")
        self.rmq_client.connect("test3")
        tmp = self.rmq_client.receive("test3", n_msg=1).pop()
        self.uri = json.loads(tmp.decode("utf-8"))["ml_uri"]

    def start(self):
        flux_cmd = [
            "flux",
            "proxy",
            "--force",
            f"{self.uri}",
            "flux",
            "python",
            "daemon.py",
        ]
        subprocess.run(flux_cmd)


def main():
    f = StartFlux()
    f.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
