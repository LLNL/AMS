# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import datetime
import socket
import subprocess
import uuid
from pathlib import Path

from typing import Tuple

def get_unique_fn():
    # Randomly generate the output file name. We use the uuid4 function with the socket name and the current
    # date,time to create a unique filename with some 'meaning'.

    fn = [
        uuid.uuid4().hex,
        socket.gethostname(),
        str(datetime.datetime.now()).replace("-", "D").replace(" ", "T").replace(":", "C").replace(".", "p"),
    ]
    return "_".join(fn)

def generate_tls_certificate(host: str, port: int) -> Tuple[bool,str]:
    """Generate TLS certificate for RabbitMQ

        :param str host: The RabbitMQ hostname
        :param int port: The RabbitMQ port

        :rtype: Tuple[bool,str]
        :return: return a tuple with a boolean set to True if certificate got generated and the TLS certificate (other contains stderr)
    """
    openssl = subprocess.run(["openssl", "s_client", "-connect", f"{host}:{port}", "-showcerts"], check=True, capture_output=True)
    if openssl.returncode != 0:
        return False, openssl.stderr.decode().strip()
    sed = subprocess.run(["sed", "-ne", r"/-BEGIN CERTIFICATE-/,/-END CERTIFICATE-/p"],  input=openssl.stdout, capture_output=True)
    if sed.returncode != 0:
        return False, sed.stderr.decode().strip()
    return True, sed.stdout.decode().strip()

def mkdir(root_path, fn):
    _tmp = root_path / Path(fn)
    if not _tmp.exists():
        _tmp.mkdir(parents=True, exist_ok=True)
    return _tmp
