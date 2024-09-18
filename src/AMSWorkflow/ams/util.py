# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import datetime
import socket
import uuid
from pathlib import Path


def get_unique_fn():
    # Randomly generate the output file name. We use the uuid4 function with the socket name and the current
    # date,time to create a unique filename with some 'meaning'.

    fn = [
        uuid.uuid4().hex,
        socket.gethostname(),
        str(datetime.datetime.now()).replace("-", "D").replace(" ", "T").replace(":", "C").replace(".", "p"),
    ]
    return "_".join(fn)


def mkdir(root_path, fn):
    _tmp = root_path / Path(fn)
    if not _tmp.exists():
        _tmp.mkdir(parents=True, exist_ok=True)
    return _tmp
