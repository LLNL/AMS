# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import datetime
import socket
import uuid


def get_unique_fn():
    # Randomly generate the output file name. We use the uuid4 function with the socket name and the current
    # date,time to create a unique filename with some 'meaning'.

    fn = [
        uuid.uuid4().hex,
        socket.gethostname(),
        str(datetime.datetime.now()).replace("-", "D").replace(" ", "T").replace(":", "C").replace(".", "p"),
    ]
    return "_".join(fn)
