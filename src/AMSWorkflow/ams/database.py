#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
import logging
import csv
import numpy as np
import os
import sys
from typing import Dict, List, Any, Union, Type


class DBInterface(ABC):
    """
    Represents a database instance in AMS.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        """
        Ensure subclass implement all the abstract method
        defined in the interface. Errors will be raised
        if all methods aren't overridden.
        """
        return (hasattr(subclass, "__str__") and
                callable(subclass.__str__) and
                hasattr(subclass, "open") and
                callable(subclass.open) and
                hasattr(subclass, "close") and
                callable(subclass.close) and
                hasattr(subclass, "store") and
                callable(subclass.store) or
                NotImplemented)

    @abstractmethod
    def __str__(self) -> str:
        """ Return a string representation of the broker """
        raise NotImplementedError

    def __repr__(self) -> str:
        """ Return a string representation of the broker """
        return self.__str__()

    @abstractmethod
    def open(self):
        """ Connect to the DB (or open file if file-based DB) """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """ Close DB """
        raise NotImplementedError

    @abstractmethod
    def store(self, inputs, outputs) -> int:
        """
        Store the two arrays using a given backend
        Return the number of characters written
        """
        raise NotImplementedError

class csvDB(DBInterface):
    """
    A simple CSV backend.
    """
    def __init__(self, file_name: str, delimiter: str = ':'):
        super().__init__()
        self.file_name = file_name
        self.delimiter = delimiter
        self.fd = None

    def __str__(self) -> str:
       return f"{__class__.__name__}(fd={self.fd}, delimiter={self.delimiter})"

    def open(self):
        self.fd = open(self.file_name, 'a')

    def close(self):
        self.fd.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def store(self, inputs: np.array, outputs: np.array) -> int:
        """ Store the two arrays in a CSV file """
        assert len(inputs) == len(outputs)
        if self.fd and self.fd.closed:
            return 0
        csvwriter = csv.writer(self.fd,
                delimiter = self.delimiter,
                quotechar = "'",
                quoting = csv.QUOTE_MINIMAL
            )
        nelem = len(inputs)
        elem_wrote: int = 0
        # We follow the mini-app format, inputs elem and then output elems
        for i in range(nelem):
            elem_wrote += csvwriter.writerow(np.concatenate((inputs[i], outputs[i]), axis=0))
        return elem_wrote
