#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import csv
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import numpy as np


class FileReader(ABC):
    @classmethod
    def __subclasshook__(cls, subclass):
        """
        Ensure subclass implement all the abstract method
        defined in the interface. Errors will be raised
        if all methods aren't overridden.
        """
        return (
            hasattr(subclass, "__str__")
            and callable(subclass.__str__)
            and hasattr(subclass, "open")
            and callable(subclass.open)
            and hasattr(subclass, "close")
            and callable(subclass.close)
            and hasattr(subclass, "load")
            and callable(subclass.load)
            or NotImplemented
        )

    def _map_name_to_index(self, dsets_keys, name):
        # select keys of interest
        keys = [i for i in dsets_keys if name in i]
        keys = [(k, int(k.split("_")[-1])) for k in keys]
        return keys

    @abstractmethod
    def open(self):
        """Open File"""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Close File"""
        raise NotImplementedError

    @abstractmethod
    def load(self) -> tuple:
        """
        load the data in the file and return a tupple of the inputs, outputs
        """
        raise NotImplementedError


class CSVReader(FileReader):
    """
    A CSV File Reader
    """

    suffix = "csv"

    def __init__(self, file_name: str, delimiter: str = ":"):
        super().__init__()
        self.file_name = file_name
        self.delimiter = delimiter
        self.fd = None

    def open(self):
        self.fd = open(self.file_name, "r")
        return self

    def close(self):
        self.fd.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def load(self) -> tuple:
        """
        load the data in the file and return a tupple of the inputs, outputs

        We assume the file is produced by the C/C++ front-end. Thus the file
        will have a generic header row specifying the inputs/outputs

        Returns:
            A pair of input, output data values
        """

        if self.fd and self.fd.closed:
            return None, None

        file_data = list(csv.reader(self.fd, delimiter=self.delimiter))
        header = file_data[0]
        data = file_data[1:]
        output_start = header.index("output_0")
        data = np.array(data)
        input_data = data[:, :output_start]
        output_data = data[:, output_start:]
        return (input_data.astype(np.float64), output_data.astype(np.float64))

    @classmethod
    def get_file_format_suffix(cls):
        return cls.suffix


class HDF5CLibReader(FileReader):
    """
    An HDF5 reader for files generated directly by the C/C++ code.
    """

    suffix = "h5"

    def __init__(self, file_name: str):
        super().__init__()
        self.file_name = file_name
        self.fd = None

    def open(self):
        self.fd = h5py.File(self.file_name, "r")
        return self

    def close(self):
        self.fd.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _pack_dsets_to_list(self, dsets, selector):
        data = [dsets[k[0]] for k in selector]
        data = [data[k[1]] for k in selector]
        return data

    def load(self) -> tuple:
        """
        load the data in the file and return a tupple of the inputs, outputs

        We assume the file is produced by the C/C++ front-end. Thus the file
        will have a generic header row specifying the inputs/outputs

        Returns:
            A pair of input, output data values
        """

        dsets = self.fd.keys()
        input_map = self._map_name_to_index(dsets, "input")
        output_map = self._map_name_to_index(dsets, "output")

        input_data = np.array(self._pack_dsets_to_list(self.fd, input_map)).T
        output_data = np.array(self._pack_dsets_to_list(self.fd, output_map)).T

        return input_data, output_data

    @classmethod
    def get_file_format_suffix(cls):
        return cls.suffix


class HDF5PackedReader(FileReader):
    """
    load the data in the file and return a tupple of the inputs, outputs

    This reader DOES NOT assume the data being written by the application in C/C++.
    Instead it assumes 2 datasets in the hdf5 file one for the inputs and one for the outputs

    Returns:
        A pair of input, output numpy darrays
    """

    suffix = "h5"

    def __init__(self, file_name: str):
        super().__init__()
        self.file_name = file_name
        self.fd = None

    def open(self):
        self.fd = h5py.File(self.file_name, "r")
        return self

    def close(self):
        self.fd.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def load(self) -> tuple:
        """
        load the data in the file and return a tupple of the inputs, outputs
        """

        input_data = self.fd["inputs"]
        output_data = self.fd["outputs"]

        return np.array(input_data), np.array(output_data)

    @classmethod
    def get_file_format_suffix(cls):
        return cls.suffix


class FileWriter(ABC):
    """
    Represents a File to be written by AMS.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        """
        Ensure subclass implement all the abstract method
        defined in the interface. Errors will be raised
        if all methods aren't overridden.
        """
        return (
            hasattr(subclass, "__str__")
            and callable(subclass.__str__)
            and hasattr(subclass, "open")
            and callable(subclass.open)
            and hasattr(subclass, "close")
            and callable(subclass.close)
            and hasattr(subclass, "store")
            and callable(subclass.store)
            or NotImplemented
        )

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the FileWriter interface"""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a string representation of the FileWriter interface"""
        return self.__str__()

    @abstractmethod
    def open(self):
        """Connect to the DB (or open file if file-based DB)"""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Close DB"""
        raise NotImplementedError

    @abstractmethod
    def store(self, inputs, outputs) -> int:
        """
        Store the two arrays using a given backend
        """
        raise NotImplementedError


class CSVWriter(FileWriter):
    """
    A simple CSV backend.
    """

    suffix = "csv"

    def __init__(self, file_name: str, delimiter: str = ":"):
        super().__init__()
        self.file_name = file_name
        self.delimiter = delimiter
        self.fd = None
        self.write_header = False

    def __str__(self) -> str:
        return f"{__class__.__name__}(fd={self.fd}, delimiter={self.delimiter})"

    def open(self):
        if not Path(self.file_name).exists():
            self.write_header = True

        self.fd = open(self.file_name, "a")
        return self

    def close(self):
        self.write_header = False
        self.fd.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def store(self, inputs: np.array, outputs: np.array) -> int:
        """Store the two arrays in a CSV file"""
        assert len(inputs) == len(outputs)
        if self.fd and self.fd.closed:
            return 0
        if self.write_header:
            writer = csv.DictWriter(
                self.fd,
                fieldnames=[f"input_{i}" for i in range(inputs.shape[-1])]
                + [f"output_{i}" for i in range(outputs.shape[-1])],
                delimiter=self.delimiter,
            )
            writer.writeheader()
            self.write_header = False

        csvwriter = csv.writer(self.fd, delimiter=self.delimiter, quotechar="'", quoting=csv.QUOTE_MINIMAL)
        nelem = len(inputs)
        elem_wrote: int = 0
        # We follow the mini-app format, inputs elem and then output elems
        for i in range(nelem):
            elem_wrote += csvwriter.writerow(np.concatenate((inputs[i], outputs[i]), axis=0))
        return elem_wrote

    @classmethod
    def get_file_format_suffix(cls):
        return cls.suffix


class HDF5Writer(FileWriter):
    """
    A simple hdf5 backend.
    """

    suffix = "h5"

    def __init__(self, file_name: str):
        super().__init__()
        self.file_name = file_name
        self.fd = None
        self.datasets = dict()

    def __str__(self) -> str:
        return f"{__class__.__name__}(file_name={self.filename}, fd={self.fd})"

    def open(self):
        self.fd = h5py.File(self.file_name, "a")
        return self

    def close(self):
        self.fd.close()
        self.datasets.clear()
        self.fd = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _store_dataset(self, dset_name, data):
        if dset_name not in self.fd:
            max_shape = list(data.shape)
            max_shape[0] = None
            self.datasets[dset_name] = self.fd.create_dataset(dset_name, data=data, chunks=True, maxshape=max_shape)
            return
        self.fd[dset_name].resize(size=self.fd[dset_name].shape[0] + data.shape[0], axis=0)
        self.fd[dset_name][-data.shape[0] :] = data

    def store(self, inputs: np.array, outputs: np.array) -> int:
        """Store the two arrays in a hdf5 file"""
        assert len(inputs) == len(outputs)

        if self.fd is None:
            raise RuntimeError("HDF5 file {self.file_name} is not open")
        for i in range(inputs.shape[-1]):
            self._store_dataset(f"input_{i}", inputs[..., i])

        for i in range(outputs.shape[-1]):
            self._store_dataset(f"output_{i}", outputs[..., i])

    @classmethod
    def get_file_format_suffix(cls):
        return cls.suffix


class HDF5PackedWriter(HDF5Writer):
    """
    A simple hdf5 backend.
    """

    suffix = "h5"

    def __init__(self, file_name: str):
        super().__init__(file_name)

    def __str__(self) -> str:
        return f"{__class__.__name__}(file_name={self.filename}, fd={self.fd})"

    def store(self, inputs: np.array, outputs: np.array) -> int:
        """Store the two arrays in a hdf5 file"""
        assert len(inputs) == len(outputs)

        if self.fd is None:
            raise RuntimeError("HDF5 file {self.file_name} is not open")

        super()._store_dataset("inputs", inputs)
        super()._store_dataset("outputs", outputs)

    @classmethod
    def get_file_format_suffix(cls):
        return cls.suffix

def get_reader(ftype="hdf5"):
    """
    Factory method return a AMS file reader depending on the requested filetype
    """

    readers = {"hdf5": HDF5CLibReader, "csv": CSVReader}
    return readers[ftype]


def get_writer(ftype="hdf5"):
    """
    Factory method return a AMS file writer depending on the requested filetype
    """

    writers = {"hdf5": HDF5Writer, "csv": CSVWriter}
    return writers[ftype]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="version to assign to data file", choices=["hdf5", "csv"], required=True)
    parser.add_argument("--action", "-a", help="action", choices=["write", "read"], required=True)
    parser.add_argument("filename")
    args = parser.parse_args()

    if args.action == "write":
        if args.type == "csv":
            db = CSVWriter
        elif args.type == "hdf5":
            db = HDF5PackedWriter

        with db(args.filename) as fd:
            inputs = np.zeros((10, 4))
            outputs = np.zeros((10, 2))
            fd.store(inputs, outputs)

    elif args.action == "read":
        if args.type == "csv":
            db = CSVReader
        elif args.type == "hdf5":
            db = HDF5CLibReader

        with db(args.filename) as fd:
            input_data, output_data = fd.load()


if __name__ == "__main__":
    main()
