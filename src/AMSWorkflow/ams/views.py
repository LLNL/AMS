# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import abc
import os
import tempfile
from pathlib import Path

import h5py

from ams.store import AMSDataStore


class AMSHDF5VirtualDBReader:
    class DataDescr:
        def __init__(self, input_shape, i_type, output_shape, o_type):
            self._i_shape = input_shape
            self._o_shape = output_shape
            self._i_type = i_type
            self._o_type = o_type

        @property
        def i_shape(self):
            return self._i_shape

        @property
        def o_shape(self):
            return self._o_shape

        @property
        def i_type(self):
            return self._i_type

        @property
        def o_type(self):
            return self._o_type

    @staticmethod
    def verify_dsets(dsets_descr):
        i_types = set()
        o_types = set()
        i_shape = set()
        o_shape = set()
        for k, v in dsets_descr.items():
            i_types.add(v.i_type)
            o_types.add(v.o_type)
            i_shape.add(v.i_shape[-1])
            o_shape.add(v.o_shape[-1])
            if len(i_types) != 1:
                raise RuntimeError(f"File {k} has un-expected data-type")

            if len(o_types) != 1:
                raise RuntimeError(f"File {k} has un-expected data-type")

            if len(i_shape) != 1:
                raise RuntimeError(f"File {k} has un-expected input shape")

            if len(o_shape) != 1:
                raise RuntimeError(f"File {k} has un-expected output shape")

        return

    @staticmethod
    def create_vds_layout(dsets_descr):
        fn = next(iter(dsets_descr))
        i_shape = list(dsets_descr[fn].i_shape)
        o_shape = list(dsets_descr[fn].o_shape)
        i_type = dsets_descr[fn].i_type
        o_type = dsets_descr[fn].o_type

        i_shape[0] = sum(v.i_shape[0] for k, v in dsets_descr.items())
        o_shape[0] = sum(v.o_shape[0] for k, v in dsets_descr.items())

        assert i_shape[0] == o_shape[0], "Outer dimension of input/output shape does not match"

        print(fn, i_type, i_shape)
        print(fn, o_type, o_shape)

        i_layout = h5py.VirtualLayout(shape=tuple(i_shape), dtype=i_type)
        o_layout = h5py.VirtualLayout(shape=tuple(o_shape), dtype=o_type)

        outer_index = 0
        for k, v in dsets_descr.items():
            i_source = h5py.VirtualSource(k, "inputs", shape=v.i_shape)
            i_layout[outer_index : outer_index + v.i_shape[0], ...] = i_source
            o_source = h5py.VirtualSource(k, "outputs", shape=v.o_shape)
            o_layout[outer_index : outer_index + v.o_shape[0], ...] = o_source
            outer_index += v.i_shape[0]

        return i_layout, o_layout

    def __init__(self, files, i_names=list(), o_names=list()):
        dsets_descr = dict()

        if not files:
            return

        for f in files:
            print(f"Processing file: {f}")
            # Every file has both input, output data
            # Open file and pick the data types and the shapes.
            # We need those to map them correctly to a virtual file.
            with h5py.File(f, "r") as fd:
                i_shape = fd["inputs"].shape
                i_type = fd["inputs"].dtype
                o_shape = fd["outputs"].shape
                o_type = fd["outputs"].dtype

                if not i_names:
                    i_names = [f"input_{i}" for i in range(i_shape[-1])]
                else:
                    if len(i_names) != i_shape[-1]:
                        raise RuntimeError(f"Input name description {i_names} differs in size with {i_shape[-1]}")

                if not o_names:
                    o_names = [f"output_{i}" for i in range(o_shape[-1])]
                else:
                    if len(o_names) != o_shape[-1]:
                        raise RuntimeError(f"Ouput name description {o_names} differs in size with {o_shape[-1]}")

                dsets_descr[f] = AMSHDF5VirtualDBReader.DataDescr(i_shape, i_type, o_shape, o_type)

        self.verify_dsets(dsets_descr)
        i_vds, o_vds = self.create_vds_layout(dsets_descr)
        self._fn = Path(tempfile.mkdtemp()) / Path("VDS.h5")
        print(f"VDS name is {self._fn}")

        with h5py.File(self._fn, "w") as fd:
            fd.create_virtual_dataset("inputs", i_vds)
            fd.create_virtual_dataset("outputs", o_vds)

    @property
    def fn(self):
        return self._fn

    def destroy(self):
        parent = self._fn.parents[0]
        os.remove(str(self._fn))
        os.rmdir(parent)
        self._fn = None

    def __del__(self):
        if self._fn is not None:
            print("Deleting object but virtual data set file is not delete from file system")


class AMSDataView(abc.ABC):
    def __init__(self, ams_store, domain_name, entry="data", versions=None, **options):
        assert len(self.input_feature_names) == len(
            self.input_feature_dims
        ), "input feature names does not match dimensions"
        assert len(self.input_feature_names) == len(
            self.input_feature_types
        ), "input feature types does not match dimensions of inputs"
        assert len(self.output_feature_names) == len(
            self.output_feature_dims
        ), "output feature names does not match output feature dimensions"
        assert len(self.output_feature_names) == len(
            self.output_feature_types
        ), "output feature names does not match type dimensions"

        assert entry in AMSDataStore.valid_entries, "entry is not a valid store entry"

        assert entry != "model", "AMSDataviewcannot 'read' models"

        self._store = ams_store
        self._entry = entry
        self._versions = versions
        self._domain_name = domain_name
        self._data_files = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return self.close()

    def close(self):
        self._fd.close()
        self._fd = None
        self._hvds.destroy()
        self._hvds = None

    def open(self):
        if self._store.is_open():
            self._data_files = self._store.get_files(self._domain_name, self._entry, self._versions)
        else:
            store = self._store.open()
            self._data_files = self._store.get_files(self._domain_name, self._entry, self._versions)
            store.close()
        if not self._data_files:
            raise ValueError(
                f"Opening AMS Store in entry '{self._entry}' does not have files for the requested versions"
            )

        self._hvds = AMSHDF5VirtualDBReader(self._data_files, self.input_feature_names, self.output_feature_names)
        self._fd = h5py.File(self._hvds.fn, "r")
        return self

    @abc.abstractproperty
    def input_feature_names(self):
        """a list of the names of the input features"""

    @abc.abstractproperty
    def input_feature_dims(self):
        """a list of the dimensions of input features"""

    @abc.abstractproperty
    def input_feature_types(self):
        """a list of the types of input features"""

    def describe_inputs(self):
        return {
            "feature names": self.input_feature_names,
            "feature dims": self.input_feature_dims,
            "feature types": self.input_feature_types,
        }

    @abc.abstractproperty
    def output_feature_names(self):
        """a list of the names of the output features"""

    @abc.abstractproperty
    def output_feature_dims(self):
        """a list of the dimensions of output features"""

    @abc.abstractproperty
    def output_feature_types(self):
        """a list of the types of output features"""

    def describe_outputs(self):
        return {
            "feature names": self.output_feature_names,
            "feature dims": self.output_feature_dims,
            "feature types": self.output_feature_types,
        }

    def get_files(self):
        return self._data_files

    # methods for collecting data. Should be overloaded for more complex workflows
    def get_input_data(self):
        """Return the input data for this dataset"""

        if self._fd is None:
            raise RuntimeError("Trying to access closed AMS dataset")
        print("Input keys are", self._fd.keys())
        print(len(self._fd["inputs"]))
        return self._fd["inputs"]

    def get_output_data(self):
        """return the output data for this dataset"""

        if self._fd is None:
            raise RuntimeError("Trying to access closed AMS dataset")

        return self._fd["outputs"]

    def get_data(self):
        return self.get_input_data(), self.get_output_data()

    def get(self, k):
        return self.kosh_dataset.get(k)

    @property
    def versions(self):
        return self.versions
        pass

    @versions.getter
    def versions(self):
        return self._versions
