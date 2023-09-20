import csv
import pathlib
import unittest

import h5py
import numpy as np

from ams import database


def test_open(cls, fn):
    fd = cls(fn)
    fd = fd.open()
    return fd


class TestWriter(unittest.TestCase):
    def _open_close(self, cls, fn):
        fd = test_open(cls, fn)
        fd.close()
        fn = pathlib.Path(fn)
        self.assertTrue(fn.is_file(), msg="Open/Close did not create a file")
        fn.unlink()

    def _store(self, cls, fn):
        fd = test_open(cls, fn)
        inputs = np.random.rand(3, 2)
        outputs = np.random.rand(4, 3)
        self.assertRaises(AssertionError, fd.store, inputs, outputs)
        inputs = np.random.rand(3, 2)
        outputs = np.random.rand(3, 3)
        fd.store(inputs, outputs)
        fd.close()
        return inputs, outputs


class TestCSVWriter(TestWriter):
    def test_csv_open_close(
        self,
    ):
        super()._open_close(database.CSVWriter, "ams_test." + database.CSVReader.get_file_format_suffix())

    def test_csv_store(self):
        fn = "ams_test." + database.CSVReader.get_file_format_suffix()
        inputs, outputs = super()._store(database.CSVWriter, fn)
        with open(fn, "r") as x:
            sample_data = list(csv.reader(x, delimiter=":"))
        data = np.delete(np.array(sample_data), (0), axis=0).astype(inputs.dtype)
        _cdata = np.hstack((inputs, outputs))
        self.assertTrue(np.array_equal(data, _cdata), msg="Writting data loses information")

    def tearDown(self):
        fn = pathlib.Path("ams_test." + database.CSVReader.get_file_format_suffix())
        if fn.exists():
            fn.unlink()


class TestHDF5Writer(TestWriter):
    def _pack_dsets_to_list(self, dsets, selector):
        data = [dsets[k[0]] for k in selector]
        data = [data[k[1]] for k in selector]
        return data

    def _map_name_to_index(self, dsets_keys, name):
        # select keys of interest
        keys = [i for i in dsets_keys if name in i]
        keys = [(k, int(k.split("_")[-1])) for k in keys]
        return keys

    def test_csv_open_close(
        self,
    ):
        super()._open_close(database.HDF5Writer, "ams_test." + database.HDF5Writer.get_file_format_suffix())

    def test_csv_store(self):
        fn = "ams_test." + database.HDF5Writer.get_file_format_suffix()
        inputs, outputs = super()._store(database.HDF5Writer, fn)

        with h5py.File(fn, "r") as fd:
            dsets = fd.keys()
            input_map = self._map_name_to_index(dsets, "input")
            output_map = self._map_name_to_index(dsets, "output")
            f_inputs = np.array(self._pack_dsets_to_list(fd, input_map)).T
            f_outputs = np.array(self._pack_dsets_to_list(fd, output_map)).T

        self.assertTrue(np.array_equal(f_inputs, inputs), msg="Writting data loses information")
        self.assertTrue(np.array_equal(f_outputs, outputs), msg="Writting data loses information")

    def tearDown(self):
        fn = pathlib.Path("ams_test." + database.HDF5Writer.get_file_format_suffix())
        if fn.exists():
            fn.unlink()


class TestH5PackedWriter(TestWriter):
    def test_csv_open_close(
        self,
    ):
        super()._open_close(database.HDF5PackedWriter, "ams_test." + database.HDF5PackedWriter.get_file_format_suffix())

    def test_csv_store(self):
        fn = "ams_test." + database.HDF5PackedWriter.get_file_format_suffix()
        inputs, outputs = super()._store(database.HDF5PackedWriter, fn)

        with h5py.File(fn, "r") as fd:
            f_inputs = np.array(fd["inputs"])
            f_outputs = np.array(fd["outputs"])

        self.assertTrue(np.array_equal(f_inputs, inputs), msg="Writting data loses information")
        self.assertTrue(np.array_equal(f_outputs, outputs), msg="Writting data loses information")

    def tearDown(self):
        fn = pathlib.Path("ams_test." + database.HDF5PackedWriter.get_file_format_suffix())
        if fn.exists():
            fn.unlink()


class TestReader(unittest.TestCase):
    def _open_close(self, cls, fn):
        fd = test_open(cls, fn)
        fd.close()
        fn = pathlib.Path(fn)
        self.assertTrue(fn.is_file(), msg="Open/Close did not create a file")

    def _write(self, writer_cls, fn):
        fd = test_open(writer_cls, fn)
        inputs = np.random.rand(3, 2)
        outputs = np.random.rand(3, 3)
        fd.store(inputs, outputs)
        fd.close()
        return inputs, outputs

    def _read(self, reader_cls, fn):
        fd = test_open(reader_cls, fn)
        data = fd.load()
        fd.close()
        return data

    def _cmp(self, writer_cls, reader_cls, fn):
        inputs, outputs = self._write(writer_cls, fn)
        read_inputs, read_outputs = self._read(reader_cls, fn)
        self.assertTrue(
            np.array_equal(read_inputs, inputs), msg=f"Writting with {writer_cls} and reading with {reader_cls}"
        )
        self.assertTrue(
            np.array_equal(read_outputs, outputs), msg="Writting with {writer_cls} and reading with {reader_cls}"
        )


class TestCSVReader(TestReader):
    def test_load(self):
        fn = "ams_test." + database.CSVReader.get_file_format_suffix()
        super()._cmp(database.CSVWriter, database.CSVReader, fn)

    def tearDown(self):
        fn = "ams_test." + database.CSVReader.get_file_format_suffix()
        fn = pathlib.Path(fn)
        if fn.exists():
            fn.unlink()


class TestHDF5Reader(TestReader):
    def test_load(self):
        fn = "ams_test." + database.HDF5CLibReader.get_file_format_suffix()
        super()._cmp(database.HDF5Writer, database.HDF5CLibReader, fn)

    def tearDown(self):
        fn = "ams_test." + database.HDF5CLibReader.get_file_format_suffix()
        fn = pathlib.Path(fn)
        if fn.exists():
            fn.unlink()


class TestHDF5PackedReader(TestReader):
    def test_load(self):
        fn = "ams_test." + database.HDF5PackedReader.get_file_format_suffix()
        super()._cmp(database.HDF5PackedWriter, database.HDF5PackedReader, fn)

    def tearDown(self):
        fn = "ams_test." + database.HDF5PackedReader.get_file_format_suffix()
        fn = pathlib.Path(fn)
        if fn.exists():
            fn.unlink()


if __name__ == "__main__":
    unittest.main()
