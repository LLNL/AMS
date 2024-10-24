import os
import unittest
from pathlib import Path

from ams import store
from ams.store_types import AMSModelDescr
from ams.store_types import UQType
from ams.store_types import UQAggregate


def create_file(directory, fn):
    fileSizeInBytes = 16
    fn = Path(directory) / Path(fn)
    with open(str(fn), "w") as fd:
        fd.write(str(os.urandom(fileSizeInBytes)))
    return str(fn)


class TestStore(unittest.TestCase):
    h5_files = list()
    model_files = list()
    candidate_files = list()
    num_files = 10
    store_dir = None

    @classmethod
    def setUpClass(cls):
        cls.store_dir = Path("store_testing_dir")
        cls.store_dir.mkdir(parents=True, exist_ok=True)
        for i in range(cls.num_files):
            cls.h5_files.append(create_file(str(cls.store_dir), f"input_{i}.h5"))
            cls.model_files.append(create_file(str(cls.store_dir), f"model_{i}.pt"))
            cls.candidate_files.append(create_file(str(cls.store_dir), f"candidate_file_{i}.h5"))

    def _add_entries(self, add_func, getter, elements, *args, as_list=True):
        for i, f in enumerate(elements):
            if as_list:
                add_func("test", [f], *args, version=i)
            else:
                add_func("test", f, *args, version=i)

        versions = getter("test")
        self.assertTrue(len(versions) == len(elements), f"Adding elements using {add_func} not working properly")

    def _remove_entries(self, func, getter, elements):
        func("test", elements, False)
        return getter("test")

    def test_store_open(self):
        ams_store = store.AMSDataStore(self.__class__.store_dir, "test.sql", "ams_test")
        self.assertFalse(ams_store.is_open(), "AMS Store should be close, but isn't")
        ams_store = ams_store.open()
        self.assertTrue(ams_store.is_open(), "AMS Store should be opened, but isn't")
        ams_store.close()
        self.assertFalse(ams_store.is_open(), "AMS Store should be close, but isn't")

    def test_store_add_remove_query_data(self):
        ams_store = store.AMSDataStore(self.__class__.store_dir, "test.sql", "ams_test")
        ams_store = ams_store.open()
        self._add_entries(ams_store.add_data, ams_store.get_data_versions, self.__class__.h5_files)
        versions = self._remove_entries(ams_store.remove_data, ams_store.get_data_versions, self.__class__.h5_files)
        self.assertTrue(len(versions) == 0, f"Store should be empty but isn't {versions}")

        ams_store.close()

    def test_store_add_remove_query_candidates(self):
        ams_store = store.AMSDataStore(self.__class__.store_dir, "test.sql", "ams_test")
        ams_store = ams_store.open()
        self._add_entries(ams_store.add_candidates, ams_store.get_candidate_versions, self.__class__.candidate_files)
        versions = self._remove_entries(
            ams_store.remove_candidates, ams_store.get_candidate_versions, self.__class__.candidate_files
        )
        self.assertTrue(len(versions) == 0, f"Store should be empty but isn't {versions}")
        ams_store.close()

    def test_store_add_remove_query_model(self):
        ams_store = store.AMSDataStore(self.__class__.store_dir, "test.sql", "ams_test")
        ams_store = ams_store.open()

        model_descrs = []
        for m in self.__class__.model_files:
            model_descrs.append(AMSModelDescr(path=m, threshold=0.5, uq_type=UQType.Random))

        self._add_entries(
            ams_store.add_model,
            ams_store.get_model_versions,
            model_descrs,
            0.1,
            0.1,
            as_list=False,
        )
        versions = self._remove_entries(
            ams_store.remove_models, ams_store.get_candidate_versions, self.__class__.model_files
        )
        self.assertTrue(len(versions) == 0, f"Store should be empty but isn't {versions}")
        ams_store.close()

    def test_store_add_data_as_list(self):
        ams_store = store.AMSDataStore(self.__class__.store_dir, "test.sql", "ams_test")
        ams_store = ams_store.open()
        self._add_entries(ams_store.add_data, ams_store.get_data_versions, [self.__class__.h5_files], as_list=False)
        versions = self._remove_entries(ams_store.remove_data, ams_store.get_data_versions, self.__class__.h5_files)
        self.assertTrue(len(versions) == 0, f"Store should be empty but isn't {versions}")
        ams_store.close()

    def test_store_add_candidate_as_list(self):
        ams_store = store.AMSDataStore(self.__class__.store_dir, "test.sql", "ams_test")
        ams_store = ams_store.open()
        self._add_entries(
            ams_store.add_candidates,
            ams_store.get_candidate_versions,
            [self.__class__.candidate_files],
            as_list=False,
        )
        versions = self._remove_entries(
            ams_store.remove_candidates, ams_store.get_candidate_versions, self.__class__.candidate_files
        )
        self.assertTrue(len(versions) == 0, f"Store should be empty but isn't {versions}")
        ams_store.close()

    @classmethod
    def tearDownClass(cls):
        for _list in [cls.h5_files, cls.model_files, cls.candidate_files]:
            for f in _list:
                Path(f).unlink()
        for child in cls.store_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                child.rmdir()

        cls.store_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
