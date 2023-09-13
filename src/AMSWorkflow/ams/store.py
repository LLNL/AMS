# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import datetime
from .config import AMSConfig
import kosh
from pathlib import Path


class AMSDataStore:
    data_schema = {"problem": str, "version": int}
    valid_entries = {"data", "models", "candidates"}

    def __init__(self, store_path, store_name, name, delete_all_contents=False):
        self._delete_contents = delete_all_contents
        self._name = name
        self._store_path = Path(store_path) / Path(store_name)
        self._AMS_schema = kosh.KoshSchema(required=AMSDataStore.data_schema)
        self._store = None

    def is_open(self):
        return self._store is not None

    def open(self):
        if self.is_open():
            return self

        self._store = kosh.connect(
            str(self._store_path), delete_all_contents=self._delete_contents
        )

        return self

    def _get_or_create_dataset(self, ensemble, _name, version):
        dsets = ensemble.find_datasets(name=_name)
        versions = {d.version: d for d in dsets}

        if version is None:
            version = 0 if not versions else max(versions.keys()) + 1

        if version in versions:
            return versions[version]

        ds = ensemble.create(name=_name, metadata={"version": version})
        print(ds)
        return ds

    def _get_or_create_ensebmle(self):
        ensemble = next(self._store.find_ensembles(name=self._name), None)
        if ensemble is None:
            ensemble = self._store.create_ensemble(name=self._name)

        return ensemble

    def _add_entry(self, entry, mime_type, data_files=list(), version=None, metadata=dict()):
        if not self.is_open():
            raise RuntimeError("Trying to add data in a closed database")

        data_files = [str(Path(d).absolute()) for d in data_files]

        ensemble = self._get_or_create_ensebmle()
        metadata["date"] = str(datetime.datetime.now())
        ds = self._get_or_create_dataset(ensemble, entry, version)

        for f in data_files:
            ds.associate(f, mime_type=mime_type, metadata=metadata, absolute_path=True)

    def add_data(self, data_files=list(), version=None, metadata=dict()):
        self._add_entry("data", "hdf5", data_files, version, metadata)

    def add_model(self, model, version=None, metadata=dict()):
        self._add_entry("models", "zip", [model], version, metadata)

    def add_candidates(self, data_files=list(), version=None, metadata=dict()):
        self._add_entry("candidates", "hdf5", data_files, version, metadata)

    def _remove_entry_file(self, entry, data_files=list(), delete_files=False):
        if not data_files:
            return

        data_files = [str(Path(d).absolute()) for d in data_files]

        ensembles = self._store.find_ensembles(name=self._name)
        for e in ensembles:
            for dset in e.find_datasets(name=entry):
                dset_files = [f.uri for f in dset.find()]
                for rd in data_files:
                    if rd in dset_files:
                        dset.dissociate(rd)

            for dset in e.find_datasets(name=entry):
                if not list(dset.find()):
                    e.remove(dset)

        if delete_files:
            for d in data_files:
                os.remove(d)

    def remove_data(self, data_files=list(), delete_files=False):
        self._remove_entry_file("data", data_files, delete_files)

    def remove_models(self, model=list(), delete_files=False):
        if not model:
            return

        self._remove_entry_file("models", model, delete_files)

    def remove_candidates(self, data_files=list(), delete_files=False):
        self._remove_entry_file("candidates", data_files, delete_files)

    def close(self):
        self._store.close()
        self._store = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return self.close()

    def get_raw_content(self):
        ensembles = self._store.find_ensembles(name=self._name)
        data = {"data": {}, "models": {}, "candidates": {}}
        for e in ensembles:
            for entry_type in AMSDataStore.valid_entries:
                for d in e.find_datasets(name=entry_type):
                    dset = list()
                    for associated in d.find():
                        print(associated.listattributes(True))
                        dset.append(associated.listattributes(True))
                    data[entry_type][d.version] = dset
        return data

    def __str__(self):
        return "AMS Kosh Wrapper Store(path={0}, name={1}, status={2})".format(
            self._store_path, self._name, "Open" if self.is_open() else "Closed"
        )
