# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import datetime
import os
from pathlib import Path

import kosh


class AMSDataStore:
    """A thin wrapper around a kosh store

    The class abstract the 'view' of AMS persistent data storage through
    a kosh store. The AMSDataStore consists of three essential pieces of information.
        1. 'data' : A collection of files stored in some PFS directory that will train some model
        2. 'model' : A collection of torch-scripted models.
        3. 'candidates' : A collection of files stored in some PFS directory that can be added as data.
    The class will provide mechanism to add, remove, query about files in the store. It refers to the pieces
    as entries.

    Attributes:
        delete_contents: a boolean value instructing the kosh-store to delete all contents.
        name: The name of the data in the store
        store_path: The path to the kosh-database file
        store: The Kosh store
    """

    data_schema = {"problem": str, "version": int}
    valid_entries = {"data", "models", "candidates"}

    def __init__(self, store_path, store_name, name, delete_all_contents=False):
        """
        Initializes the AMSDataStore class. Upon init the kosh-store is closed and not connected
        """

        self._delete_contents = delete_all_contents
        self._name = name
        self._store_path = Path(store_path) / Path(store_name)
        self._AMS_schema = kosh.KoshSchema(required=AMSDataStore.data_schema)
        self._store = None

    def is_open(self):
        """
        Check whether the kosh store is open and accessible
        """

        return self._store is not None

    def open(self):
        """
        Open and connect to the kosh store
        """

        if self.is_open():
            return self

        self._store = kosh.connect(str(self._store_path), delete_all_contents=self._delete_contents)

        return self

    def _get_or_create_dataset(self, ensemble, entry, version):
        """
        Getter of a kosh-dataset with the requested version.

        Args:
            ensemble: The kosh-ensemble to create/get the dataset from.
            entry: The 'entry' of the dataset we are looking for
            version: The version of the dataset we are searching for.
            If None we create a new version larger than the current maximum one.

        Returns:
            A kosh-dataset for the requested version and entry-type.
        """
        dsets = ensemble.find_datasets(name=entry)
        versions = {d.version: d for d in dsets}

        if version is None:
            version = 0 if not versions else max(versions.keys()) + 1

        if version in versions:
            return versions[version]

        ds = ensemble.create(name=entry, metadata={"version": version})
        return ds

    def _get_or_create_ensebmle(self):
        """
        Getter of the kosh-enseble this instance is operating upon.

        Returns:
            A kosh-ensebmle.
        """
        ensemble = next(self._store.find_ensembles(name=self._name), None)
        if ensemble is None:
            ensemble = self._store.create_ensemble(name=self._name)

        return ensemble

    def _add_entry(self, entry, mime_type, data_files=list(), version=None, metadata=dict()):
        """
        Adds files of mime_type in the kosh-store and associates them appropriately.

        Args:
            entry: The entry type we will add can be either 'models', 'candidates', 'data'.
            mime_type: Indicator of the format of the document.
            data_files: A list of files to add in the entry
            version: The version to assign to all files
            metadata: The metadata to associate with this file
        """
        if not self.is_open():
            raise RuntimeError("Trying to add data in a closed database")

        data_files = [str(Path(d).absolute()) for d in data_files]

        ensemble = self._get_or_create_ensebmle()
        metadata["date"] = str(datetime.datetime.now())
        ds = self._get_or_create_dataset(ensemble, entry, version)

        for f in data_files:
            ds.associate(f, mime_type=mime_type, metadata=metadata, absolute_path=True)

        return

    def add_data(self, data_files=list(), version=None, metadata=dict()):
        """
        Adds files in the kosh-store and associates them to the 'data' entry.

        The function assumes data_files to always be in hdf5 format.

        Args:
            data_files: A list of files to add in the entry
            version: The version to assign to all files
            metadata: The metadata to associate with this file
        """
        self._add_entry("data", "hdf5", data_files, version, metadata)

    def add_model(self, model, version=None, metadata=dict()):
        """
        Adds a model in the kosh-store and associates them to the 'models' entry.

        The function assumes models to always be in torchscript format.

        Args:
            model: The path containing the torchscript model
            version: The version to assing to the model
            metadata: The metadata to associate with this model
        """
        self._add_entry("models", "zip", [model], version, metadata)

    def add_candidates(self, data_files=list(), version=None, metadata=dict()):
        """
        Adds files in the kosh-store and associates them to the 'candidates' entry.

        The function assumes candidates to always be in hdf5 format.

        Args:
            data_files: A list of candidate files
            version: The version to assing to the model
            metadata: The metadata to associate with this model
        """
        self._add_entry("candidates", "hdf5", data_files, version, metadata)

    def _remove_entry_file(self, entry, data_files=list(), delete_files=False):
        """
        Remove files from kosh-store. When delete_files is true, delete the actual file as well

        Args:
            entry: The entry to look for the specified files
            data_files: A list of files to be deleted
            delete_files: delete the file from persistent storage
        """

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
        """
        Remove data-files from kosh-store. When delete_files is true, delete the actual file as well

        Args:
            data_files: A list of files to be deleted
            delete_files: delete the file from persistent storage
        """

        self._remove_entry_file("data", data_files, delete_files)

    def remove_models(self, model=list(), delete_files=False):
        """
        Remove models from kosh-store. When delete_files is true, delete the actual file as well

        Args:
            data_files: A list of models to be deleted
            delete_files: delete the file from persistent storage
        """

        if not model:
            return

        self._remove_entry_file("models", model, delete_files)

    def remove_candidates(self, data_files=list(), delete_files=False):
        """
        Remove candidates from kosh-store. When delete_files is true, delete the actual file as well

        Args:
            data_files: A list of candidates to be deleted
            delete_files: delete the file from persistent storage
        """

        self._remove_entry_file("candidates", data_files, delete_files)

    def _get_entry_versions(self, entry):
        """
        Returns a list of versions existing for the specified entry

        Args:
            entry: The entry type we are looking for

        Returns:
            A list of existing versions in our database
        """
        ensembles = self._store.find_ensembles(name=self._name)
        versions = list()
        for e in ensembles:
            for dset in e.find_datasets(name=entry):
                versions.append(dset.version)
        return versions

    def get_data_versions(self):
        """
        Returns a list of versions existing for the data entry


        Returns:
            A list of existing versions in our database
        """
        return self._get_entry_versions("data")

    def get_model_versions(self):
        """
        Returns a list of versions existing for the model entry


        Returns:
            A list of existing model versions in our database
        """

        return self._get_entry_versions("models")

    def get_candidates_versions(self):
        """
        Returns a list of versions existing for the candidate entry


        Returns:
            A list of existing candidate versions in our database
        """

        return self._get_entry_versions("candidates")

    def close(self):
        """
        Closes the connection with Kosh-store
        """

        self._store.close()
        self._store = None

    def __enter__(self):
        """
        Context Manager for kosh store
        """

        return self.open()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Exit context manager
        """

        return self.close()

    def get_raw_content(self):
        """
        Returns a dictionary with all data in our AMS store
        """

        ensembles = self._store.find_ensembles(name=self._name)
        data = {"data": {}, "models": {}, "candidates": {}}
        for e in ensembles:
            for entry_type in AMSDataStore.valid_entries:
                for d in e.find_datasets(name=entry_type):
                    dset = list()
                    for associated in d.find():
                        dset.append(associated.listattributes(True))
                    data[entry_type][d.version] = dset
        return data

    def __str__(self):
        return "AMS Kosh Wrapper Store(path={0}, name={1}, status={2})".format(
            self._store_path, self._name, "Open" if self.is_open() else "Closed"
        )


def create_store_directories(store_path):
    """
    Creates the directory structure AMS prefers under the store_path.
    """
    store_path = Path(store_path)
    if not store_path.exists():
        store_path.mkdir(parents=True, exist_ok=True)

    for fn in list(AMSDataStore.valid_entries):
        _tmp = store_path / Path(fn)
        if not store_path.exists():
            _tmp.mkdir(parents=True, exist_ok=True)
