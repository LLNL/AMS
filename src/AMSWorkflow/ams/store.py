# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import datetime
import os
import shutil
import json
from pathlib import Path

import kosh

from ams.util import get_unique_fn
from ams.util import mkdir
from ams.config import AMSInstance
from ams.store_types import AMSModelDescr


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
    entry_suffix = {"data": "h5", "models": "pt", "candidates": "h5"}
    entry_mime_types = {"data": "hdf5", "models": "zip", "candidates": "hdf5"}

    def __init__(self, store_path, store_name, name, delete_all_contents=False):
        """
        Initializes the AMSDataStore class. Upon init the kosh-store is closed and not connected
        """
        print("Creating store under path", store_path)
        create_store_directories(store_path)
        self._root_path = Path(store_path)
        self._delete_contents = delete_all_contents
        self._name = name
        self._store_path = Path(store_path).absolute() / Path(store_name)
        self._AMS_schema = kosh.KoshSchema(required=AMSDataStore.data_schema)
        self._store = None
        self._entry_paths = {k: Path(store_path) / Path(k) for k in self.__class__.valid_entries}

        # FIXME: I don't like the fact that we have 2 representations of the information of the store
        if not (Path(store_path) / Path("ams_config.json")).exists():
            with open(str(Path(store_path) / Path("ams_config.json")), "w") as fd:
                config = AMSInstance.create_config(store_path, store_name, name)
                json.dump(config, fd, indent=6)

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

    @property
    def store_path(self):
        return self._store_path

    @property
    def root_path(self):
        return self._root_path

    def get_candidate_path(self):
        return self._entry_paths["candidates"]

    def get_data_path(self):
        return self._entry_paths["data"]

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

    def _get_or_create_ensebmle(self, domain_name):
        """
        Getter of the kosh-enseble this instance is operating upon.

        Returns:
            A kosh-ensebmle.
        """
        ensemble = next(self._store.find_ensembles(name=domain_name), None)
        if ensemble is None:
            ensemble = self._store.create_ensemble(name=domain_name)

        return ensemble

    def _add_entry(self, domain_name, entry, mime_type, data_files=list(), version=None, metadata=dict()):
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

        if entry not in AMSDataStore.valid_entries:
            raise RuntimeError("Trying to add entry that does not exist")

        data_files = [str(Path(d).absolute()) for d in data_files]

        ensemble = self._get_or_create_ensebmle(domain_name)
        metadata["date"] = str(datetime.datetime.now())
        ds = self._get_or_create_dataset(ensemble, entry, version)

        for f in data_files:
            ds.associate(f, mime_type=mime_type, metadata=metadata, absolute_path=True)

        return

    def add_data(self, domain_name, data_files=list(), version=None, metadata=dict()):
        """
        Adds files in the kosh-store and associates them to the 'data' entry.

        The function assumes data_files to always be in hdf5 format.

        Args:
            data_files: A list of files to add in the entry
            version: The version to assign to all files
            metadata: The metadata to associate with this file
        """
        self._add_entry(domain_name, "data", "hdf5", data_files, version, metadata)

    def add_model(self, domain_name, model, test_error, val_error, version=None, metadata=dict()):
        """
        Adds a model in the kosh-store and associates them to the 'models' entry.

        The function assumes models to always be in torchscript format.

        Args:
            model: The path containing the torchscript model
            version: The version to assing to the model
            metadata: The metadata to associate with this model
        """
        if not isinstance(model, AMSModelDescr):
            raise TypeError(f"AMSStore expects AMSModelDescr as a model-entry, got {type(model)}")

        info = model.to_dict()
        del info["path"]
        info["val_error"] = val_error
        info["test_error"] = test_error
        for k, v in metadata.items():
            if k in info.keys():
                raise RuntimeError(f"Key {k} exists in both info and metadata")
            else:
                info[k] = str(v)
        print("Waiting in model path", model.path)
        print(json.dumps(info, indent=6))
        self._add_entry(domain_name, "models", "zip", [model.path], version, info)

    def add_candidates(self, domain_name, data_files=list(), version=None, metadata=dict()):
        """
        Adds files in the kosh-store and associates them to the 'candidates' entry.

        The function assumes candidates to always be in hdf5 format.

        Args:
            data_files: A list of candidate files
            version: The version to assign to the model
            metadata: The metadata to associate with this model
        """
        self._add_entry(domain_name, "candidates", "hdf5", data_files, version, metadata)

    def _remove_entry_file(self, domain_name, entry, data_files=list(), delete_files=False):
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
        if domain_name is None:
            ensembles = self._store.find_ensembles()
        else:
            ensembles = self._store.find_ensembles(name=domain_name)

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

    def remove_data(self, domain_name, data_files=list(), delete_files=False):
        """
        Remove data-files from kosh-store. When delete_files is true, delete the actual file as well

        Args:
            data_files: A list of files to be deleted
            delete_files: delete the file from persistent storage
        """

        self._remove_entry_file(domain_name, "data", data_files, delete_files)

    def remove_models(self, domain_name, model=list(), delete_files=False):
        """
        Remove models from kosh-store. When delete_files is true, delete the actual file as well

        Args:
            data_files: A list of models to be deleted
            delete_files: delete the file from persistent storage
        """

        if not model:
            return

        self._remove_entry_file(domain_name, "models", model, delete_files)

    def remove_candidates(self, domain_name, data_files=list(), delete_files=False):
        """
        Remove candidates from kosh-store. When delete_files is true, delete the actual file as well

        Args:
            data_files: A list of candidates to be deleted
            delete_files: delete the file from persistent storage
        """

        self._remove_entry_file(domain_name, "candidates", data_files, delete_files)

    def _get_entry_versions(self, domain_name, entry, associate_files=False):
        """
        Returns a list of versions existing for the specified entry

        Args:
            entry: The entry type we are looking for
            associate_files: Associate files in store with the versions

        Returns:
            A list of existing versions in our database or a dictionary of versions to lists associating files with the specific version
        """
        ensembles = self._store.find_ensembles(name=domain_name)
        if associate_files:
            versions = dict()
        else:
            versions = list()
        for e in ensembles:
            for dset in e.find_datasets(name=entry):
                if associate_files:
                    if dset.version not in versions:
                        versions[dset.version] = list()
                    for associated in dset.find():
                        versions[dset.version].append(associated.uri)
                else:
                    versions.append(dset.version)
        return versions

    def get_data_versions(self, domain_name, associate_files=False):
        """
        Returns a list of versions existing for the data entry


        Returns:
            A list of existing versions in our database
        """
        return self._get_entry_versions(domain_name, "data", associate_files)

    def get_model_versions(self, domain_name, associate_files=False):
        """
        Returns a list of versions existing for the model entry


        Returns:
            A list of existing model versions in our database
        """

        return self._get_entry_versions(domain_name, "models", associate_files)

    def get_candidate_versions(self, domain_name, associate_files=False):
        """
        Returns a list of versions existing for the candidate entry


        Returns:
            A list of existing candidate versions in our database
        """

        return self._get_entry_versions(domain_name, "candidates", associate_files)

    def get_files(self, domain_name, entry, versions=None):
        """
        Returns a list of paths to files for the specified version

        Args:
            entry: The entry in the ensemble can be any of candidates, model, data
            versions: A list of versions we are looking for.
                If 'None'   return all files in entry
                If "latest" return the latest version in the store

        Returns:
            A list of existing files in the kosh-store
        """
        files = self._get_entry_versions(domain_name, entry, True)

        if len(files) == 0:
            return list()

        if isinstance(versions, str) and versions == "latest":
            max_version = max(files.keys())
            return files[max_version]

        file_paths = list()
        for k, v in files.items():
            if versions is None or v in versions:
                file_paths = file_paths + v
        return file_paths

    def get_candidate_files(self, domain_name, versions=None):
        """
        Returns a list of paths to files for the specified version

        Args:
            versions: A list of versions we are looking for.
                If 'None'   return all files in entry
                If "latest" return the latest version in the store
                If "list" return only files matching these versions

        Returns:
            A list of existing files in the kosh-store candidates ensemble
        """
        return self.get_files(domain_name, "candidates", versions)

    def get_model_files(self, domain_name, versions=None):
        """
        Returns a list of paths to files for the specified version

        Args:
            versions: A list of versions we are looking for.
                If 'None'   return all files in entry
                If "latest" return the latest version in the store
                If "list" return only files matching these versions

        Returns:
            A list of existing files in the kosh-store model ensemble
        """

        return self.get_files(domain_name, "models", versions)

    def get_data_files(self, domain_name, versions=None):
        """
        Returns a list of paths to files for the specified version

        Args:
            versions: A list of versions we are looking for.
                If 'None'   return all files in entry
                If "latest" return the latest version in the store
                If "list" return only files matching these versions

        Returns:
            A list of existing files in the kosh-store model ensemble
        """

        return self.get_files(domain_name, "data", versions)

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

    def get_raw_content(self, domain_name, entry):
        """
        Returns a dictionary with all data in our AMS store
        """

        if domain_name is None:
            ensembles = self._store.find_ensembles()
        else:
            ensembles = self._store.find_ensembles(name=domain_name)

        if entry is None:
            entries = AMSDataStore.valid_entries
        else:
            entries = {entry}

        data = {}
        for e in ensembles:
            data[e.name] = {}
            for entry_type in entries:
                data[e.name][entry_type] = {}
                for d in e.find_datasets(name=entry_type):
                    dset = list()
                    for associated in d.find():
                        dset.append(associated.listattributes(True))
                    data[e.name][entry_type][d.version] = dset
        return data

    def move(self, domain_name, src_entry, dest_entry, files):
        """
        Moves files between entries in kosh-store. It follows a "safe" approach: copy, add, delete the file instead of moving the underlying file.

        Args:
            src_entry: the ensemble name containing the original files
            dest_entry: the ensemble name of the files
            files: The files to be moved

        NOTE: The current implementation will lose all metadata associated with the original src files. We need to consider whether we want to "migrate"
        those to the destination entry dataset.
        """
        if src_entry not in self.__class__.valid_entries:
            raise RuntimeError(f"Entry: {src_entry} not a valid AMSDataStore entry")

        if dest_entry not in self.__class__.valid_entries:
            raise RuntimeError(f"Entry: {dest_entry} not a valid AMSDataStore entry")

        entry_files = self.get_files(domain_name, src_entry)
        for e in files:
            if e not in entry_files:
                raise RuntimeError(
                    f"Moving file {e} from {src_entry} to {dest_entry} not possible as file does not exist in kosh-entry"
                )

        dest_dir = self._entry_paths[dest_entry]
        new_files = list()
        for f in files:
            dest_name = dest_dir / Path(f).name
            shutil.copy(f, dest_name)
            new_files.append(str(dest_name))

        print(new_files)
        print(files)
        self._add_entry(domain_name, dest_entry, "hdf5", new_files)
        self._remove_entry_file(domain_name, src_entry, files, True)

    def search(self, domain_name=None, entry=None, version=None, metadata=dict()):
        """
        Search for items in the database that match the metadata
        Args:
            entry: Which entry to search for ('data', 'models', 'candidates')
            version: Specific version to look for, when 'version' is 'latest' we
                return the entry with the largest version. If None, we are not matching
                versions.
            metadata: A dictionary of key values to search in our database

        Returns:
            A list of matching entries described as dictionaries
        """
        all_contents = self.get_raw_content(domain_name, entry)

        found = []

        for d_name, contents in all_contents.items():
            for e_name, entries in contents.items():
                for ver, dsets in entries.items():
                    if version is not None:
                        if (version != "latest") and (version != ver):
                            continue

                    for dset in dsets:
                        insert = True
                        for k, v in metadata.items():
                            if k in dset.keys():
                                if dset[k] != v:
                                    insert = False
                                    break
                            else:
                                insert = False
                                break
                        if insert:
                            value = {"domain": d_name, "entry": e_name, "version": ver, "file": dset["uri"]}
                            value.update(dset)
                            del value["fast_sha"]
                            del value["mime_type"]
                            del value["associated"]
                            del value["id"]
                            del value["uri"]
                            found.append(value)

        if len(found) != 0 and version == "latest":
            found = [max(found, key=lambda item: item["version"])]

        return found

    def __str__(self):
        return "AMS Kosh Wrapper Store(path={0}, name={1}, status={2})".format(
            self._store_path, self._name, "Open" if self.is_open() else "Closed"
        )

    def _suggest_entry_file_name(self, entry, domain_name):
        if domain_name is None:
            return str(self._entry_paths[entry] / Path(f"{get_unique_fn()}.{self.__class__.entry_suffix[entry]}"))
        return str(
            self._entry_paths[entry] / Path(f"{domain_name}_{get_unique_fn()}.{self.__class__.entry_suffix[entry]}")
        )

    def suggest_model_file_name(self, domain_name=None):
        return self._suggest_entry_file_name("models", domain_name)

    def suggest_candidate_file_name(self, domain_name=None):
        return self._suggest_entry_file_name("candidates", domain_name)

    def suggest_data_file_name(self, domain_name=None):
        return self._suggest_entry_file_name("data", domain_name)


def create_store_directories(store_path):
    """
    Creates the directory structure AMS prefers under the store_path.
    """
    store_path = Path(store_path)
    if not store_path.exists():
        store_path.mkdir(parents=True, exist_ok=True)

    for fn in list(AMSDataStore.valid_entries):
        mkdir(store_path, fn)
