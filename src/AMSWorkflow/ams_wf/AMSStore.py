#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
from pathlib import Path
import shutil

from ams.config import AMSInstance
from ams.store import AMSDataStore
from ams.store import create_store_directories


class StoreCommand:
    __cli_root_options__ = [["path", "p", "path of the AMS-store directory", {"required": True}]]

    def __init__(self, **kwargs):
        self._store_path = kwargs.get("path")
        self.ams_config = AMSInstance.from_path(self._store_path)
        pass

    @classmethod
    def add_cli_options(cls, parser):
        for v in cls.__cli_options__:
            parser.add_argument(f"--{v[0]}", f"-{v[1]}", help=v[2], **v[3])

        for v in cls.__cli_root_options__:
            parser.add_argument(f"--{v[0]}", f"-{v[1]}", help=v[2], **v[3])

    @classmethod
    def from_cli(cls, args):
        cli_options = vars(args)
        kwargs = {}
        for v in cls.__cli_options__:
            kwargs[v[0]] = cli_options[v[0]]

        for v in cls.__cli_root_options__:
            kwargs[v[0]] = cli_options[v[0]]

        return cls(**kwargs)


class CreateStore(StoreCommand):
    __cli_options__ = [
        ["name", "n", "Name of the application to store data", {"required": True}],
        ["sname", "sn", "Name of the underlying kosh db-store", {"default": "ams_store.sql"}],
    ]

    def __init__(self, path, name, sname):
        db_path = Path(path)
        config_path = db_path / Path("ams_config.json")
        create_store_directories(db_path)

        if not config_path.exists():
            config = AMSInstance.create_config(path, sname, name)

            with open(str(config_path), "w") as fd:
                json.dump(config, fd, indent=4)

        super().__init__(path=str(db_path))

    def __call__(self, store):
        pass


class AddToStore(StoreCommand):
    __cli_options__ = [
        ["entry", "e", "Which entry to access", {"choices": AMSDataStore.valid_entries, "required": True}],
        ["metadata", "m", "metadata to search for", {"default": None}],
        ["file", "f", "File to add to store", {"required": True}],
        ["version", "v", "Assign specific version to the file", {}],
        ["copy", "cp", "Copy data to the underlying AMS directory", {"action": "store_true"}],
    ]

    def __init__(self, entry=None, file="", metadata=None, copy=True, version=None, **kwargs):
        """
        Initializes the 'AddToStore' class with the specific query

        Args:
            entry: Optional name of an ensemble to search for names
            metadata: JSON-like string with key-values to match the kosh-store
            copy: copy the file from the initial location to the underlying store location
            version: Request a search for a specific version
        """

        super().__init__(**kwargs)

        if entry not in AMSDataStore.valid_entries:
            raise RuntimeError(f"{entry} is not a valid entry for AMSStore")

        self._entry = entry

        assert Path(file).exists(), f"{file} is not an existing file."

        self._fn = Path(file)

        self._md = None
        if metadata != None:
            self._md = json.loads(metadata)

        self._copy = copy
        self._version = version

    def __call__(self, store):
        # If move is supported we first copy the file
        fn = self._fn
        if self._copy:
            fn = store._suggest_entry_file_name(self._entry)
            shutil.copy(self._fn, fn)

        metadata = self._md if self._md is not None else dict()

        store._add_entry(self._entry, store.__class__.entry_mime_types[self._entry], [fn], self._version, metadata)


class RemoveFromStore(StoreCommand):
    __cli_options__ = [
        ["entry", "e", "Which entry to access", {"choices": AMSDataStore.valid_entries, "required": True}],
        ["metadata", "m", "metadata to search for", {"default": None}],
        ["version", "v", "Assign specific version to the file", {}],
        ["purge", "rm", "Delete the files associated with this item", {"action": "store_true"}],
    ]

    def __init__(self, entry=None, metadata=None, version=None, purge=False):
        """
        Initializes the 'RemoveFromStore' class with the specific query

        Args:
            entry: Optional name of an ensemble to search for names
            metadata: JSON-like string with key-values to match the kosh-store
            version: Request a search for a specific version
            purge: Delete the actual underlying file from the filesystem
        """

        if entry not in AMSDataStore.valid_entries:
            raise RuntimeError(f"{entry} is not a valid entry for AMSStore")

        self._entry = entry

        self._md = dict()
        if metadata != None:
            self._md = json.loads(metadata)

        self._version = None
        if version is not None:
            self._version = "latest"
            if version != "latest":
                assert version.isdigit(), "Version must be an integer"
                self._version = int(version)
        self._purge = purge

    def __call__(self, store):
        found = store.search(self._entry, self._version, self._md)
        to_remove = [v["uri"] for v in found]

        store._remove_entry_file(self._entry, to_remove, self._purge)


class SearchStore(StoreCommand):
    """A class serving kosh-queries"""

    __cli_options__ = [
        ["entry", "e", "Query this entry for information", {"choices": AMSDataStore.valid_entries}],
        ["version", "v", "Specific version to query for", {"default": None}],
        ["metadata", "m", "metadata to search for", {"default": None}],
        ["field", "f", "Return field of requested item", {"default": None}],
    ]

    def __init__(self, entry=None, metadata=None, version=None, field=None, **kwargs):
        """
        Initializes the 'SearchStore' class with the specific query

        Args:
            entry: Optional name of an ensemble to search for names
            metadata: JSON-like string with key-values to match the kosh-store
            version: Request a search for a specific version
            field: Print only the specified field
        """
        super().__init__(**kwargs)

        if entry not in [None] + list(AMSDataStore.valid_entries):
            raise RuntimeError(f"{entry} is not a valid entry for AMSStore")

        self._entry = entry
        self._md = {}
        if metadata != None:
            self._md = json.loads(metadata)

        self._version = None
        if version is not None:
            self._version = "latest"
            if version != "latest":
                assert version.isdigit(), "Version must be an integer"
                self._version = int(version)

        self._field = field

    def __call__(self, store):
        """Searches the 'sttore' for the requested queries"""
        found = store.search(self._entry, self._version, self._md)

        if self._field != None:
            found = [p[self._field] for p in found]
            for p in found:
                print(p)
            return

        print(json.dumps(found, indent=4))

        return


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(dest="command", help="Commands to the AMS database")

    add_parser = subparsers.add_parser("add", help="Add information to the AMS database")
    AddToStore.add_cli_options(add_parser)

    delete_parser = subparsers.add_parser("remove", help="Remove/Delete entry from the AMS database")
    RemoveFromStore.add_cli_options(delete_parser)

    query_parser = subparsers.add_parser("query", help="Query AMS database")
    SearchStore.add_cli_options(query_parser)

    create_parser = subparsers.add_parser("create", help="Create an AMS database")
    CreateStore.add_cli_options(create_parser)

    args = parser.parse_args()

    if args.command == "add":
        action_cls = AddToStore
    elif args.command == "remove":
        action_cls = RemoveFromStore
    elif args.command == "query":
        action_cls = SearchStore
    elif args.command == "create":
        action_cls = CreateStore
    else:
        raise RuntimeError("Unknown action to be performed")

    action = action_cls.from_cli(args)

    with AMSDataStore(action.ams_config.db_path, action.ams_config.db_store, action.ams_config.name, False) as store:
        action(store)


if __name__ == "__main__":
    main()
