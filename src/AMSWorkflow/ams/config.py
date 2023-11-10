# -*- coding: utf-8 -*-
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
from os import environ
from pathlib import Path


class AMSSingleton(type):
    instance = None

    def __call__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super(AMSSingleton, cls).__call__(*args, **kwargs)
        return cls.instance


class AMSInstance(metaclass=AMSSingleton):
    """
    The class represents the ams configuration we run. It should be accessed by anywhere
    and get the configuration of AMS for the current process

    System is a singleton class.
    """

    def __init__(self, name, path, ftype, fstore):
        self._name = name
        self._db_path = path
        self._db_type = ftype
        self._db_store = "ams_store.sql"
        if fstore is not None:
            self._db_store = fstore

    @property
    def name(self) -> str:
        return self._name

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def db_type(self) -> str:
        return self._db_type

    @property
    def db_store(self) -> str:
        return self._db_store

    @staticmethod
    def create_config(store_path, store_name, name):
        return {"name": name, "ams_persistent_db": {"path": str(store_path), "type": "hdf5", "store": str(store_name)}}

    @classmethod
    def from_dict(cls, config):
        if len(config) == 0:
            raise RuntimeError(f"{cls.__name__} valid config is missing ")

        if "name" not in config:
            raise RuntimeError("AMS configuration does not include 'name' field.")

        db = config.get("ams_persistent_db", None)
        if db is None:
            raise RuntimeError("Config file expects a 'db' entry\n")

        for key in {"path", "type"}:
            assert key in db, f"Config does not have {k} entry"

        return cls(config["name"], db["path"], db["type"], db["store"] if "store" in db else None)

    @classmethod
    def from_env(cls):
        path = environ.get("AMS_CONFIG_FILE", None)
        if path is not None:
            ams_conf_fn = Path(path)
            if not ams_conf_fn.exists():
                raise RuntimeError(f"AMS_CONFIG_FILE is set to {ams_conf_fn} but file does not exist")
            config = ams_conf_fn
            with open(config, "r") as fd:
                config = json.load(fd)
            return cls.from_dict(config)
        return None

    @classmethod
    def from_path(cls, db_path):
        _fn = Path(db_path) / Path("ams_config.json")
        assert _fn.exists(), "AMS Configuration file does not exist"
        with open(str(_fn), "r") as fd:
            config = json.load(fd)

        return cls.from_dict(config)
