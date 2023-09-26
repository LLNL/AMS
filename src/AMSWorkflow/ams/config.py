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

    def __init__(self, config=None):
        if environ.get("AMS_CONFIG_FILE") is not None:
            ams_conf_fn = Path(environ.get("AMS_CONFIG_FILE"))
            if not ams_conf_fn.exists():
                raise RuntimeError(f"AMS_CONFIG_FILE is set to {ams_conf_fn} but file does not exist")
            config = ams_conf_fn
            with open(config, "r") as fd:
                self._options = json.load(fd)
        else:
            self._options = config

        if config is None:
            raise RuntimeError(
                f"{self.__class__.__name__} valid config is missing please set AMS_CONFIG_FILE env variable to point to a valid config file"
            )


        if "name" not in self._options:
            raise RuntimeError("AMS configuration does not include 'name' field.")

        self._name = self._options["name"]

        db = self._options.get("ams_persistent_db", None)
        if db is None:
            raise RuntimeError("AMS config file expects a 'db' entry\n")

        self._db_path = db["path"]

        self._db_type = db["type"]

        self._db_store = "ams_store.sql"
        if "store" in db:
            self._db_store = db["store"]

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

    @property
    def do_stage(self) -> bool:
        return self._stage

    @property
    def stage_dir(self) -> str:
        return self._stage_dir
