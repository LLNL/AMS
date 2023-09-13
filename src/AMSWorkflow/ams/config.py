# -*- coding: utf-8 -*-
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
from pathlib import Path
from os import environ

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

    def __init__(self, config = None):
        if environ.get('AMS_CONFIG_FILE') is not None:
            ams_conf_fn = Path(environ.get('AMS_CONFIG_FILE'))
            if not ams_conf_fn.exists():
                raise RuntimeError(f'AMS_CONFIG_FILE is set to {ams_conf_fn} but file does not exist')
            config = ams_conf_fn

        if config is None:
            raise RuntimeError(f'{self.__class__.__name__} valid config is missing')

        self._config = config

        with open(config, 'r') as fd:
            self._options = json.load(fd)

        if 'name' not in self._options:
            raise RuntimeError('AMS configuration does not include \'name\' field.')

        self._name = self._options['name']

        db = self._options['db']
        if db is None:
            raise RuntimeError('AMS config file expects a \'db\' entry\n')

        self._db_path = db['path']
        if not Path(self._db_path).exists():
            raise RuntimeError(f'AMS data base path {self._db_path} should exist\n')

        self._db_type = db['type']

        self._db_store = 'ams_store.sql'
        if 'store' in db:
            self._db_store = db['store']

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
    def config(self) -> str:
        return self._config

