# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import setuptools

setuptools.setup(
    name="ams-wf",
    version="1.0",
    packages=['ams_wf', 'ams'],
    install_requires = [
        'argparse',
        'pika',
        ],
    entry_points={
        'console_scripts': [
            'AMSDBStage=ams_wf.AMSDBStage:main',
            'AMSOrchestrator=ams_wf.AMSOrchestrator:main',
            'AMSTrain=ams_wf.AMSTrain:main']
    },
)
