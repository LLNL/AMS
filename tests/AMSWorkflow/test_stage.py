#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import glob
import json
import os
import signal
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ams import stage, store
from ams.config import AMSInstance
from ams.faccessors import get_reader, get_writer


class TestTimeout(Exception):
    pass


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class TestStage(unittest.TestCase):
    i_dir = ""
    o_dir = ""

    @classmethod
    def setUpClass(cls):
        cls.i_dir = tempfile.mkdtemp()
        cls.o_dir = tempfile.mkdtemp()
        config_path = cls.o_dir / Path("ams_config.json")
        config = AMSInstance.create_config(cls.o_dir, "ams_store.sql", "test_name")
        with open(str(config_path), "w") as fd:
            json.dump(config, fd, indent=4)
        # Initialize our sigleton
        AMSInstance.from_path(cls.o_dir)

    def setUp(self):
        self.i_dir = TestStage.i_dir
        self.o_dir = TestStage.o_dir

    #    @classmethod
    #    def tearDownClass(cls):
    #        for entry in ["models", "candidates", "data"]:
    #            for f in glob.glob(f"{TestStage.o_dir}/{entry}/*"):
    #                Path(f).unlink()
    #            for f in glob.glob(f"{TestStage.i_dir}/*"):
    #                Path(f).unlink()
    #            Path(f"{TestStage.o_dir}/{entry}").rmdir()
    #        Path(f"{TestStage.o_dir}/ams_config.json").unlink()
    #        Path(f"{TestStage.o_dir}/ams_store.sql").unlink()
    #        Path(f"{TestStage.o_dir}").rmdir()
    #
    #        for f in glob.glob(f"{TestStage.i_dir}/*"):
    #            Path(f).unlink()
    #        Path(f"{TestStage.i_dir}").rmdir()

    def test_q_message(self):
        msg = stage.QueueMessage(stage.MessageType.Process, None)
        self.assertTrue(msg.is_process())
        self.assertFalse(msg.is_terminate())
        self.assertFalse(msg.is_new_model())

        msg = stage.QueueMessage(stage.MessageType.NewModel, None)
        self.assertFalse(msg.is_process())
        self.assertFalse(msg.is_terminate())
        self.assertTrue(msg.is_new_model())

        msg = stage.QueueMessage(stage.MessageType.Terminate, None)
        self.assertFalse(msg.is_process())
        self.assertTrue(msg.is_terminate())
        self.assertFalse(msg.is_new_model())

    def test_forward_task(self):
        from queue import Queue

        def fw_callback(ins, outs):
            return ins, outs

        i_q = Queue()
        o_q = Queue()
        fw_task = stage.ForwardTask(i_q, o_q, fw_callback)

        msgs = list()
        for i in range(0, 10):
            in_data, out_data = np.random.rand(3, 2), np.random.rand(3, 2)
            msgs.append((in_data, out_data))
            i_q.put(stage.QueueMessage(stage.MessageType.Process, stage.DataBlob(in_data, out_data)))

        i_q.put(stage.QueueMessage(stage.MessageType.Terminate, None))

        with timeout(10, error_message="ForwardTask took too long"):
            fw_task()

        for i, o in msgs:
            msg = o_q.get()
            self.assertTrue(msg.is_process(), "Received message of FWTask is not of type process")
            self.assertTrue(np.array_equal(i, msg.data().inputs), "Inputs do not match the fw task inputs")
            self.assertTrue(np.array_equal(o, msg.data().outputs), "Ouputs do not match the fw task outputs")

        msg = o_q.get()
        self.assertTrue(msg.is_terminate(), "Message should had been terminate")

    def verify(self, data, reader):
        ams_config = AMSInstance()
        with store.AMSDataStore(ams_config.db_path, ams_config.db_store, ams_config.name, False) as fd:
            versions = fd.get_candidate_versions("unknown-domain", associate_files=True)
            r_inputs = list()
            r_outputs = list()
            files = list()
            for k in sorted(versions.keys()):
                for fn in versions[k]:
                    files.append(fn)
                    with reader(fn) as tmp_fd:
                        _, i, o = tmp_fd.load()
                        r_inputs.append(i)
                        r_outputs.append(o)
            pipe_in_data = np.sort(np.concatenate(r_inputs, axis=0).flatten())
            origin_in_data = np.sort(np.concatenate([d[0] for d in data], axis=0).flatten())

            self.assertTrue(
                np.array_equal(pipe_in_data, origin_in_data),
                f"inputs {pipe_in_data} {origin_in_data} do not match after writting them with pipeline",
            )

            pipe_out_data = np.sort(np.concatenate(r_outputs, axis=0).flatten())
            origin_out_data = np.sort(np.concatenate([d[1] for d in data], axis=0).flatten())
            self.assertTrue(
                np.array_equal(pipe_out_data, origin_out_data),
                "outputs {pipe_out_data} {r_outputs} do not match after writting them with pipeline",
            )
            fd.remove_candidates("unknown-domain", data_files=files, delete_files=True)

        return

    def test_fs_pipeline(self):
        data = list()
        for i in range(0, 10):
            in_data, out_data = np.random.rand(3, 2), np.random.rand(3, 3)
            data.append((in_data, out_data))

        for src_fmt in stage.Pipeline.supported_writers:
            src_wr = get_writer(src_fmt)
            src_rd = get_reader(src_fmt)
            for dest_fmt in stage.Pipeline.supported_writers:
                dest_rd = get_reader(dest_fmt)
                dest_wr = get_writer(dest_fmt)

                for p in stage.Pipeline.supported_policies:
                    if "thread" in p:
                        continue

                    # Write the files to disk
                    for j, (i, o) in enumerate(data):
                        fn = "{0}/data_{1}.{2}".format(self.i_dir, j, src_wr.get_file_format_suffix())
                        with src_wr(fn) as fd:
                            fd.store(i, o)

                    pipe = stage.FSPipeline(
                        self.o_dir,
                        True,
                        self.o_dir,
                        None,
                        dest_fmt,
                        self.i_dir,
                        src_fmt,
                        "*.{0}".format(src_wr.get_file_format_suffix()),
                    )
                    with timeout(
                        10,
                        error_message="Copying files from {0} to {1} using writer {2} and policy {3}".format(
                            self.i_dir, self.o_dir, dest_fmt, p
                        ),
                    ):
                        pipe.execute(p)
                    self.verify(data, dest_rd)


if __name__ == "__main__":
    unittest.main()
