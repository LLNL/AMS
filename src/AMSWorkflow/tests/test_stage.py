#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import glob
import os
import signal
import tempfile
import unittest

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
        tmp_dict = dict()
        tmp_dict["name"] = "ams_test"
        db = dict()
        db["path"] = cls.o_dir
        db["type"] = "hdf5"
        db["store"] = "test_ams.sql"
        tmp_dict["ams_persistent_db"] = db
        # Initialize our sigleton
        AMSInstance(config=tmp_dict)

    def setUp(self):
        self.i_dir = TestStage.i_dir
        self.o_dir = TestStage.o_dir

    @classmethod
    def tearDownClass(cls):
        for f in glob.glob(f"{TestStage.i_dir}/*"):
            os.remove(f)
        os.rmdir(cls.i_dir)
        for f in glob.glob(f"{TestStage.o_dir}/*"):
            os.remove(f)
        os.rmdir(cls.o_dir)

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
            versions = fd.get_candidate_versions(True)
            r_inputs = list()
            r_outputs = list()
            files = list()
            for k in sorted(versions.keys()):
                for fn in versions[k]:
                    files.append(fn)
                    with reader(fn) as tmp_fd:
                        i, o = tmp_fd.load()
                        r_inputs.append(i)
                        r_outputs.append(o)
            pipe_in_data = np.concatenate(r_inputs, axis=0)
            origin_in_data = np.concatenate([d[0] for d in data], axis=0)
            self.assertTrue(
                np.array_equal(pipe_in_data, origin_in_data), "inputs do not match after writting them with pipeline"
            )

            pipe_out_data = np.concatenate(r_outputs, axis=0)
            origin_out_data = np.concatenate([d[1] for d in data], axis=0)
            self.assertTrue(
                np.array_equal(pipe_out_data, origin_out_data), "outputs do not match after writting them with pipeline"
            )
            fd.remove_candidates(data_files=files, delete_files=True)

        return

    def test_fs_pipeline(self):
        data = list()
        for i in range(0, 10):
            in_data, out_data = np.random.rand(300, 2), np.random.rand(300, 3)
            data.append((in_data, out_data))

        for wr in stage.Pipeline.supported_writers:
            writer = get_writer(wr)
            reader = get_reader(wr)
            # Write the files to disk
            for j, (i, o) in enumerate(data):
                fn = "{0}/data_{1}.{2}".format(self.i_dir, j, writer.get_file_format_suffix())
                with writer(fn) as fd:
                    fd.store(i, o)

            for p in stage.Pipeline.supported_policies:
                if "thread" in p:
                    continue

                pipe = stage.FSPipeline(
                    True,
                    self.o_dir,
                    None,
                    writer.get_file_format_suffix(),
                    self.i_dir,
                    writer.get_file_format_suffix(),
                    "*.{0}".format(writer.get_file_format_suffix()),
                )
                with timeout(
                    10,
                    error_message="Copying files from {0} to {1} using writer {2} and policy {3}".format(
                        self.i_dir, self.o_dir, wr, p
                    ),
                ):
                    pipe.execute(p)
                self.verify(data, reader)


if __name__ == "__main__":
    unittest.main()
