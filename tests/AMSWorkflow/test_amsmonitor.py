#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import datetime
import json
import os
import time
import unittest

from ams.monitor import AMSMonitor
from ams.stage import Task


class ExampleTask1(Task):
    def __init__(self):
        self.x = 0
        self.y = 100

    @AMSMonitor()
    def __call__(self):
        i = 0
        with AMSMonitor(obj=self, record=["x"], tag="while_loop", accumulate=False):
            while 1:
                time.sleep(1)
                self.x += i
                if i == 3:
                    break
                i += 1
        self.y += 100


def read_json(path: str):
    with open(path) as f:
        d = json.load(f)
    return d


class TestMonitorTask1(unittest.TestCase):
    def setUp(self):
        self.task1 = ExampleTask1()

    def test_populating_monitor(self):
        AMSMonitor.reset()
        self.task1()

        self.assertNotEqual(AMSMonitor.stats.copy(), {})
        self.assertIn("ExampleTask1", AMSMonitor.stats)
        self.assertIn("while_loop", AMSMonitor.stats["ExampleTask1"])
        self.assertIn("__call__", AMSMonitor.stats["ExampleTask1"])

        for ts in AMSMonitor.stats["ExampleTask1"]["__call__"].keys():
            self.assertIsInstance(datetime.datetime.strptime(ts, AMSMonitor.format_ts), datetime.datetime)
            self.assertIn("x", AMSMonitor.stats["ExampleTask1"]["__call__"][ts])
            self.assertIn("y", AMSMonitor.stats["ExampleTask1"]["__call__"][ts])
            self.assertIn("amsmonitor_duration", AMSMonitor.stats["ExampleTask1"]["__call__"][ts])
            self.assertEqual(AMSMonitor.stats["ExampleTask1"]["__call__"][ts]["x"], 6)
            self.assertEqual(AMSMonitor.stats["ExampleTask1"]["__call__"][ts]["y"], 200)

        for ts in AMSMonitor.stats["ExampleTask1"]["while_loop"].keys():
            self.assertIsInstance(datetime.datetime.strptime(ts, AMSMonitor.format_ts), datetime.datetime)
            self.assertIn("x", AMSMonitor.stats["ExampleTask1"]["while_loop"][ts])
            self.assertIn("amsmonitor_duration", AMSMonitor.stats["ExampleTask1"]["while_loop"][ts])
            self.assertEqual(AMSMonitor.stats["ExampleTask1"]["while_loop"][ts]["x"], 6)

    def test_json_output(self):
        print(f"test_json_output {AMSMonitor.stats.copy()}")
        AMSMonitor.reset()
        self.task1()
        path = "test_amsmonitor.json"
        AMSMonitor.json(path)
        self.assertTrue(os.path.isfile(path))
        d = read_json(path)
        self.assertEqual(AMSMonitor.stats.copy(), d)

    def tearDown(self):
        try:
            os.remove("test_amsmonitor.json")
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
