from threading import Lock
import unittest
import json
from queue import Queue
from unittest.mock import patch, MagicMock

from ams import orchestrator
from ams.stage import MessageType, QueueMessage


class TestAvailableDomains(unittest.TestCase):
    def test_contains(self):
        domains = orchestrator.AvailableDomains(Lock())
        domains["test"] = orchestrator.DomainSpec("test")
        self.assertTrue("test" in domains, "__setitem__ does not work correctly")
        self.assertFalse("test1" in domains, "__setitem__ does not work correctly (item should not exist)")

    @unittest.expectedFailure
    def test_contains_wrong_name(self):
        domains = orchestrator.AvailableDomains(Lock())
        domains["test1"] = orchestrator.DomainSpec("test")

    def test_len(self):
        domains = orchestrator.AvailableDomains(Lock())
        keys = [f"test_{i}" for i in range(10)]
        for k in keys:
            domains[k] = orchestrator.DomainSpec(k)
        self.assertTrue(len(domains) == 10, "__len__ not operating correctly")

    def test_keys(self):
        domains = orchestrator.AvailableDomains(Lock())
        keys = {f"test_{i}" for i in range(10)}
        for k in keys:
            domains[k] = orchestrator.DomainSpec(k)
        self.assertTrue(len(keys.difference(domains.keys())) == 0, "Set should be empty")


class TestAMSJobReceiverStage(unittest.TestCase):
    @patch("ams.orchestrator.RMQDomainDataLoaderTask")
    def test_jobreceiver(self, MockRMQLoaderTask):
        # This is the 'fake' class instance of RMQLoaderTask
        def mock_init(self, queue, *args, **kwargs):
            self.queue = queue

        MockRMQLoaderTask.side_effect = mock_init
        q = Queue()
        job_receiver_instance = orchestrator.AMSJobReceiverStage(q, "", "", "", "", "", "", "", "")
        dicts = [{f"test_{v}": v} for v in range(10)]
        for d in dicts:
            job_receiver_instance.callback_message("", "", "", json.dumps(d))

        result_list = []
        while not q.empty():
            msg = q.get()
            self.assertTrue(msg.is_process(), "Expecting a 'process' message in the output queue")
            result_list.append(msg.data())

        for i, (msg1, msg2) in enumerate(zip(dicts, result_list)):
            key = f"test_{i}"
            self.assertTrue(key in msg1, "key test_{i} expected to be in entry message")
            self.assertTrue(key in msg2, "key test_{i} expected to be in outro message")
            self.assertTrue(i == msg2[key], "value {i} expected to be in outro message")


class TestRequestProcessor(unittest.TestCase):
    req_train_spec = [
        {
            "request_type": "register_job_spec",
            "domain_name": "test",
            "job_type": "train",
            "spec": {
                "name": "test",
                "executable": "echo",
                "stdout": "test.out",
                "stderr": "test.err",
                "cli_args": ["my", "name", "is", "ams", "and", "I", "emulate", "a", "train", "job"],
                # This is important to allow correct checking of dictionaries
                "cli_kwargs": {},
                "resources": {
                    "nodes": 1,
                    "tasks_per_node": 4,
                    "cores_per_task": 1,
                    "exclusive": False,
                    "gpus_per_task": 2,
                },
            },
        }
    ]

    req_sub_select_spec = [
        {
            "request_type": "register_job_spec",
            "domain_name": "test",
            "job_type": "sub_select",
            "spec": {
                "name": "test",
                "executable": "echo",
                "stdout": "test.out",
                "stderr": "test.err",
                "cli_args": ["my", "name", "is", "ams", "and", "emulate", "a", "sub-select", "job"],
                # This is important to allow correct checking of dictionaries
                "cli_kwargs": {},
                "resources": {
                    "nodes": 1,
                    "tasks_per_node": 4,
                    "cores_per_task": 1,
                    "exclusive": False,
                    "gpus_per_task": 2,
                },
            },
        }
    ]

    req_candidate_increase = [
        {
            "request_type": "new_candidates",
            "domain_name": "test",
            "size": 31 * 1024,
        },
    ]

    def test_reqprocessor_register_spec(self):
        domains = orchestrator.AvailableDomains(Lock())
        i_q = Queue()
        o_q = Queue()
        req_proc = orchestrator.RequestProcessor(i_q, o_q, domains)
        i_q.put(QueueMessage(MessageType.Process, TestRequestProcessor.req_train_spec))
        i_q.put(QueueMessage(MessageType.Process, TestRequestProcessor.req_sub_select_spec))
        i_q.put(QueueMessage(MessageType.Terminate, None))
        req_proc()
        self.assertTrue(len(domains) == 1, "Expected to have a single domain")
        self.assertTrue("test" in domains, "Expected to have a single domain")
        domain_spec = domains["test"]
        train_spec = domain_spec.train_job_spec.to_dict()
        sub_select_spec = domain_spec.sub_select_job_spec.to_dict()
        self.assertDictEqual(train_spec, TestRequestProcessor.req_train_spec[0]["spec"])
        self.assertDictEqual(sub_select_spec, TestRequestProcessor.req_sub_select_spec[0]["spec"])
        # We send to message of job spec. The test now will schedule a job on the second message.
        scheduled = o_q.get_nowait()
        self.assertTrue(
            scheduled.is_process() and scheduled.data()["request_type"] == "schedule",
            f"Last message should had been a terminate one {scheduled.data()}",
        )
        # The last message needs to be a schedule one.
        front = o_q.get_nowait()
        self.assertTrue(front.is_terminate(), f"Last message should had been a terminate one {front.data()}")
        for i in range(10):
            i_q.put(QueueMessage(MessageType.Process, TestRequestProcessor.req_candidate_increase))
        i_q.put(QueueMessage(MessageType.Terminate, None))
        req_proc()
        front = o_q.get_nowait()
        self.assertTrue(front.is_terminate(), "Last message should had been a terminate one")
        self.assertTrue(
            domain_spec.candidate_data == 10 * TestRequestProcessor.req_candidate_increase[0]["size"],
            f"Size has un expected value of {domain_spec.candidate_data}",
        )


if __name__ == "__main__":
    unittest.main()
