# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import threading
import signal
import time
from queue import Queue
import warnings
from enum import Enum
from typing import Optional

import flux
from ams.monitor import AMSMonitor
from ams.stage import RMQLoaderTask
from ams.rmq import AMSSyncProducer, AMSRMQConfiguration
from ams.stage import MessageType, QueueMessage
from ams.ams_jobs import AMSJob
from ams.ams_flux import AMSFluxOrchestratorExecutor, AMSFluxExecutorFuture, AMSFakeFluxOrchestatorExecutor
from functools import wraps


def thread_safe_call(lock_name):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):  # 'self' is necessary here
            lock = getattr(self, lock_name)  # Accessing the lock via 'self'
            with lock:
                return func(self, *args, **kwargs)  # 'self' is passed along to the original function

        return wrapper

    return decorator


class JobState(Enum):
    IDLE = 1
    QUEUE = 2
    SUBSELECT = 3
    TRAINING = 4


class DomainSpec:
    class TrackJobStats:
        def __init__(self):
            self.untrained_sizes = 0
            self.num_jobs = 0
            self.num_done_jobs = 0
            self.num_cancelled_jobs = 0
            self.running = False

        def start(self):
            self.running = True
            self.num_jobs += 1

        def done(self):
            self.running = False
            self.num_done_jobs += 1

        def cancel(self):
            self.running = False
            self.num_cancelled_jobs += 1

    @staticmethod
    def done_train_cb(future):
        if not isinstance(future, AMSFluxExecutorFuture):
            raise TypeError(f"Done job call back received a future of an unsupported type: {type(future)}")

        domain_handle = future.get_domain_descr()
        if domain_handle is None:
            raise ValueError("Domain description is not set")

        flux_executor = future.flux_executor()
        if flux_executor is None:
            raise ValueError("Flux executor is not set")

        # TODO: Is it done/cancelled or failed?
        domain_handle.done_train()

        flux_executor.get_o_queue().put(
            QueueMessage(MessageType.Process, {"request_type": "done-training", "domain": domain_handle.name})
        )
        domain_handle.state = JobState.IDLE
        # TODO: Overhere I will need to push some of the results in a tracking queue.
        # so that we have an idea on which jobs are done etc...

    @staticmethod
    def done_sub_select_cb(future):
        if not isinstance(future, AMSFluxExecutorFuture):
            raise TypeError(f"Done job call back received a future of an unsupported type: {type(future)}")

        domain_handle = future.get_domain_descr()
        if domain_handle is None:
            raise ValueError("Domain description is not set")

        flux_executor = future.flux_executor()
        if flux_executor is None:
            raise ValueError("Flux executor is not set")

        # TODO: Is it done/cancelled or failed?
        domain_handle.done_sub_select()

        train_spec = domain_handle.train_job_spec.to_flux_jobspec()
        domain_handle.state = JobState.TRAINING
        submit_fut = flux_executor.submit(domain_handle, train_spec)
        submit_fut.add_done_callback(DomainSpec.done_train_cb)

        # TODO: Overhere I will need to push some of the results in a tracking queue.
        # so that we have an idea on which jobs are done etc...

    def __init__(self, name):
        self.training = DomainSpec.TrackJobStats()
        self.sub_select = DomainSpec.TrackJobStats()
        self._name = name
        self._trained_data = 0
        self._candidate_data = 0
        self._lock = threading.Lock()
        self._in_queue = False
        self._sub_select_job_spec = None
        self._train_job_spec = None
        self._state = JobState.IDLE

    @property
    def name(self):
        """The name property."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @thread_safe_call("_lock")
    def start_train(self):
        self.training.start()

    @thread_safe_call("_lock")
    def done_train(self):
        self.training.done()

    @thread_safe_call("_lock")
    def cancelled_train(self):
        self.training.cancel()

    @thread_safe_call("_lock")
    def start_sub_select(self):
        self.sub_select.start()

    @thread_safe_call("_lock")
    def done_sub_select(self):
        self.sub_select.done()

    @thread_safe_call("_lock")
    def cancelled_sub_select(self):
        self.sub_select.cancel()

    @property
    def trained_data(self):
        """The trained_data property."""
        return self._trained_data

    @property
    def in_queue(self):
        """The in_queue property."""
        return self._in_queue

    @in_queue.setter
    @thread_safe_call("_lock")
    def in_queue(self, value):
        self._in_queue = value

    @trained_data.setter
    @thread_safe_call("_lock")
    def trained_data(self, value):
        self._trained_data = value

    @property
    def candidate_data(self):
        """The candidate_data property."""
        return self._candidate_data

    @candidate_data.setter
    @thread_safe_call("_lock")
    def candidate_data(self, value):
        self._candidate_data = value

    @property
    def train_job_spec(self):
        """The train_job_spec property."""
        return self._train_job_spec

    @train_job_spec.setter
    @thread_safe_call("_lock")
    def train_job_spec(self, value):
        if isinstance(value, AMSJob):
            self._train_job_spec = value
        elif isinstance(value, dict):
            self._train_job_spec = AMSJob.from_dict(value)
        else:
            raise ValueError("The train job spec expects either a dictionary or a AMSJob type")
        return

    @property
    def sub_select_job_spec(self):
        """The stage_job_spec property."""
        return self._sub_select_job_spec

    @sub_select_job_spec.setter
    @thread_safe_call("_lock")
    def sub_select_job_spec(self, value):
        if isinstance(value, AMSJob):
            self._sub_select_job_spec = value
        elif isinstance(value, dict):
            self._sub_select_job_spec = AMSJob.from_dict(value)
        else:
            raise ValueError("The train job spec expects either a dictionary or a AMSJob type")
        return

    @thread_safe_call("_lock")
    def running(self):
        return self._state != JobState.IDLE

    def estimated_effort(self):
        return self._candidate_data - self._trained_data

    @property
    def state(self):
        """The state property."""
        return self._state

    @state.setter
    def state(self, value):
        if not isinstance(value, JobState):
            raise TypeError(f"Job state needs to be of type JobState but was of type {type(value)}")
        self._state = value

    def fully_described(self):
        return self._sub_select_job_spec is not None and self._train_job_spec is not None


class AvailableDomains(dict):
    """
    All the Available Domains/Jobs
    Attributes:
        _global_lock: A global lock to be used for accesing the datastructures
        _active_domains: The domains registered
    """

    def __init__(self, global_lock: threading.Lock, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_lock = global_lock
        self._active_domains = dict()

    def __contains__(self, key):
        with self._global_lock:
            return key in self._active_domains

    def __setitem__(self, domain_name, value: DomainSpec):
        if not isinstance(value, DomainSpec):
            raise TypeError(
                f"Catalog of available domains expects only DomainSpec as a field, instead got {type(value)}"
            )
        if domain_name != value.name:
            raise ValueError(f"Inconsistencies in key/value name Key is {domain_name} Value name is {value.name}")
        with self._global_lock:
            if domain_name not in self._active_domains:
                self._active_domains[domain_name] = value
            else:
                self._active_domains[domain_name].sub_select_job_spec = value.sub_select_job_spec
                self._active_domains[domain_name].train_job_spec = value.train_job_spec

    def __getitem__(self, key):
        return self._active_domains[key]

    def get(self, key, default=None):
        with self._global_lock:
            return self._active_domains.get(key, None)

    def items(self):
        for key, value in self._active_domains.items():
            yield key, value

    def keys(self):
        return self._active_domains.keys()

    def __len__(self):
        return len(self.keys())


class AMSJobReceiverStage(RMQLoaderTask):
    """
    A AMSJobReceiver receives job specifications for existing domains running on the
    system and get status updates regarding how many new data elements are being inserted in our database.

    Attributes:
        o_queue: The output queue to write the transformed messages
        credentials: A JSON file with the credentials to log on the RabbitMQ server.
        certificates: TLS certificates
        rmq_queue: The RabbitMQ queue to listen to.
        prefetch_count: Number of messages prefected by RMQ (impact performance)
    """

    def __init__(
        self,
        o_queue: Queue,
        host: str,
        port: int,
        vhost: str,
        user: str,
        password: str,
        cert: str,
        rmq_queue: str,
        from_file: Optional[str] = None,
        prefetch_count: Optional[int] = 1,
        signals=[signal.SIGTERM, signal.SIGINT, signal.SIGUSR1],
    ):
        super().__init__(
            o_queue,
            host,
            port,
            vhost,
            user,
            password,
            cert,
            rmq_queue,
            "thread",
            prefetch_count,
            signals=[signal.SIGTERM, signal.SIGINT, signal.SIGUSR1],
        )
        self.num_messages = 0
        self._from_file = from_file

    def callback_message(self, ch, basic_deliver, properties, body):
        """
        Callback to be called each time the RMQ client consumes a message.
        """
        start_time = time.time()
        data = json.loads(body)

        self.o_queue.put(QueueMessage(MessageType.Process, data))

        self.num_messages += 1
        self.total_time += time.time() - start_time

    @AMSMonitor(record=["total_time", "num_messages"])
    def __call__(self):
        """
        Busy loop of consuming messages from RMQ queue
        """
        if self._from_file is None:
            self.rmq_consumer.run()
        else:
            with open(self._from_file, "r") as fd:
                requests = json.load(fd)
                for r in requests:
                    item = [r]
                    self.callback_message(None, None, None, json.dumps(item))


class RequestProcessor:
    """
    @brief a dataflow step that reads a message from the AMSJobReceiverStage.

    This step performs the following actions:
        1. Registers job-specification into the global known 'AvailableDomains'
        2. Updates the size of collected data in the candidate database

        and through a heuristic informs the next stage to scheudle jobs
    """

    def __init__(self, i_queue: Queue, o_queue: Queue, domains: AvailableDomains):
        self.i_queue = i_queue
        self.o_queue = o_queue
        self._domains = domains

    def new_candidates(self, domain, size):
        if domain not in self._domains:
            raise KeyError(f"Domain {domain} does not exist in tracking job descriptions")

        print(f"Updating candidate from size {self._domains[domain].candidate_data} with {size}")
        self._domains[domain].candidate_data += size

    def register_job_spec(self, domain: str, spec: dict):
        """
        @brief Updates the specification of jobs in the AvailableDomains
        """
        if domain not in self._domains:
            self._domains[domain] = DomainSpec(domain)

        if spec["job_type"] == "sub_select":
            print("Setting sub select spec")
            self._domains[domain].sub_select_job_spec = spec["spec"]
        elif spec["job_type"] == "train":
            self._domains[domain].train_job_spec = spec["spec"]
        else:
            raise ValueError(f"Unknown Job Type:{spec['job_type']}")
        return

    def process_request(self, domain, data):
        if "request_type" not in data:
            raise ValueError("Received Request does not have request_type field")

        if data["request_type"] == "new_candidates":
            self.new_candidates(domain, data["size"])
        elif data["request_type"] == "register_job_spec":
            self.register_job_spec(domain, data)
        else:
            warnings.warn("Unknown request type:", data["request_type"])

    def __call__(self):
        while True:
            request = self.i_queue.get(block=True)
            if request.is_terminate():
                self.o_queue.put(QueueMessage(MessageType.Terminate, None))
                break

            if request.is_process():
                data = request.data()
                for v in data:
                    if "domain_name" not in v:
                        raise ValueError("Expected a domain name specification")
                    self.process_request(v["domain_name"], v)

            for i, (k, v) in enumerate(self._domains.items()):
                # TODO: We should not submit a job if size of candidates is 0 <-- be careful this can lead to edge cases
                #       of receiving candidate size 0 and rescheduling some work.
                # TODO: Here we need a better heuristic to decide when to schedule a job
                # 1. Job can only be schedule when they are completely described, there is
                # both a sub-selection job and a trainin job description
                # 2. We don't have another job running
                # 3. Maybe increase the priority through flux.
                # 4. Priority list and some sorting criteria.
                # 5. Use the "new_candidates" size message as a heuristic to drive the frequency of job submission.
                #    or the priority of a submitted job.
                # 6. Nice to have a 'programmable' way to drive a heuristic. Like a callable.
                if v.state == JobState.IDLE and v.fully_described():
                    v.state = JobState.QUEUE
                    print(f"Scheduling  domain {k}")
                    self.o_queue.put(QueueMessage(MessageType.Process, {"request_type": "schedule", "domain": k}))
                else:
                    print("Skip cause job is running")


class TrainJobScheduler:
    """
    @brief a dataflow step that accepts (eventually) requests from the 'RequestProcessor'
        to schedule jobs in an existing flux instance. A job in the context of the AMSFluxOrchestratorExecutor
        is the pair of sub-selection job and a training job.
    """

    def __init__(self, flux_uri: str, i_queue: Queue, o_queue: Queue, domains: AvailableDomains, fake_flux=False):
        self._flux_uri = flux_uri
        self.i_queue = i_queue
        self.o_queue = o_queue
        self._domains = domains
        self._fake_flux = fake_flux

    def _schedule(self, executor, domain_name: str):
        "Schedules the job described by 'domain_name'"
        domain_handle = self._domains.get(domain_name, None)
        if domain_handle is None:
            raise KeyError(f"{domain_name} does not exist in registered domains {self.domains.keys()}")

        if domain_handle.sub_select_job_spec is None:
            raise ValueError(f"{domain_name} does not have a sub-select job specification")
        if domain_handle.train_job_spec is None:
            raise ValueError(f"{domain_name} does not have a train job specification")
        sub_select_spec = domain_handle.sub_select_job_spec.to_flux_jobspec()
        domain_handle.start_sub_select()
        domain_handle.state = JobState.SUBSELECT
        submit_fut = executor.submit(domain_handle, sub_select_spec)
        submit_fut.add_done_callback(DomainSpec.done_sub_select_cb)

    def _run(self, executor):
        while True:
            request = self.i_queue.get(block=True)
            if request.is_terminate():
                self.o_queue.put(QueueMessage(MessageType.Terminate, None))
                break
            elif request.is_process():
                data = request.data()
                if data["request_type"] == "schedule":
                    self._schedule(executor, data["domain"])
                self._schedule(executor, data["domain"])

    def _local_run(self):
        with AMSFakeFluxOrchestatorExecutor(self.o_queue, self._domains, max_workers=1) as fake_executor:
            print("Scheduling with fake executor", type(fake_executor))
            self._run(fake_executor)

    def _flux_run(self):
        # The AMSFluxOrchestratorExecutor context manager defaults to the FluxExecutor context manager and calls 'shutdown' and
        # waits for all pedning jobs/futures. As such we do not need any special
        # handling of the jobs. CallBacks will make it work
        with AMSFluxOrchestratorExecutor(
            self.o_queue, self._domains, threads=1, handle_args=(self._flux_uri,)
        ) as flux_executor:
            self._run(flux_executor)

    def __call__(self):
        if self._fake_flux:
            self._local_run()
        else:
            self._flux_run()


class RMQStatusUpdate:
    def __init__(
        self,
        i_queue: Queue,
        host: str,
        port: int,
        vhost: str,
        user: str,
        password: str,
        cert: str,
        publish_queue: str,
        signals=[signal.SIGTERM, signal.SIGINT, signal.SIGUSR1],
    ):

        self.producer = AMSSyncProducer(host, port, vhost, user, password, cert, publish_queue)
        self.publish_queue = publish_queue
        self.i_queue = i_queue

    def __call__(self):
        with self.producer as fd:
            while True:
                request = self.i_queue.get(block=True)
                if request.is_terminate():
                    fd.send_message(json.dumps({"request_type": "terminate"}))
                    return
                elif request.is_process():
                    fd.send_message(json.dumps(request.data()))


class StatusPrinter:
    def __init__(self, i_queue):
        self.i_queue = i_queue

    def __call__(self):
        while True:
            request = self.i_queue.get(block=True)
            if request.is_terminate():
                print("Received", json.dumps({"request_type": "terminate"}))
                return
            elif request.is_process():
                print("Received", json.dumps(request.data()))


class AMSRMQMessagePrinter(RMQLoaderTask):
    """
    A AMSJobReceiver receives job specifications for existing domains running on the
    system and get status updates regarding how many new data elements are being inserted in our database.

    Attributes:
        o_queue: The output queue to write the transformed messages
        credentials: A JSON file with the credentials to log on the RabbitMQ server.
        certificates: TLS certificates
        rmq_queue: The RabbitMQ queue to listen to.
        prefetch_count: Number of messages prefected by RMQ (impact performance)
    """

    def __init__(
        self,
        host: str,
        port: int,
        vhost: str,
        user: str,
        password: str,
        cert: str,
        rmq_queue: str,
        prefetch_count: Optional[int] = 1,
        signals=[signal.SIGTERM, signal.SIGINT, signal.SIGUSR1],
    ):
        super().__init__(
            host,
            port,
            vhost,
            user,
            password,
            cert,
            rmq_queue,
            "thread",
            prefetch_count,
            signals=[signal.SIGTERM, signal.SIGINT, signal.SIGUSR1],
        )
        self.num_messages = 0

    def callback_message(self, ch, basic_deliver, properties, body):
        """
        Callback to be called each time the RMQ client consumes a message.
        """
        start_time = time.time()
        data = json.loads(body)
        self.num_messages += 1
        self.total_time += time.time() - start_time

    @AMSMonitor(record=["total_time", "num_messages"])
    def __call__(self):
        """
        Busy loop of consuming messages from RMQ queue
        """
        self.rmq_consumer.run()


def run(flux_uri, rmq_config, file=None, fake_flux=False, fake_rmq_update=False):
    tasks = []
    rmq_config = AMSRMQConfiguration.from_json(rmq_config)
    rmq_o_queue = Queue()
    tasks.append(
        AMSJobReceiverStage(
            rmq_o_queue,
            rmq_config.service_host,
            rmq_config.service_port,
            rmq_config.rabbitmq_vhost,
            rmq_config.rabbitmq_user,
            rmq_config.rabbitmq_password,
            rmq_config.rabbitmq_cert,
            rmq_config.rabbitmq_ml_submit_queue,
            from_file=file,
        )
    )

    # Use the global lock in case of synchronization
    global_lock = threading.Lock()
    # All the training jobs in the system their status and at some point some loggin information
    domains = AvailableDomains(global_lock)
    # The queue to be used to push jobs into flux
    flux_in_queue = Queue()
    # The 'thread' that will transform rmq messages to scheduling events
    tasks.append(RequestProcessor(rmq_o_queue, flux_in_queue, domains))
    # The queue to be used to publish finished jobs etc
    flux_out_queue = Queue()
    # The flux scheduler
    tasks.append(TrainJobScheduler(flux_uri, flux_in_queue, flux_out_queue, domains, fake_flux=fake_flux))
    if not fake_rmq_update:
        # The rmq publisher of done jobs
        tasks.append(
            RMQStatusUpdate(
                flux_out_queue,
                rmq_config.service_host,
                rmq_config.service_port,
                rmq_config.rabbitmq_vhost,
                rmq_config.rabbitmq_user,
                rmq_config.rabbitmq_password,
                rmq_config.rabbitmq_cert,
                rmq_config.rabbitmq_ml_status_queue,
            )
        )
    else:
        tasks.append(StatusPrinter(flux_out_queue))

    threads = []
    for t in tasks:
        threads.append(threading.Thread(target=t))

    for t in threads:
        t.start()

    for e in threads:
        e.join()
