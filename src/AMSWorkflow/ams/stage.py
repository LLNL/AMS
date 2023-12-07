#!/usr/bin/env python3
# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import glob
import json
import os
import shutil
import signal
import sys
import os
import time
from abc import ABC, abstractclassmethod, abstractmethod
from enum import Enum
from multiprocessing import Process
from multiprocessing import Queue as mp_queue
from pathlib import Path
from queue import Queue as ser_queue
from threading import Thread
from typing import Callable
import warnings

import numpy as np
from ams.config import AMSInstance
from ams.faccessors import get_reader, get_writer
from ams.monitor import AMSMonitor
from ams.rmq import AMSMessage, AsyncConsumer
from ams.store import AMSDataStore
from ams.util import get_unique_fn

from flux.kvs import KVSDir
import flux

BATCH_SIZE = 32 * 1024 * 1024


class MessageType(Enum):
    Process = 1
    NewModel = 2
    Terminate = 3


class DataBlob:
    """
    Class wrapping input, outputs in a single class

    Attributes:
        inputs: A ndarray of the inputs.
        outputs: A ndarray of the outputs.
    """

    def __init__(self, inputs, outputs, domain_name=None):
        self._domain_name = domain_name
        self._inputs = inputs
        self._outputs = outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def domain_name(self):
        return self._domain_name


class QueueMessage:
    """
    A message in the IPC Queues.

    Attributes:
        msg_type: The type of the message. We currently support 3 types Process, NewModel, Terminate
        blob: The contents of the message
    """

    def __init__(self, msg_type, blob):
        if not isinstance(msg_type, MessageType):
            raise TypeError("Message Type should be of type MessageType")
        self.msg_type = msg_type
        self.blob = blob

    def is_terminate(self):
        return self.msg_type == MessageType.Terminate

    def is_process(self):
        return self.msg_type == MessageType.Process

    def is_new_model(self):
        return self.msg_type == MessageType.NewModel

    def data(self):
        return self.blob


class Task(ABC):
    """
    An abstract interface encapsulating a
    callable mechanism to be performed during
    the staging mechanism.
    """

    @abstractmethod
    def __call__(self):
        pass


class ForwardTask(Task):
    """
    A ForwardTask reads messages from some input queues performs some
    action/transformation and forwards the outcome to some output queue.

    Attributes:
        i_queue: The input queue to read input message
        o_queue: The output queue to write the transformed messages
        callback: A callback to be applied on every message before pushing it to the next stage.
    """

    def __init__(self, i_queue, o_queue, callback):
        """
        initializes a ForwardTask class with the queues and the callback.
        """
        if not isinstance(callback, Callable):
            raise TypeError(f"{callback} argument is not Callable")

        self.i_queue = i_queue
        self.o_queue = o_queue
        self.callback = callback
        self.datasize = 0

    def _action(self, data):
        """
        Apply an 'action' to the incoming data

        Args:
            data: A DataBlob of inputs, outputs to be transformed

        Returns:
            A pair of inputs, outputs of the data after the transformation
        """
        inputs, outputs = self.callback(data.inputs, data.outputs)
        # This can be too conservative, we may want to relax it later
        if not (isinstance(inputs, np.ndarray) and isinstance(outputs, np.ndarray)):
            raise TypeError(f"{self.callback.__name__} did not return numpy arrays")
        return inputs, outputs

    @AMSMonitor(record=["datasize"])
    def __call__(self):
        """
        A busy loop reading messages from the i_queue, acting on those messages and forwarding
        the output to the output queue. In the case of receiving a 'termination' messages informs
        the tasks waiting on the output queues about the terminations and returns from the function.
        """
        received_in = [0, 0]
        received_out = [0, 0]
        sent_in = [0, 0]
        sent_out = [0, 0]
        while True:
            # This is a blocking call
            item = self.i_queue.get(block=True)

            if item.is_terminate():
                self.o_queue.put(QueueMessage(MessageType.Terminate, None))
                break
            elif item.is_process():
                received_in = [o + i for o, i in zip(received_in, item.data().inputs.shape)]
                received_out = [o + i for o, i in zip(received_in, item.data().outputs.shape)]

                inputs, outputs = self._action(item.data())
                sent_in = [o + i for o, i in zip(sent_in, inputs.shape)]
                sent_out = [o + i for o, i in zip(sent_out, outputs.shape)]
                self.o_queue.put(QueueMessage(MessageType.Process, DataBlob(inputs, outputs)))
                self.datasize += inputs.nbytes + outputs.nbytes
            elif item.is_new_model():
                # This is not handled yet
                continue
        print(f"Inputs {received_in}/{sent_in} outputs {received_out}/{sent_out}")
        return


class FSLoaderTask(Task):
    """
    A FSLoaderTask reads files from the filesystem bundles the data of
    the files into batches and forwards them to the next task waiting on the
    output queuee.

    Attributes:
        o_queue: The output queue to write the transformed messages
        loader: A child class inheriting from FileReader that loads data from the filesystem.
        pattern: The (glob-)pattern of the files to be read.
    """

    def __init__(self, o_queue, loader, pattern):
        self.o_queue = o_queue
        self.pattern = pattern
        self.loader = loader
        self.datasize = 0

    @AMSMonitor(record=["datasize"])
    def __call__(self):
        """
        Busy loop of reading all files matching the pattern and creating
        '100' batches which will be pushed on the queue. Upon reading all files
        the Task pushes a 'Terminate' message to the queue and returns.
        """

        start = time.time()
        input_rows = 0
        output_rows = 0
        for fn in glob.glob(self.pattern):
            with self.loader(fn) as fd:
                input_data, output_data = fd.load()
                row_size = input_data[0, :].nbytes + output_data[0, :].nbytes
                input_rows += input_data.shape[0]
                output_rows += output_data.shape[0]
                rows_per_batch = int(np.ceil(BATCH_SIZE / row_size))
                num_batches = int(np.ceil(input_data.shape[0] / rows_per_batch))
                input_batches = np.array_split(input_data, num_batches)
                output_batches = np.array_split(output_data, num_batches)
                for j, (i, o) in enumerate(zip(input_batches, output_batches)):
                    self.o_queue.put(QueueMessage(MessageType.Process, DataBlob(i, o)))
                self.datasize += input_data.nbytes + output_data.nbytes
        self.o_queue.put(QueueMessage(MessageType.Terminate, None))

        end = time.time()
        print(f"Spend {end - start} at {self.__class__.__name__}")
        print(f"Read {input_rows} and {output_rows}")


class RMQLoaderTask(Task):
    """
    A RMQLoaderTask consumes data from RabbitMQ bundles the data of
    the files into batches and forwards them to the next task waiting on the
    output queuee.

    Attributes:
        o_queue: The output queue to write the transformed messages
        credentials: A JSON file with the credentials to log on the RabbitMQ server.
        certificates: TLS certificates
        rmq_queue: The RabbitMQ queue to listen to.
        prefetch_count: Number of messages prefected by RMQ (impact performance)
    """

    def __init__(
        self,
        o_queue,
        host,
        port,
        vhost,
        user,
        password,
        cert,
        rmq_queue,
        policy,
        prefetch_count=1,
        signals=[signal.SIGTERM, signal.SIGINT, signal.SIGUSR1],
    ):
        self.o_queue = o_queue
        self.cert = cert
        self.rmq_queue = rmq_queue
        self.prefetch_count = prefetch_count
        self.datasize = 0
        self.total_time = 0
        self.signals = signals
        self.orig_sig_handlers = {}
        self.policy = policy

        # Signals can only be used within the main thread
        if self.policy != "thread":
            # We ignore SIGTERM, SIGUSR1, SIGINT by default so later
            # we can override that handler only for RMQLoaderTask
            for s in self.signals:
                self.orig_sig_handlers[s] = signal.getsignal(s)
                signal.signal(s, signal.SIG_IGN)

        self.rmq_consumer = AsyncConsumer(
            host=host,
            port=port,
            vhost=vhost,
            user=user,
            password=password,
            cert=self.cert,
            queue=self.rmq_queue,
            on_message_cb=self.callback_message,
            on_close_cb=self.callback_close,
            prefetch_count=self.prefetch_count,
        )

    def callback_close(self):
        """
        Callback that will be called when RabbitMQ will close
        the connection (or if a problem happened with the connection).
        """
        self.o_queue.put(QueueMessage(MessageType.Terminate, None))

    def callback_message(self, ch, basic_deliver, properties, body):
        """
        Callback that will be called each time a message will be consummed.
        the connection (or if a problem happened with the connection).
        """
        start_time = time.time()
        domain_name, input_data, output_data = AMSMessage(body).decode()
        row_size = input_data[0, :].nbytes + output_data[0, :].nbytes
        rows_per_batch = int(np.ceil(BATCH_SIZE / row_size))
        num_batches = int(np.ceil(input_data.shape[0] / rows_per_batch))
        input_batches = np.array_split(input_data, num_batches)
        output_batches = np.array_split(output_data, num_batches)

        self.datasize += input_data.nbytes + output_data.nbytes

        for j, (i, o) in enumerate(zip(input_batches, output_batches)):
            self.o_queue.put(QueueMessage(MessageType.Process, DataBlob(i, o, domain_name)))

        self.total_time += time.time() - start_time

    def signal_wrapper(self, name, pid):
        def handler(signum, frame):
            print(f"Received SIGNUM={signum} for {name}[pid={pid}]: stopping process")
            self.stop()

        return handler

    def stop(self):
        self.rmq_consumer.stop()
        self.o_queue.put(QueueMessage(MessageType.Terminate, None))
        print(f"Spend {self.total_time} at {self.__class__.__name__}")

    @AMSMonitor(record=["datasize", "total_time"])
    def __call__(self):
        """
        Busy loop of consuming messages from RMQ queue
        """
        # Installing signal callbacks only for RMQLoaderTask
        if self.policy != "thread":
            for s in self.signals:
                signal.signal(s, self.signal_wrapper(self.__class__.__name__, os.getpid()))
        self.rmq_consumer.run()


class FSWriteTask(Task):
    """
    A Class representing a task flushing data in the specified output directory

    Attributes:
        i_queue: The input queue to read data from.
        o_queue: The output queue to write the path of the saved file.
        writer_cls: A child class inheriting from FileWriter that writes to the specified file.
        out_dir: The directory to write data to.
    """

    def __init__(self, i_queue, o_queue, writer_cls, out_dir):
        """
        initializes the writer task to read data from the i_queue write them using
        the writer_cls and store the data in the out_dir.
        """
        self.data_writer_cls = writer_cls
        self.out_dir = out_dir
        self.i_queue = i_queue
        self.o_queue = o_queue
        self.suffix = writer_cls.get_file_format_suffix()

    @AMSMonitor(record=["datasize"])
    def __call__(self):
        """
        A busy loop reading messages from the i_queue, writting the input,output data in a file
        using the instances 'writer_cls' and inform the task waiting on the output_q about the
        path of the file.
        """

        start = time.time()
        total_bytes_written = 0
        data_files = dict()
        # with self.data_writer_cls(fn) as fd:
        with AMSMonitor(obj=self, tag="internal_loop", accumulate=False):
            while True:
                # This is a blocking call
                item = self.i_queue.get(block=True)
                if item.is_terminate():
                    for k, v in data_files.items():
                        v[0].close()
                        self.o_queue.put(QueueMessage(MessageType.Process, v[0].file_name))
                    del data_files
                    self.o_queue.put(QueueMessage(MessageType.Terminate, None))
                    break
                elif item.is_process():
                    data = item.data()
                    if data.domain_name not in data_files:
                        fn = get_unique_fn()
                        fn = f"{self.out_dir}/{data.domain_name}_{fn}.{self.suffix}"
                        # TODO: bytes_written should be an attribute of the file
                        # to keep track of the size of the current file. Currently we keep track of this
                        # by keeping a value in a list
                        data_files[data.domain_name] = [self.data_writer_cls(fn).open(), 0]
                    bytes_written = data.inputs.size * data.inputs.itemsize
                    bytes_written += data.outputs.size * data.outputs.itemsize
                    data_files[data.domain_name][0].store(data.inputs, data.outputs)
                    data_files[data.domain_name][1] += bytes_written
                    total_bytes_written += data.inputs.size * data.inputs.itemsize
                    total_bytes_written += data.outputs.size * data.outputs.itemsize

                    if data_files[data.domain_name][1] >= 2 * 1024 * 1024 * 1024:
                        data_files[data.domain_name][0].close()
                        self.o_queue.put(QueueMessage(MessageType.Process, data_files[data.domain_name][0].file_name))
                        del data_files[data.domain_name]

        end = time.time()
        self.datasize = total_bytes_written
        print(f"Spend {end - start} {total_bytes_written} at {self.__class__.__name__}")

class PushToStore(Task):
    """
    PushToStore is the epilogue of the pipeline. Effectively (if instructed so) it informs the kosh store
    about the existence of a new file.

    Attributes:
        ams_config: The AMS configuration storing information regarding the AMS setup.
        i_queue: The queue to read file locations from
        dir: The directory of the database
        store: The Kosh Store
    """

    def __init__(self, i_queue, ams_config, db_path, store):
        """
        Tnitializes the PushToStore Task. It reads files from i_queue, if the file
        is not under db_path, it copies the file to this location and if store defined
        it makes the kosh-store aware about the existence of the file.
        """

        self.ams_config = ams_config
        self.i_queue = i_queue
        self.dir = Path(db_path).absolute()
        self._store = store
        self.nb_requests = 0
        self.total_filesize = 0
        if not self.dir.exists():
            self.dir.mkdir(parents=True, exist_ok=True)
        print("AMS Store directories created")

    @AMSMonitor(record=["nb_requests"])
    def __call__(self):
        """
        A busy loop reading messages from the i_queue publishing them to the kosh store.
        """
        print("In busy loop of AMS Store")
        start = time.time()
        if self._store:
            db_store = AMSDataStore(
                self.ams_config.db_path, self.ams_config.db_store, self.ams_config.name, False
            ).open()

        with AMSMonitor(obj=self, tag="internal_loop", record=[]):
            while True:
                item = self.i_queue.get(block=True)
                if item.is_terminate():
                    break
                elif item.is_process():
                    with AMSMonitor(obj=self, tag="request_block", record=["nb_requests", "total_filesize"]):
                        self.nb_requests += 1
                        src_fn = Path(item.data())
                        dest_file = self.dir / src_fn.name
                        if src_fn != dest_file:
                            shutil.move(src_fn, dest_file)
                        # TODO: Fix me candidates now will be "indexed by the name"
                        warnings.warn("AMS Kosh manager does not operate with multi-models")
                        if self._store:
                            db_store.add_candidates([str(dest_file)])

                        self.total_filesize += os.stat(src_fn).st_size

        end = time.time()
        print(f"Spend {end - start} at {self.__class__.__name__}")


class Pipeline(ABC):
    """
    An interface class representing a sequence of transformations/actions to be performed
    to store data in the AMS kosh-store. The actions can be performed either sequentially,
    or in parallel using different poclies/vehicles (threads or processes).

    Attributes:
        ams_config: The AMS configuration required when publishing to the AMS store.
        dest_dir: The final path to store data to.
        stage_dir: An intermediate location to store files. Usefull if the configuration requires
            storing the data in some scratch directory (SSD) before making them public to the parallel filesystem.
        actions: A list of actions to be performed before storing the data in the filesystem
        db_type: The file format of the data to be stored
        writer: The class to be used to write data to the filesystem.
    """

    supported_policies = {"sequential", "thread", "process"}
    supported_writers = {"shdf5", "dhdf5", "csv"}

    def __init__(self, db_dir, store, dest_dir=None, stage_dir=None, db_type="hdf5"):
        """
        initializes the Pipeline class to write the final data in the 'dest_dir' using a file writer of type 'db_type'
        and optionally caching the data in the 'stage_dir' before making them available in the cache store.
        """
        self.ams_config = AMSInstance.from_env()
        if self.ams_config is None:
            self.ams_config = AMSInstance.from_path(db_dir)

        if dest_dir is not None:
            self.dest_dir = dest_dir

        if dest_dir is None and store:
            self.dest_dir = self.ams_config.db_path

        self.stage_dir = self.dest_dir

        if stage_dir is not None:
            self.stage_dir = stage_dir

        self.actions = list()

        self.db_type = db_type

        self._writer = get_writer(self.db_type)

        self.store = store

    def add_data_action(self, callback):
        """
        Adds an action to be performed at the data before storing them in the filesystem

        Args:
            callback: A callback to be called on every input, output.
        """
        if not callable(callback):
            raise TypeError(f"{self.__class__.__name__} requires a callable as an argument")

        self.actions.append(callback)

    def _seq_execute(self):
        """
        Executes all tasks sequentially. Every task starts after all incoming messages
        are processed by the previous task.
        """
        for t in self._tasks:
            t()

    def _parallel_execute(self, exec_vehicle_cls):
        """
        parallel execute of all tasks using the specified vehicle type

        Args:
            exec_vehicle_cls: The class to be used to generate entities
            executing actions by reading data from i/o_queue(s).
        """
        executors = list()
        for i, a in enumerate(self._tasks):
            print(f"Instantiating Task: {i}:{a.__class__.__name__}")
            executors.append(exec_vehicle_cls(target=a))

        for i, e in enumerate(executors):
            print(f"Start Executor: {i}")
            e.start()
            print(f"Executor {i} started")

        # This is currently is hardcoded. We need to abstract it
        # to make this callable from the command line.
        # Addtionally a good communication protocol with KVS of
        # AMS-deploy can help us to gracefull shutdown connections
        flux_handle = flux.Flux()
        kvs_dir = KVSDir(flux_handle=flux_handle)
        print("Instance level is ", flux_handle.attr_get("instance-level"))
        print(f"Flux URI is {os.environ.get('FLUX_URI')}")
        sys.stdout.flush()
        print("KVS RESOURCE DIR IS", kvs_dir["resource.R"])
        sys.stdout.flush()
        kvs_dir["AMSStager"] = "Active"
        kvs_dir.commit()
        print("Commit to KVS Store")
        sys.stdout.flush()

        for e in executors:
            e.join()

    def _execute_tasks(self, policy):
        """
        Executes all tasks using the specified policy

        Args:
            policy: The policy to be used to execute the pipeline
        """
        executors = {"thread": Thread, "process": Process}

        if policy == "sequential":
            self._seq_execute()
            return

        self._parallel_execute(executors[policy])
        return

    def _link_pipeline(self, policy):
        """
        Links all actions/stages of the pipeline with input/output queues.

        Args:
            policy: The policy to be used to execute the pipeline
        """
        _qType = self.get_q_type(policy)
        # We need 1 queue to copy incoming data to the pipeline
        # Every action requires 1 input and one output q. But the output
        # q is used as an inut q on the next action thus we need num actions -1.
        # 2 extra queues to store to data-store and publish on kosh
        num_queues = 1 + len(self.actions) - 1 + 2
        self._queues = [_qType() for i in range(num_queues)]

        self._tasks = [self.get_load_task(self._queues[0], policy)]
        for i, a in enumerate(self.actions):
            self._tasks.append(ForwardTask(self._queues[i], self._queues[i + 1], a))

        # After user actions we store into a file
        self._tasks.append(FSWriteTask(self._queues[-2], self._queues[-1], self._writer, self.stage_dir))
        # After storing the file we make it public to the kosh store.
        self._tasks.append(PushToStore(self._queues[-1], self.ams_config, self.dest_dir, self.store))

    def execute(self, policy):
        """
        Execute the pipeline of tasks using the specified policy (blocking).

        Args:
            policy: The policy to be used to execute the pipeline
        """
        if policy not in self.__class__.supported_policies:
            raise RuntimeError(
                f"Pipeline execute does not support policy: {policy}, please select from  {Pipeline.supported_policies}"
            )

        # Create a pipeline of actions and link them with appropriate queues
        self._link_pipeline(policy)
        # Execute them
        self._execute_tasks(policy)

    @abstractmethod
    def get_load_task(self, o_queue, policy):
        """
        Callback to the child class to return the task that loads data from some unspecified entry-point.
        """
        pass

    @staticmethod
    @abstractmethod
    def add_cli_args(parser):
        """
        Initialize root pipeline class cli parser with the options.
        """
        parser.add_argument("--dest", "-d", dest="dest_dir", help="Where to store the data (Directory should exist)")
        parser.add_argument(
            "--stage-dir",
            dest="stage_dir",
            help="Where to 'stage' data (some directory either under /dev/shm/ or under local storage (SSD)",
            default=None,
        )
        parser.add_argument(
            "--db-type",
            dest="db_type",
            choices=Pipeline.supported_writers,
            help="File format to store the data to",
            default="dhdf5",
        )
        # parser.add_argument("--db-dir", "-d", help="path to the AMS store directory", required=True)
        parser.add_argument("--persistent-db-path", "-db", help="The path of the AMS database", required=True)
        parser.add_argument("--store", dest="store", action="store_true")
        parser.add_argument("--no-store", dest="store", action="store_false")
        parser.set_defaults(store=True)
        return

    @abstractclassmethod
    def from_cli(cls):
        pass

    @staticmethod
    def get_q_type(policy):
        """
        Returns the type of the queue to be used to create Queues for the specified policy.
        """

        p_to_type = {"sequential": ser_queue, "thread": ser_queue, "process": mp_queue}
        return p_to_type[policy]


class FSPipeline(Pipeline):
    """
    A 'Pipeline' reading data from the Filesystem and storing them back to the filesystem.

    Attributes:
        src: The source directory to read data from.
        pattern: The pattern to glob files from.
        src_type: The file format of the source data
    """

    supported_readers = ("shdf5", "dhdf5", "csv")

    def __init__(self, db_dir, store, dest_dir, stage_dir, db_type, src, src_type, pattern):
        """
        Initialize a FSPipeline that will write data to the 'dest_dir' and optionally publish
        these files to the kosh-store 'store' by using the stage_dir as an intermediate directory.
        """
        super().__init__(db_dir, store, dest_dir, stage_dir, db_type)
        self._src = Path(src)
        self._pattern = pattern
        self._src_type = src_type

    def get_load_task(self, o_queue, policy):
        """
        Return a Task that loads data from the filesystem

        Args:
            o_queue: The queue the load task will push read data.

        Returns: An FSLoaderTask instance reading data from the filesystem and forwarding the values to the o_queue.
        """
        loader = get_reader(self._src_type)
        return FSLoaderTask(o_queue, loader, pattern=str(self._src) + "/*" + self._pattern)

    @staticmethod
    def add_cli_args(parser):
        """
        Add cli arguments to the parser required by this Pipeline.
        """
        Pipeline.add_cli_args(parser)
        parser.add_argument("--src", "-s", help="Where to copy the data from", required=True)
        parser.add_argument("--src-type", "-st", choices=FSPipeline.supported_readers, default="shdf5")
        parser.add_argument("--pattern", "-p", help="Glob pattern to read data from", required=True)
        return

    @classmethod
    def from_cli(cls, args):
        """
        Create FSPipeline from the user provided CLI.
        """
        return cls(
            args.persistent_db_path,
            args.store,
            args.dest_dir,
            args.stage_dir,
            args.db_type,
            args.src,
            args.src_type,
            args.pattern,
        )


class RMQPipeline(Pipeline):
    """
    A 'Pipeline' reading data from RabbitMQ and storing them back to the filesystem.

    Attributes:
        host: RabbitMQ host
        port: RabbitMQ port
        vhost: RabbitMQ virtual host
        user: RabbitMQ username
        password: RabbitMQ password for username
        cert: The TLS certificate
        rmq_queue: The RMQ queue to listen to.
    """

    def __init__(self, db_dir, store, dest_dir, stage_dir, db_type, host, port, vhost, user, password, cert, rmq_queue):
        """
        Initialize a RMQPipeline that will write data to the 'dest_dir' and optionally publish
        these files to the kosh-store 'store' by using the stage_dir as an intermediate directory.
        """
        super().__init__(db_dir, store, dest_dir, stage_dir, db_type)
        self._host = host
        self._port = port
        self._vhost = vhost
        self._user = user
        self._password = password
        self._cert = Path(cert)
        self._rmq_queue = rmq_queue

    def get_load_task(self, o_queue, policy):
        """
        Return a Task that loads data from the filesystem

        Args:
            o_queue: The queue the load task will push read data.

        Returns: An RMQLoaderTask instance reading data from the
        filesystem and forwarding the values to the o_queue.
        """
        return RMQLoaderTask(
            o_queue,
            self._host,
            self._port,
            self._vhost,
            self._user,
            self._password,
            self._cert,
            self._rmq_queue,
            policy,
        )

    @staticmethod
    def add_cli_args(parser):
        """
        Add cli arguments to the parser required by this Pipeline.
        """
        Pipeline.add_cli_args(parser)
        parser.add_argument("-c", "--creds", help="Credentials file (JSON)", required=True)
        parser.add_argument("-t", "--cert", help="TLS certificate file", required=True)
        parser.add_argument("-q", "--queue", help="On which queue to receive messages", required=True)
        return

    @classmethod
    def from_cli(cls, args):
        """
        Create RMQPipeline from the user provided CLI.
        """

        # TODO: implement an interface so users can plug any parser for RMQ credentials
        config = cls.parse_credentials(cls, args.creds)
        host = config["service-host"]
        port = config["service-port"]
        vhost = config["rabbitmq-vhost"]
        user = config["rabbitmq-user"]
        password = config["rabbitmq-password"]

        return cls(
            args.persistent_db_path,
            args.store,
            args.dest_dir,
            args.stage_dir,
            args.db_type,
            host,
            port,
            vhost,
            user,
            password,
            args.cert,
            args.queue,
        )

    @staticmethod
    def parse_credentials(self, json_file: str) -> dict:
        """Internal method to parse the credentials file"""
        data = {}
        with open(json_file, "r") as f:
            data = json.load(f)
        return data


def get_pipeline(src_mechanism="fs"):
    """
    Factory method to return the pipeline mechanism for the given source entry point

    Args:
        src_mechanism: The entry mechanism to read data from.

    Returns: A Pipeline class to start the stage AMS service
    """
    pipe_mechanisms = {"fs": FSPipeline, "network": RMQPipeline}
    if src_mechanism not in pipe_mechanisms.keys():
        raise RuntimeError(f"Pipeline {src_mechanism} storing mechanism does not exist")

    return pipe_mechanisms[src_mechanism]
