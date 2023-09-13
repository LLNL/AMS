from abc import ABC, abstractmethod
import time
import numpy as np
from enum import Enum
import glob
from ams.database import get_reader
from ams.database import get_writer
from ams.config import AMSInstance
from ams.store import AMSDataStore
import uuid
import socket
import datetime
from typing import Callable
from multiprocessing import Process
from multiprocessing import Queue as mp_queue
from queue import Queue as ser_queue
from threading import Thread
from threading import get_native_id
from threading import get_ident
from pathlib import Path
import shutil

class MessageType(Enum):
    Process = 1
    NewModel = 2
    Terminate = 3

class DataBlob:
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

class QueueMessage:
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

class DataBlobAction(ABC):
    def __init__(self):
        self.data_actions_processed = 0
        self.total_input_memory = 0
        self.total_output_memory = 0

    @abstractmethod
    def __call__(self, inputs, outputs):
        pass

    def stats(self):
        return f'Processed: {self.data_blobs_processed} blobs, Input Memory: {self.total_input_memory}, Output Memory: {self.total_output_memory}'

class Task(ABC):
    @abstractmethod
    def __call__(self):
        pass

class PipeTask(Task):
    def __init__(self, i_queue, o_queue, callback):
        if not isinstance(callback, Callable):
            raise TypeError(f'{callback} argument is not Callable')

        self.i_queue = i_queue
        self.o_queue = o_queue
        self.callback = callback

    def _action(self, data):
        inputs, outputs = self.callback(data.inputs, data.outputs)
        # This can be too conservative, we may want to relax it later
        if not ( isinstance(inputs, np.ndarray) and isinstance(outputs, np.ndarray) ):
            raise TypeError(f'{self.callback.__name__} did not return numpy arrays')
        return inputs, outputs

    def __call__(self):
        start = time.time()
        while True:
            # This is a blocking call
            item = self.i_queue.get(block=True)
            if item.is_terminate():
                self.o_queue.put(QueueMessage(MessageType.Terminate, None))
                break
            elif item.is_process():
                inputs, outputs = self._action(item.data())
                self.o_queue.put(QueueMessage(MessageType.Process, DataBlob(inputs, outputs)))
            elif item.is_new_model():
                # This is not handled yet
                continue
        end = time.time()
        print(f'Spend {end - start} at {self.callback}')
        return

class FSLoaderTask(Task):
    def __init__(self, o_queue, loader,  pattern):
        self.o_queue = o_queue
        self.pattern = pattern
        self.loader = loader

    def __call__(self):
        start = time.time()
        for fn in glob.glob(self.pattern):
            with self.loader(fn) as fd:
                input_data, output_data = fd.load()
                input_batches =  np.split(input_data,100)
                output_batches =  np.split(output_data,100)
                for i, o in zip(input_batches, output_batches):
                    self.o_queue.put(QueueMessage(MessageType.Process, DataBlob(i, o)))
        self.o_queue.put(QueueMessage(MessageType.Terminate, None))

        end = time.time()
        print(f'Spend {end - start} at {self.__class__.__name__}')


class FSWriteTask(Task):
    def __init__(self, i_queue, o_queue, writer_cls, out_dir):
        self.data_writer_cls = writer_cls
        self.out_dir = out_dir
        self.i_queue = i_queue
        self.o_queue = o_queue
        self.suffix = writer_cls.get_file_format_suffix()

    def __call__(self):
        start = time.time()
        while True:
            fn = [ uuid.uuid4().hex, socket.gethostname(),
                  str(datetime.datetime.now()).replace('-', 'D').replace(' ', 'T').replace(':','C').replace('.','p')]
            fn = '_'.join(fn)
            fn = f'{self.out_dir}/{fn}.{self.suffix}'
            is_terminate = False
            with self.data_writer_cls(fn) as fd:
                bytes_written = 0
                while True:
                    # This is a blocking call
                    item = self.i_queue.get(block=True)
                    if item.is_terminate():
                        is_terminate = True
                    elif item.is_process():
                        data = item.data()
                        bytes_written += data.inputs.size * data.inputs.itemsize
                        bytes_written += data.outputs.size * data.outputs.itemsize
                        fd.store(data.inputs, data.outputs)

                    if is_terminate or bytes_written >= 2*1024*1024*1024:
                        break

            self.o_queue.put(QueueMessage(MessageType.Process, fn))
            if is_terminate:
                self.o_queue.put(QueueMessage(MessageType.Terminate, None))
                break

        end = time.time()
        print(f'Spend {end - start} at {self.__class__.__name__}')

class PushToStore(Task):
    def __init__(self, i_queue, ams_config, db_path):
        self.ams_config = ams_config
        self.i_queue = i_queue
        self.dir = Path(db_path).absolute()
        if not self.dir.exists():
            self.dir.mkdir(parents=True, exist_ok=True)

    def __call__(self):
        start = time.time()
        with AMSDataStore(self.ams_config.db_path, self.ams_config.db_store, self.ams_config.name, False) as db_store:
            while True:
                item = self.i_queue.get(block=True)
                if item.is_terminate():
                    break
                elif item.is_process():
                    src_fn = Path(item.data())
                    dest_file = self.dir / src_fn.name
                    # requires python 3.9
                    print(src_fn, dest_file)
                    if src_fn != dest_file:
                        shutil.move(src_fn, dest_file)
                    db_store.add_candidates([str(dest_file)])

        end = time.time()
        print(f'Spend {end - start} at {self.__class__.__name__}')

class Pipeline(ABC):
    supported_policies = { 'sequential', 'thread', 'process' }

    @staticmethod
    def get_q_type(policy):
        p_to_type = {'sequential' : ser_queue,
                     'thread'     : ser_queue,
                     'process'    : mp_queue
                     }
        return p_to_type[policy]

    def __init__(self, ams_config):
        self.ams_config = ams_config
        self.db_path = Path(self.ams_config.db_path)/Path('candidates')
        self.actions = list()
        self.stage_dir = self.db_path
        # Simplest case we need a single Q
        self._writer = get_writer(self.ams_config.db_type)

    @abstractmethod
    def get_load_task(self, o_queue):
        pass

    def add_data_action(self, callback):
        if not callable(callback):
            raise TypeError(f"{self.__class__.__name__} requires a callable as an argument")

        self.actions.append(callback)

    def enable_stage(self, stage_dir):
        self.stage_dir = Path(stage_dir)

    def _seq_execute(self):
        for t in self._tasks:
            t()

    def _parallel_execute(self, exec_vehicle_cls):
        executors = list()
        for a in self._tasks:
            executors.append(exec_vehicle_cls(target = a))

        for e in executors:
            e.start()

        for e in executors:
            e.join()

    def _execute_tasks(self, policy):
        executors = { 'thread' : Thread,
                     'process' : Process
                    }

        if policy == 'sequential':
            self._seq_execute()
            return

        self._parallel_execute(executors[policy])
        return

    def _link_pipeline(self, policy):
        _qType = self.get_q_type(policy)
        # We need 1 queue to copy incoming data to the pipeline
        # Every action requires 1 input and one output q. But the output
        # q is used as an inut q on the next action thus we need num actions -1.
        # 2 extra queues to store to data-store and publish on kosh
        num_queues = 1 + len(self.actions) - 1 + 2
        self._queues = [ _qType() for i in range(num_queues) ]

        self._tasks = [self.get_load_task(self._queues[0])]
        for i,a in enumerate(self.actions):
            self._tasks.append(PipeTask(self._queues[i], self._queues[i+1], a))

        # After user actions we store into a file
        self._tasks.append(FSWriteTask(self._queues[-2], self._queues[-1], self._writer, self.stage_dir))
        # After storing the file we make it public to the kosh store.
        self._tasks.append(PushToStore(self._queues[-1], self.ams_config, self.db_path))

    def execute(self, policy):
        if policy not in self.__class__.supported_policies:
            raise RuntimeError(f"Pipeline execute does not support policy: {policy}, please select from  {Pipeline.supported_policies}")

        # Create a pipeline of actions and link them with appropriate queues
        self._link_pipeline(policy)
        # Execute them
        self._execute_tasks(policy)

class FSPipeline(Pipeline):
    def __init__(self, src, pattern):
        ams_instance = AMSInstance()
        super().__init__(ams_instance)
        self._src = Path(src)
        self._pattern = pattern

    def get_load_task(self, o_queue):
        loader = get_reader(self.ams_config.db_type)
        return FSLoaderTask(o_queue, loader, pattern=str(self._src) + "/" + self._pattern)

def get_pipeline(src_mechanism='fs'):
    PipeMechanisms = { 'fs' : FSPipeline,
                     'network' : None }
    if src_mechanism not in PipeMechanisms.keys():
        raise RuntimeError(f"Pipeline {src_mechanism} storing mechanism does not exist")

    return PipeMechanisms[src_mechanism]
