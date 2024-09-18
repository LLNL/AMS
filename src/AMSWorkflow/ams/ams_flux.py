# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# AMSLib Project Developers
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import threading
import logging
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import concurrent
import collections
import subprocess
import types
from typing import Union


from flux.job.event import MAIN_EVENTS
from flux.job import FluxExecutor
import flux
from queue import Queue


# pylint: disable=too-many-instance-attributes
class AMSFluxExecutorFuture(Future):
    """A ``concurrent.futures.Future`` subclass that represents a AMSTrain job.

    The class provides a future abstraction for training jobs. In 'AMS' training jobs
    are actually a sequence of 2 jobs. First we do the sub-selection and right after
    we schedule a ml model training job. The future provides accessor to the description
    of the current AMS training job and the phase in which the job is part off.

    In addition to all of the ``flux.job.FluxExecutorFuture`` functionality,
    ``AMSFluxExecutorFuture`` instances offer:

    * The ``uri`` and ``add_uri_callback`` methods for retrieving the
      Flux uri of the instance executing this job. This is convenient for
      nested jobs

    Valid events are contained in the ``EVENTS`` class attribute.
    """

    # NOTE: This is the primary difference of the original FluxExecutorFuture.
    # FluxExecutorFuture uses frozensets without adding "memo" and requires all registered
    # callbacks  to be part of the EVENTS.
    EVENTS = frozenset(("memo", *list(MAIN_EVENTS)))

    @staticmethod
    def __get_uri_cb(fut, eventlog):
        if "uri" not in eventlog.context:
            fut._set_uri(None, KeyError(f"uri does not exist in memo callback {eventlog}"))
        fut._set_uri(eventlog.context["uri"])

    def __init__(
        self,
        owning_thread_id: int,
        flux_executor,
        track_uri: bool,
        domain_descr,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        # Thread.ident of thread tasked with completing this future
        self.__owning_thread_id = owning_thread_id
        self.__jobid_condition = threading.Condition()
        self.__jobid = None
        self.__jobid_set = False  # True if the jobid has been set to something
        self.__jobid_exception = None
        self.__jobid_callbacks = []
        # NOTE: These are the 'additional variables of the Future'
        self.__uri_condition = threading.Condition()
        self.__uri = None
        self.__uri_set = False
        self.__uri_exception = None
        self.__uri_callbacks = []
        self.__flux_executor = flux_executor
        self.__domain_descr = domain_descr

        self.__event_lock = threading.RLock()
        self.__events_occurred = {state: collections.deque() for state in self.EVENTS}
        self.__event_callbacks = {state: collections.deque() for state in self.EVENTS}
        if track_uri:
            self.add_event_callback("memo", self.__get_uri_cb)

    def get_domain_descr(self):
        return self.__domain_descr

    def _set_uri(self, uri, exc=None):
        """Sets the Flux uri associated with the future.

        If `exc` is not None, raise `exc` instead of returning the jobid
        in calls to `Future.jobid()`. Useful if the job ID cannot be
        retrieved.

        Should only be used by Executor implementations and unit tests.
        """
        if self.__uri:
            raise RuntimeError("invalid state: uri already set")

        with self.__uri_condition:
            self.__uri = uri
            self.__uri_set = True
            if exc is not None:
                self.__uri_exception = exc
            self.__uri_condition.notify_all()
        for callback in self.__uri_callbacks:
            self._invoke_flux_callback(callback)

    def _set_jobid(self, jobid, exc=None):
        """Sets the Flux jobid associated with the future.

        If `exc` is not None, raise `exc` instead of returning the jobid
        in calls to `Future.jobid()`. Useful if the job ID cannot be
        retrieved.

        Should only be used by Executor implementations and unit tests.
        """
        if self.__jobid_set:
            # should be InvalidStateError in 3.8+
            raise RuntimeError("invalid state: jobid already set")
        with self.__jobid_condition:
            self.__jobid = jobid
            self.__jobid_set = True
            if exc is not None:
                self.__jobid_exception = exc
            self.__jobid_condition.notify_all()
        for callback in self.__jobid_callbacks:
            self._invoke_flux_callback(callback)

    def add_done_callback(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """Attaches a callable that will be called when the future finishes.

        :param fn: A callable that will be called with this future as its only
            argument when the future completes or is cancelled. The callable
            will always be called by a thread in the same process in which
            it was added. If the future has already completed or been
            cancelled then the callable will be called immediately. These
            callables are called in the order that they were added.
        :return: ``self``
        """
        super().add_done_callback(*args, **kwargs)
        return self

    def uri(self, timeout=None):
        """Return the uri of the Flux Instance that the future represents.

        :param timeout: The number of seconds to wait for the jobid.
            If None, then there is no limit on the wait time.

        :return: a flux uri.

        :raises concurrent.futures.TimeoutError: If the jobid is not available
            before the given timeout.
        :raises concurrent.futures.CancelledError: If the future was cancelled.
        :raises RuntimeError: If the job could not be submitted (e.g. if
            the jobspec was invalid).
        """
        if self.__uri_set:
            return self._get_uri()
        with self.__uri_condition:
            self.__uri_condition.wait(timeout)
            if self.__uri_set:
                return self._get_uri()
            raise concurrent.futures.TimeoutError()

    def _get_uri(self):
        """Get the jobid, checking for cancellation and invalid job ids."""
        if self.__uri_exception is not None:
            raise self.__uri_exception
        return self.__uri

    def jobid(self, timeout=None):
        """Return the jobid of the Flux job that the future represents.

        :param timeout: The number of seconds to wait for the jobid.
            If None, then there is no limit on the wait time.

        :return: a positive integer jobid.

        :raises concurrent.futures.TimeoutError: If the jobid is not available
            before the given timeout.
        :raises concurrent.futures.CancelledError: If the future was cancelled.
        :raises RuntimeError: If the job could not be submitted (e.g. if
            the jobspec was invalid).
        """
        if self.__jobid_set:
            return self._get_jobid()
        with self.__jobid_condition:
            self.__jobid_condition.wait(timeout)
            if self.__jobid_set:
                return self._get_jobid()
            raise concurrent.futures.TimeoutError()

    def _get_jobid(self):
        """Get the jobid, checking for cancellation and invalid job ids."""
        if self.__jobid_exception is not None:
            raise self.__jobid_exception
        return self.__jobid

    def flux_executor(self):
        return self.__flux_executor

    def add_uri_callback(self, callback):
        """Attaches a callable that will be called when the uri is ready.

        Added callables are called in the order that they were added and may be called
        in another thread.  If the callable raises an ``Exception`` subclass, it will
        be logged and ignored.  If the callable raises a ``BaseException`` subclass,
        the behavior is undefined.

        :param callback: a callable taking the future as its only argument.
        :return: ``self``
        """
        with self.__uri_condition:
            if self.__uri is None:
                self.__uri_callbacks.append(callback)
                return self
        self._invoke_flux_callback(callback)
        return self

    def add_jobid_callback(self, callback):
        """Attaches a callable that will be called when the jobid is ready.

        Added callables are called in the order that they were added and may be called
        in another thread.  If the callable raises an ``Exception`` subclass, it will
        be logged and ignored.  If the callable raises a ``BaseException`` subclass,
        the behavior is undefined.

        :param callback: a callable taking the future as its only argument.
        :return: ``self``
        """
        with self.__jobid_condition:
            if self.__jobid is None:
                self.__jobid_callbacks.append(callback)
                return self
        self._invoke_flux_callback(callback)
        return self

    def _invoke_flux_callback(self, callback, *args):
        try:
            callback(self, *args)
        except Exception:  # pylint: disable=broad-except
            logging.getLogger(__name__).exception("exception calling callback for %r", self)

    def exception(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """If this method is invoked from a jobid/event callback by an executor thread,
        it will result in deadlock, since the current thread will wait
        for work that the same thread is meant to do.

        Head off this possibility by checking the current thread.
        """
        if not self.done() and threading.get_ident() == self.__owning_thread_id:
            raise RuntimeError("Cannot wait for future from inside callback")
        return super().exception(*args, **kwargs)

    def result(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """If this method is invoked from a jobid/event callback by an executor thread,
        it will result in deadlock, since the current thread will wait
        for work that the same thread is meant to do.

        Head off this possibility by checking the current thread.
        """
        if not self.done() and threading.get_ident() == self.__owning_thread_id:
            raise RuntimeError("Cannot wait for future from inside callback")
        return super().result(*args, **kwargs)

    def set_exception(self, exception):
        """When setting an exception on the future, set the jobid if it hasn't
        been set already. The jobid will already have been set unless the exception
        was generated before the job could be successfully submitted.
        """
        try:
            self.jobid(0)
        except concurrent.futures.TimeoutError:
            # set jobid to something
            self._set_jobid(None, RuntimeError(f"job could not be submitted due to {exception}"))
            self._set_uri(None, RuntimeError(f"job could not be submitted due to {exception}"))
        return super().set_exception(exception)

    def cancel(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """If a thread is waiting for the future's jobid, and another
        thread cancels the future, the waiting thread would never wake up
        because the jobid would never be set.

        When cancelling, set the jobid to something invalid.
        """
        if self.cancelled():  # if already cancelled, return True
            return True
        cancelled = super().cancel(*args, **kwargs)
        if cancelled:
            try:
                self.jobid(0)
            except concurrent.futures.TimeoutError:
                # set jobid to something
                self._set_jobid(None, concurrent.futures.CancelledError())
                self._set_uri(None, concurrent.futures.CancelledError())
        return cancelled

    def add_event_callback(self, event, callback):
        """Add a callback to be invoked when an event occurs.

        The callback will be invoked, with the future as the first argument and the
        ``flux.job.EventLogEvent`` as the second, whenever the event occurs. If the
        event occurs multiple times, the callback will be invoked with each different
        `EventLogEvent` instance. If the event never occurs, the callback
        will never be invoked.

        Added callables are called in the order that they were added and may be called
        in another thread.  If the callable raises an ``Exception`` subclass, it will
        be logged and ignored.  If the callable raises a ``BaseException`` subclass,
        the behavior is undefined.

        If the event has already occurred, the callback will be called immediately.

        :param event: the name of the event to add the callback to.
        :param callback: a callable taking the future and the event as arguments.
        :return: ``self``
        """
        if event not in self.EVENTS:
            raise ValueError(event)
        with self.__event_lock:
            self.__event_callbacks[event].append(callback)
            for log_entry in self.__events_occurred[event]:
                self._invoke_flux_callback(callback, log_entry)
        return self

    def _set_event(self, log_entry):
        """Set an event on the future.

        For use by Executor implementations and unit tests.

        :param log_entry: an ``EventLogEvent``.
        """
        event_name = log_entry.name
        if event_name not in self.EVENTS:
            raise ValueError(event_name)
        with self.__event_lock:
            self.__events_occurred[event_name].append(log_entry)
            # make a shallow copy of callbacks --- in case a user callback
            # tries to add another callback for the same event
            for callback in list(self.__event_callbacks[event_name]):
                self._invoke_flux_callback(callback, log_entry)

    def __repr__(self):
        return f"<{self.__class__.__name__} at {hex(id(self))}>"


class AMSFluxExecutor(FluxExecutor):
    """
    A Flux executor that submits jobs to the provided uri. The executor extends
    the original FluxExecutor as it generates "AMSFluxExecutorFuture" 's that can 
    track the uri of nested jobs. This is required when 'partitioning' an allocation
    to sub allocations and issuing 'nested' commands
    """
    def __init__(self, track_uri, *args, **kwargs):
        self._track_uri = track_uri
        super().__init__(*args, **kwargs)

    def _create_future(self, factory, *factory_args):
        if self._broken_event.is_set():
            raise RuntimeError("Executor is broken, new futures cannot be scheduled")
        with self._shutdown_lock:
            if self._shutdown_event.is_set():
                raise RuntimeError("cannot schedule new futures after shutdown")
            future_owner_id = self._executor_threads[self._next_thread].ident
            fut = AMSFluxExecutorFuture(future_owner_id, self, self._track_uri, None)
            self._submission_queues[self._next_thread].append(factory(*factory_args, fut))
            self._next_thread = (self._next_thread + 1) % len(self._submission_queues)
            return fut

    @property
    def shutdown_event(self):
        """The _shutdown_event property."""
        return self._shutdown_event


class _WorkItem:
    """
    A Workitem represents a scheduled jobs that will now be executed. The purpose of the class
    is to provide our own "exeucution" vehicle when flux is not present. This is to be used only for
    debugging purposes in composition with the AMSFakeFluxOrchestatorExecutor.
    """

    def __init__(self, future, spec):
        self.future = future
        self.spec = spec

    def run(self):
        if not self.future.set_running_or_notify_cancel():
            return

        try:
            out = None
            err = None
            if self.spec.stdout is not None:
                out = open(self.spec.stdout, "w")
            if self.spec.stderr == self.spec.stdout:
                err = out
            elif self.spec.stderr is not None:
                err = open(self.spec.stderr, "w")
            print(self.spec.tasks[0])
            print(self.spec.stdout)
            print(self.spec)
            proc = subprocess.run(
                " ".join(self.spec.tasks[0]["command"]),
                stdout=out,
                stderr=err,
                env=self.spec.environment,
                cwd=self.spec.cwd,
                shell=True,
            )
            result = proc.returncode
        except BaseException as exc:
            self.future.set_exception(exc)
            # Break a reference cycle with the exception 'exc'
            self = None
        else:
            self.future.set_result(result)
            if self.spec.stdout is not None:
                out.close()
            if self.spec.stderr != self.spec.stdout:
                err = out
            elif self.spec.stderr is not None:
                err.close()

    __class_getitem__ = classmethod(types.GenericAlias)


class AMSFakeFluxOrchestatorExecutor(ThreadPoolExecutor):
    """
    A class to emulate a FluxExecutor with subprocesses. For every job being submitted instead
    of running it with flux job submission we use subprocess to run a single instantiation of the
    job.
    """

    def __init__(self, o_queue: Queue, domains, *args, **kwargs):
        self._o_queue = o_queue
        self._domains = domains
        super().__init__(*args, **kwargs)

    def submit(self, domain, job_spec):
        """Submit a jobspec to Flux and return a ``FluxExecutorFuture``.

        Accepts the same positional and keyword arguments as ``flux.job.submit``,
        except for the ``flux.job.submit`` function's first argument, ``flux_handle``.
        :param domain: AMSDomain training job description
        :param jobspec: jobspec defining the job request
        :type jobspec: Jobspec or its string encoding
        :raises RuntimeError: if ``shutdown`` has been called or if an error has
            occurred and new jobs cannot be submitted (e.g. a remote Flux instance
            can no longer be communicated with).
        """
        with self._shutdown_lock:
            if self._broken:
                raise RuntimeError(f"Thread is broken {self._broken}")

            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            f = AMSFluxExecutorFuture(0, self, "", domain)
            w = _WorkItem(f, job_spec)
            print(w)

            self._work_queue.put(w)
            self._adjust_thread_count()
            return f

    def get_o_queue(self):
        return self._o_queue

    def get_domains(self):
        return self._domains


class AMSFluxOrchestratorExecutor(AMSFluxExecutor):
    def __init__(self, o_queue, domains, *args, **kwargs):
        self._o_queue = o_queue
        self._domains = domains
        super().__init__(False, *args, **kwargs)

    def submit(self, domain, *args, **kwargs):
        """Submit a jobspec to Flux and return a ``FluxExecutorFuture``.

        Accepts the same positional and keyword arguments as ``flux.job.submit``,
        except for the ``flux.job.submit`` function's first argument, ``flux_handle`` and
        instead accepts the ``domain`` argument pointing to the domain currently being scheduled
        by AMS.
        :param domain: AMSDomain training job description
        :param jobspec: jobspec defining the job request
        :type jobspec: Jobspec or its string encoding
        :param urgency: job urgency 0 (lowest) through 31 (highest)
            (default is 16).  Priorities 0 through 15 are restricted to
            the instance owner.
        :type urgency: int
        :param waitable: allow result to be fetched with ``flux.job.wait()``
            (default is False).  Waitable=True is restricted to the
            instance owner.
        :type waitable: bool
        :param debug: enable job manager debugging events to job eventlog
            (default is False)
        :type debug: bool
        :param pre_signed: jobspec argument is already signed
            (default is False)
        :type pre_signed: bool

        :raises RuntimeError: if ``shutdown`` has been called or if an error has
            occurred and new jobs cannot be submitted (e.g. a remote Flux instance
            can no longer be communicated with).
        """
        return self._create_future(domain, flux.job.executor._SubmitPackage, args, kwargs)

    def _create_future(self, domain, factory, *factory_args):
        if self._broken_event.is_set():
            raise RuntimeError("Executor is broken, new futures cannot be scheduled")
        with self._shutdown_lock:
            if self._shutdown_event.is_set():
                raise RuntimeError("cannot schedule new futures after shutdown")
            future_owner_id = self._executor_threads[self._next_thread].ident
            fut = AMSFluxExecutorFuture(future_owner_id, self, self._track_uri, domain)
            self._submission_queues[self._next_thread].append(factory(*factory_args, fut))
            self._next_thread = (self._next_thread + 1) % len(self._submission_queues)
            return fut

    def get_o_queue(self):
        return self._o_queue

    @property
    def domains(self):
        return self._domains
