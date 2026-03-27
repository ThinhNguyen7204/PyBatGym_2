"""Real BatSim adapter connecting PyBatGym to the actual C++ BatSim simulator via pybatsim."""

import json
import queue
import subprocess
import threading
import time
from typing import Optional

from pybatgym.batsim_adapter import BatsimAdapter
from pybatgym.models import Job, JobStatus, Resource, Event, EventType, ScheduleCommand, ScheduleCommandType

try:
    from batsim.batsim import BatsimScheduler, Batsim
except ImportError:
    BatsimScheduler = object
    def Batsim(*args, **kwargs):
        raise ImportError("pybatsim is not installed. Please install it to use RealBatsimAdapter.")


class RealBatsimAdapter(BatsimAdapter, BatsimScheduler):
    """Integrates real BatSim simulator using pybatsim over ZeroMQ."""
    
    def __init__(self, config, socket_endpoint: str = "tcp://*:28000"):
        BatsimAdapter.__init__(self, config)
        BatsimScheduler.__init__(self)
        
        self.socket_endpoint = socket_endpoint
        self._action_queue = queue.Queue()
        self._state_queue = queue.Queue()
        self._batsim_thread = None
        self._batsim_proc = None
        
        self._internal_time = 0.0
        self._is_done = False

    def reset(self) -> None:
        """Reset the simulation by spawning a new BatSim and PyBatsim thread."""
        self._clear_state()
        self._internal_time = 0.0
        self._is_done = False
        
        # Clear queues
        while not self._action_queue.empty(): self._action_queue.get()
        while not self._state_queue.empty(): self._state_queue.get()
        
        self._start_batsim_subprocess()
        self._start_pybatsim_thread()
        
        # Block until pybatsim signals the first state is ready
        self._wait_for_next_state()

    def advance_simulation(self) -> None:
        """Tell PyBatsim to execute our queued actions and run until it needs another decision."""
        if self._is_done:
            return
            
        cmds = self._get_queued_commands()
        
        # Send actions to pybatsim thread
        self._action_queue.put(cmds)
        
        # Wait for simulation to advance and give us control back
        self._wait_for_next_state()

    def close(self) -> None:
        """Clean up threads and subprocesses."""
        self._is_done = True
        if self._batsim_proc:
            self._batsim_proc.terminate()
            self._batsim_proc.wait()
            self._batsim_proc = None

    def _wait_for_next_state(self):
        """Block until pybatsim yields control."""
        try:
            status = self._state_queue.get(timeout=60.0)
            if status == "DONE":
                self._is_done = True
        except queue.Empty:
            print("[RealBatsimAdapter] Timeout waiting for batsim. Is batsim process running?")
            self._is_done = True

    @property
    def current_time(self) -> float:
        return self._internal_time

    @property
    def is_done(self) -> bool:
        return self._is_done

    # ------------------------------------------------------------------------
    # Subprocess & Threading Management
    # ------------------------------------------------------------------------
    def _start_batsim_subprocess(self):
        """Spawn the batsim process (can be via docker or native command)."""
        if self._batsim_proc:
            self._batsim_proc.terminate()
            
        # Example for purely native execution (assuming WSL/Linux)
        # For trace loading, batsim needs arguments.
        trace_path = self.config.workload.trace_path or "workloads/tiny_workload.json"
        platform_path = "data/platforms/small_platform.xml"
        
        cmd = [
            "batsim",
            "-p", platform_path,
            "-w", trace_path,
            "-e", "logs/batsim_out",
            "-s", self.socket_endpoint.replace("*", "localhost")
        ]
        
        print(f"[RealBatsimAdapter] Starting BatSim: {' '.join(cmd)}")
        try:
            self._batsim_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            print("[RealBatsimAdapter] WARNING: 'batsim' executable not found in PATH.")

    def _start_pybatsim_thread(self):
        """Start pybatsim server in background thread."""
        def run_pybatsim():
            bs = Batsim(
                server_endpoint=self.socket_endpoint,
                scheduler=self
            )
            bs.start()
            
        self._batsim_thread = threading.Thread(target=run_pybatsim, daemon=True)
        self._batsim_thread.start()

    # ------------------------------------------------------------------------
    # PyBatsim Callbacks (Executing in Background Thread)
    # ------------------------------------------------------------------------
    def on_simulation_begins(self):
        # Notify RL agent that initial state is ready
        self._state_queue.put("READY")
        
    def on_job_submission(self, job):
        py_job = Job(
            job_id=int(job.id.split("!")[-1] if "!" in job.id else job.id),
            submit_time=job.subtime,
            requested_walltime=job.walltime,
            actual_runtime=job.walltime,
            requested_resources=job.res
        )
        self._pending_jobs.append(py_job)
        self._events.append(Event(EventType.JOB_SUBMITTED, self._internal_time, job=py_job))

    def on_job_completion(self, job):
        py_job = next((j for j in self._running_jobs if str(j.job_id) in str(job.id)), None)
        if py_job:
            py_job.status = JobStatus.COMPLETED
            py_job.finish_time = self._internal_time
            self._running_jobs.remove(py_job)
            self._completed_jobs.append(py_job)
            self.resource.release(py_job.requested_resources)
            self._events.append(Event(EventType.JOB_COMPLETED, self._internal_time, job=py_job))

    def on_requested_call(self):
        # Simulator paused, giving control to Python RL loop
        self._internal_time = self.bs.time()
        self._state_queue.put("WAKEUP")
        
        # Block until Gym provides decisions
        cmds = self._action_queue.get()
        
        # Send translated commands to BatSim
        for cmd in cmds:
            if cmd.command_type == ScheduleCommandType.EXECUTE_JOB and cmd.job:
                self.bs.execute_job(self.bs.jobs[str(cmd.job.job_id)])
                
                cmd.job.status = JobStatus.RUNNING
                cmd.job.start_time = self._internal_time
                self._pending_jobs.remove(cmd.job)
                self._running_jobs.append(cmd.job)
                self.resource.allocate(cmd.job.requested_resources)

    def on_simulation_ends(self):
        self._state_queue.put("DONE")

