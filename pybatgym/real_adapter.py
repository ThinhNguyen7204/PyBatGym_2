"""Real BatSim adapter connecting PyBatGym to the actual C++ BatSim simulator via pybatsim."""

import queue
import subprocess
import threading
from pathlib import Path
from typing import Optional

from pybatgym.batsim_adapter import BatsimAdapter
from pybatgym.models import (
    Event,
    EventType,
    Job,
    JobStatus,
    Resource,
    ScheduleCommand,
    ScheduleCommandType,
)

try:
    from batsim.batsim import BatsimScheduler, Batsim
except ImportError:
    BatsimScheduler = object

    def Batsim(*args, **kwargs):
        raise ImportError(
            "pybatsim is not installed. Run: pip install pybatsim"
        )


# Default paths resolved relative to the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BATSIM_DATA = _PROJECT_ROOT / "batsim_data"
_DEFAULT_PLATFORM = _BATSIM_DATA / "platforms" / "small_platform.xml"
_DEFAULT_WORKLOAD = _PROJECT_ROOT / "data" / "workloads" / "tiny_workload.json"


class RealBatsimAdapter(BatsimAdapter, BatsimScheduler):
    """Integrates real BatSim simulator using pybatsim over ZeroMQ.

    Architecture:
    - Main thread (RL agent) ←→ queues ←→ background thread (pybatsim)
    - Background thread runs BatSim C++ process + pybatsim event loop
    - Queues decouple synchronous RL step() calls from async BatSim callbacks
    """

    def __init__(self, config, socket_endpoint: str = "tcp://*:28000") -> None:
        BatsimScheduler.__init__(self)

        self.config = config
        self.socket_endpoint = socket_endpoint

        # Threading bridges
        self._action_queue: queue.Queue = queue.Queue()
        self._state_queue: queue.Queue = queue.Queue()
        self._batsim_thread: Optional[threading.Thread] = None
        self._batsim_proc: Optional[subprocess.Popen] = None

        # Simulation state (shared between threads - updated by pybatsim callbacks)
        self._internal_time: float = 0.0
        self._is_done: bool = False
        self._pending_jobs: list[Job] = []
        self._running_jobs: list[Job] = []
        self._completed_jobs: list[Job] = []
        self._events: list[Event] = []
        self._resource: Resource = self._make_resource()

    # --------------------------------------------------------------------------
    # BatsimAdapter interface (called from main/RL thread)
    # --------------------------------------------------------------------------

    def start(self) -> None:
        """No-op: actual initialization happens in reset()."""
        pass

    def reset(self) -> tuple[list[Event], Resource]:
        """Restart simulation: spawn BatSim process + pybatsim thread."""
        self._clear_state()
        self._start_batsim_subprocess()
        self._start_pybatsim_thread()
        self._wait_for_next_state()
        return self._consume_events(), self._resource

    def step(self, command: Optional[ScheduleCommand]) -> tuple[list[Event], bool]:
        """Send scheduling decision and advance simulation to next decision point."""
        if self._is_done:
            return [], True

        self._action_queue.put([command] if command else [])
        self._wait_for_next_state()
        return self._consume_events(), self._is_done

    def close(self) -> None:
        """Terminate BatSim process and background thread."""
        self._is_done = True
        if self._batsim_proc:
            self._batsim_proc.terminate()
            self._batsim_proc.wait()
            self._batsim_proc = None

    def get_current_time(self) -> float:
        return self._internal_time

    def get_pending_jobs(self) -> list[Job]:
        return list(self._pending_jobs)

    def get_completed_jobs(self) -> list[Job]:
        return list(self._completed_jobs)

    # --------------------------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------------------------

    def _make_resource(self) -> Resource:
        return Resource(
            total_nodes=self.config.platform.total_nodes,
            total_cores_per_node=self.config.platform.cores_per_node,
        )

    def _clear_state(self) -> None:
        self._internal_time = 0.0
        self._is_done = False
        self._pending_jobs = []
        self._running_jobs = []
        self._completed_jobs = []
        self._events = []
        self._resource = self._make_resource()
        # Drain queues
        for q in (self._action_queue, self._state_queue):
            while not q.empty():
                q.get_nowait()

    def _consume_events(self) -> list[Event]:
        events, self._events = self._events, []
        return events

    def _wait_for_next_state(self, timeout: float = 60.0) -> None:
        """Block main thread until pybatsim yields control."""
        try:
            status = self._state_queue.get(timeout=timeout)
            if status == "DONE":
                self._is_done = True
        except queue.Empty:
            print(
                "[RealBatsimAdapter] Timeout waiting for BatSim. "
                "Is the batsim binary in PATH?"
            )
            self._is_done = True

    def _resolve_paths(self) -> tuple[str, str]:
        """Return (platform_path, workload_path) resolving config or defaults."""
        platform = str(_DEFAULT_PLATFORM)

        trace = self.config.workload.trace_path or ""
        workload = trace if trace and Path(trace).exists() else str(_DEFAULT_WORKLOAD)

        return platform, workload

    def _start_batsim_subprocess(self) -> None:
        """Spawn the BatSim C++ process."""
        if self._batsim_proc and self._batsim_proc.poll() is None:
            self._batsim_proc.terminate()

        platform, workload = self._resolve_paths()
        socket = self.socket_endpoint.replace("*", "localhost")

        Path("logs").mkdir(exist_ok=True)
        cmd = ["batsim", "-p", platform, "-w", workload, "-e", "logs/batsim_out", "-s", socket]

        print(f"[RealBatsimAdapter] Starting BatSim: {' '.join(cmd)}")
        try:
            self._batsim_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            print(
                "[RealBatsimAdapter] ERROR: 'batsim' not found in PATH.\n"
                "  Add it with: export PATH=/workspace/batsim_data/result/bin:$PATH"
            )

    def _start_pybatsim_thread(self) -> None:
        """Start pybatsim ZeroMQ server in background thread."""

        def _run() -> None:
            # pybatsim 3.x: Batsim(scheduler, network_endpoint, timeout)
            bs = Batsim(self, self.socket_endpoint, timeout=120)
            bs.start()

        self._batsim_thread = threading.Thread(target=_run, daemon=True)
        self._batsim_thread.start()

    # --------------------------------------------------------------------------
    # PyBatsim scheduler callbacks (called from background thread)
    # --------------------------------------------------------------------------

    def on_simulation_begins(self) -> None:
        self._state_queue.put("READY")

    def on_job_submission(self, job) -> None:
        job_id_str = job.id.split("!")[-1] if "!" in job.id else job.id
        try:
            job_id = int(job_id_str)
        except ValueError:
            job_id = hash(job_id_str)

        py_job = Job(
            job_id=job_id,
            submit_time=job.subtime,
            requested_walltime=job.walltime,
            actual_runtime=job.walltime,
            requested_resources=job.res,
        )
        self._pending_jobs.append(py_job)
        self._events.append(Event(EventType.JOB_SUBMITTED, self._internal_time, job=py_job))

    def on_job_completion(self, job) -> None:
        job_id_str = job.id.split("!")[-1] if "!" in job.id else job.id
        py_job = next(
            (j for j in self._running_jobs if str(j.job_id) == job_id_str), None
        )
        if py_job:
            py_job.status = JobStatus.COMPLETED
            py_job.finish_time = self._internal_time
            self._running_jobs.remove(py_job)
            self._completed_jobs.append(py_job)
            self._resource.release(py_job.requested_resources)
            self._events.append(Event(EventType.JOB_COMPLETED, self._internal_time, job=py_job))

    def on_requested_call(self) -> None:
        """BatSim paused: send state to RL agent, wait for decision."""
        self._internal_time = self.bs.time()
        self._state_queue.put("WAKEUP")

        # Block until RL agent provides decisions
        cmds: list[Optional[ScheduleCommand]] = self._action_queue.get()

        for cmd in cmds:
            if cmd and cmd.command_type == ScheduleCommandType.EXECUTE_JOB and cmd.job:
                batsim_job_id = str(cmd.job.job_id)
                batsim_jobs = getattr(self.bs, "jobs", {})
                if batsim_job_id in batsim_jobs:
                    self.bs.execute_job(batsim_jobs[batsim_job_id])
                    cmd.job.status = JobStatus.RUNNING
                    cmd.job.start_time = self._internal_time
                    if cmd.job in self._pending_jobs:
                        self._pending_jobs.remove(cmd.job)
                    self._running_jobs.append(cmd.job)
                    if self._resource.can_allocate(cmd.job.requested_resources):
                        self._resource.allocate(cmd.job.requested_resources)

    def on_simulation_ends(self) -> None:
        self._state_queue.put("DONE")
