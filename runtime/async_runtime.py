"""Asynchronous S2/S1 runtime: S2 goal worker process and S1 control path on separate GPUs.

Each /infer request only runs S1 with the latest S2 goal and forwards the frame to
the S2 process. S2 runs in its own process so its CPU preprocessing and generate
loop do not contend with S1 for the GIL. The S2 process receives every frame,
keeps the episode history exactly like the synchronous runtime, and always
processes the newest frame (older unprocessed frames only join the history).

Fail-safe STOP is returned when no goal exists yet for the current
(episode, instruction, command), when the goal is older than max_goal_age_s,
when S2 answered STOP / an invalid coordinate, or when the S2 process died.
"""
import multiprocessing as mp
import queue
import threading
import time
import traceback

import torch

from dual_model_runtime import DualVLNE2ERuntime, pseudo_lookdown_crop


def _ms(seconds):
    return round(seconds * 1000, 1)


def _s2_process_main(system2_path, system1_path, device, inbox, outbox):
    try:
        rt = DualVLNE2ERuntime(system2_path, system1_path, device, load_s1=False)
    except Exception:
        outbox.put(('fatal', traceback.format_exc()))
        return
    outbox.put(('ready', None))
    # A dedicated reader keeps draining the pipe while S2 computes. Draining only
    # between S2 calls races the parent's feeder thread (each frame is ~6 MB), so
    # get_nowait() sees one frame at a time and a backlog builds up.
    buffered, arrived = [], threading.Condition()

    def reader():
        while True:
            message = inbox.get()
            with arrived:
                buffered.append(message)
                arrived.notify()
    threading.Thread(target=reader, name='s2-inbox-reader', daemon=True).start()
    generation, dropped = None, 0
    while True:
        with arrived:
            while not buffered:
                arrived.wait()
            messages = buffered[:]
            buffered.clear()
        job = None
        for kind, data in messages:
            if kind == 'reset':
                rt.reset()
                generation, job = data, None
                continue
            if job is not None:  # superseded by a newer frame: history only
                rt.history.append(job['front'])
                dropped += 1
                job = None
            if data['process']:
                job = data
            else:
                rt.history.append(data['front'])
        rt.history = rt.history[-64:]
        if job is None:
            continue
        key = job['key']
        if key[0] != generation:  # first frame of an episode without an explicit reset
            rt.reset()
            generation = key[0]
        _, instruction, command = key
        front = job['front']
        lookdown = job['lookdown'] or pseudo_lookdown_crop(front)
        track_prior = command == 'safe_stop'
        if track_prior and rt.safe_stop_instruction != instruction:
            rt.previous_lookdown = rt.previous_goal = None
        rt.safe_stop_instruction = instruction if track_prior else None
        rt._prune_pixel_cache()
        started = time.monotonic()
        try:
            with torch.inference_mode():
                status, goal, text = rt._s2_goal(
                    instruction, rt.history, front, lookdown,
                    rt.previous_lookdown if track_prior else None,
                    rt.previous_goal if track_prior else None)
        except Exception as exc:
            status, goal, text = 'runtime_error', None, f'{exc}\n{traceback.format_exc()}'
        done = time.monotonic()
        rt.history.append(front)
        rt.history = rt.history[-64:]
        if track_prior and status == 'coordinate':
            rt.previous_lookdown, rt.previous_goal = lookdown, goal
        outbox.put(('goal', {'key': key, 'status': status, 'goal': goal, 'text': text,
                             'seq': job['seq'], 'captured_at': job['captured_at'],
                             's2_ms': _ms(done - started),
                             'queue_ms': _ms(started - job['captured_at']), 'dropped': dropped}))


class AsyncDualVLNRuntime:
    def __init__(self, system2_path, system1_path, s2_device='cuda:0', s1_device='cuda:1',
                 num_candidates=4, num_inference_steps=10, max_goal_age_s=2.25,
                 s1_gpu_preprocess=True, s2_start_timeout_s=900):
        ctx = mp.get_context('spawn')  # CUDA cannot be re-initialised in a forked child
        self.inbox, self.outbox = ctx.Queue(), ctx.Queue()
        # Do not block interpreter exit on unsent frames if the S2 process is gone.
        self.inbox.cancel_join_thread()
        self.s2_proc = ctx.Process(target=_s2_process_main, name='dualvln-s2',
                                   args=(str(system2_path), str(system1_path), s2_device,
                                         self.inbox, self.outbox), daemon=True)
        self.s2_proc.start()
        # S1 loads while the S2 process loads its model.
        self.core = DualVLNE2ERuntime(system2_path, system1_path, s1_device, num_candidates,
                                      num_inference_steps, s1_device=s1_device, load_s2=False)
        self.max_goal_age_s = max_goal_age_s
        self.s1_gpu_preprocess = s1_gpu_preprocess
        self.lock = threading.Lock()
        self.generation = 0
        self.seq = 0
        self.goal = None
        self.dropped_frames = 0
        kind, detail = self.outbox.get(timeout=s2_start_timeout_s)
        if kind != 'ready':
            raise RuntimeError(f'S2 process failed to start:\n{detail}')
        self.listener = threading.Thread(target=self._listen, name='s2-goal-listener', daemon=True)
        self.listener.start()

    def _listen(self):
        while True:
            try:
                kind, goal = self.outbox.get(timeout=1.0)
            except queue.Empty:
                if not self.s2_proc.is_alive():
                    return
                continue
            with self.lock:
                if kind == 'goal' and goal['key'][0] == self.generation:
                    self.goal = goal
                    self.dropped_frames = goal['dropped']

    def reset(self):
        with self.lock:
            self._reset_locked()

    def _reset_locked(self):
        self.generation += 1
        self.goal = None
        self.inbox.put(('reset', self.generation))

    def infer(self, front, instruction, command, lookdown=None, reset=False):
        started = time.monotonic()
        front = front.convert('RGB')
        lookdown = lookdown.convert('RGB') if lookdown is not None else None
        with self.lock:
            if reset:
                self._reset_locked()
            self.seq += 1
            key = (self.generation, instruction, command)
            self.inbox.put(('frame', {'key': key, 'seq': self.seq, 'front': front,
                                      'lookdown': lookdown, 'captured_at': started,
                                      'process': command != 'stop'}))
            goal = self.goal if self.goal is not None and self.goal['key'] == key else None
            meta = {'mode': 'async', 'frame_seq': self.seq, 'dropped_s2_frames': self.dropped_frames}

        def finish(result):
            result.update(meta)
            result['latency_ms'] = _ms(time.monotonic() - started)
            return result

        if command == 'stop':
            return finish({'status': 'stop', 'control_decision': 'stop', 'discrete_action': 0,
                           's2_text': 'STOP', 'pixel_goal_lookdown': None})
        if not self.s2_proc.is_alive():
            return finish({'status': 's2_process_dead', 'control_decision': 'stop',
                           'discrete_action': 0, 's2_text': None, 'pixel_goal_lookdown': None})
        if goal is None:
            return finish({'status': 'awaiting_s2_goal', 'control_decision': 'stop',
                           'discrete_action': 0, 's2_text': None, 'pixel_goal_lookdown': None})
        age = started - goal['captured_at']
        meta.update(goal_frame_seq=goal['seq'], goal_age_ms=_ms(age),
                    s2_latency_ms=goal['s2_ms'], s2_queue_ms=goal['queue_ms'])
        result = {'status': goal['status'], 's2_text': goal['text'],
                  'pixel_goal_lookdown': goal['goal']}
        if goal['status'] != 'coordinate':
            result.update(control_decision='stop', discrete_action=0)
            return finish(result)
        if age > self.max_goal_age_s:
            result.update(status='stale_s2_goal', control_decision='stop', discrete_action=0)
            return finish(result)
        s1_started = time.monotonic()
        with torch.inference_mode():
            trajectory = self.core._s1_trajectory(
                instruction, front, lookdown or pseudo_lookdown_crop(front), goal['goal'],
                pixel_device=self.core.s1_device if self.s1_gpu_preprocess else 'cpu')
        meta['s1_latency_ms'] = _ms(time.monotonic() - s1_started)
        result['trajectory'] = trajectory
        if trajectory['selected'] is None:
            result.update(status='unsafe_trajectory', control_decision='stop', discrete_action=0)
        else:
            result['control_decision'] = 'trajectory'
        return finish(result)
