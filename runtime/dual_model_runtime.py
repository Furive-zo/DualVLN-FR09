import importlib.metadata as ilm
import importlib.util
import json
import re
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, BatchFeature, Qwen2_5_VLForConditionalGeneration

RUNTIME_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = RUNTIME_DIR.parent
PROJECT_DIR = BUNDLE_DIR.parents[1]
# Prefer the self-contained runtime/internnav package in portable bundles.
# The project checkout paths are fallbacks for an in-place development release.
for path in reversed((RUNTIME_DIR, PROJECT_DIR / 'InternNav', PROJECT_DIR / 'scripts')):
    if path.exists():
        sys.path.insert(0, str(path))

# The inherited training dataset imports torchcodec even though image-only inference
# never decodes video. Keep deployment independent from this optional extension.
try:
    import torchcodec  # noqa: F401
except (ImportError, RuntimeError):
    tc = types.ModuleType('torchcodec')
    tc.__spec__ = importlib.util.spec_from_loader('torchcodec', loader=None)
    tc_dec = types.ModuleType('torchcodec.decoders')
    tc_dec.__spec__ = importlib.util.spec_from_loader('torchcodec.decoders', loader=None)
    tc_dec.VideoDecoder = None
    tc.decoders = tc_dec
    sys.modules.update({'torchcodec': tc, 'torchcodec.decoders': tc_dec})
    _version = ilm.version
    ilm.version = lambda name, *a, **kw: '99.0.0' if name == 'torchcodec' else _version(name, *a, **kw)

from internnav.dataset.internvla_n1_lerobot_dataset import (
    S1_TRAJ_THETA_SCALE,
    S1_TRAJ_XY_SCALE_VLA_E2E,
    S1_USE_AXIS_ROTATION_FIX,
    pseudo_lookdown_crop,
    unrotate_xy_for_pretrain_alignment,
    unrotate_theta_for_pretrain_alignment,
)
from internnav.model.basemodel.internvla_n1.internvla_n1 import (
    InternVLAN1ForCausalLM,
    InternVLAN1ModelConfig,
)
from internnav.model.basemodel.internvla_n1 import internvla_n1_arch as s1_arch
import _fix_nan_params as repair_utils
from _fix_nan_params import fix_nan_params, force_reload_s1_submodules

_COORD_RE = re.compile(r'-?\d+')


def _content(prompt, images):
    parts = prompt.split('<image>')
    if len(parts) != len(images) + 1:
        raise ValueError(f'image placeholder mismatch: {len(parts)-1} != {len(images)}')
    content = []
    for index, part in enumerate(parts):
        if part:
            content.append({'type': 'text', 'text': part})
        if index < len(images):
            content.append({'type': 'image', 'image': images[index]})
    return content


def _history_prompt(count):
    if not count:
        return ''
    placeholders = '<image>\n' * count
    return (
        'The following images are historical front observations in chronological order; '
        'the first image is the initial observation and the last image is the most recent '
        f'front observation: {placeholders}. '
    )


def sample_history(frames, budget):
    if budget <= 0 or not frames:
        return []
    ids = np.unique(np.linspace(0, len(frames) - 1, min(budget, len(frames)), dtype=np.int32))
    return [frames[int(i)] for i in ids]


def build_s2_messages(instruction, history, current_front, current_lookdown,
                      previous_lookdown=None, previous_goal=None):
    use_prior = previous_lookdown is not None and previous_goal is not None
    budget = 7 if use_prior else 8
    selected = sample_history(history, budget)
    history_text = _history_prompt(len(selected))
    if use_prior:
        first_images = selected + [previous_lookdown, current_front]
        prompt = (
            f'You are an autonomous navigation assistant. Your task is to {instruction}. '
            f'{history_text}The previous look-down observation is <image>, and the tracked '
            f'goal in that previous look-down frame was {previous_goal[0]:.0f} '
            f'{previous_goal[1]:.0f}. Keep tracking the same physical target region. The '
            'current front observation is <image>. Where should you go next? Please output '
            "the current waypoint's coordinates and output STOP when the task is complete."
        )
    else:
        first_images = selected + [current_front]
        prompt = (
            f'You are an autonomous navigation assistant. Your task is to {instruction}. '
            f'{history_text}There is no valid previous pixel-goal. Where should you go next '
            "to stay on track? Please output the next waypoint's coordinates in the image. "
            'Please output STOP when you have successfully completed the task. The current '
            'front observation is you can see <image>.'
        )
    messages = [
        {'role': 'user', 'content': _content(prompt, first_images)},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': '↓'}]},
        {'role': 'user', 'content': _content('you can see <image>.',
                                             [current_lookdown])},
    ]
    return messages, first_images + [current_lookdown]


def build_s1_messages(instruction, front, lookdown, goal_text):
    prompt = (
        f'You are an autonomous navigation assistant. Your task is to {instruction}. '
        "Where should you go next to stay on track? Please output the next waypoint's "
        'coordinates in the image. Please output STOP when you have successfully completed '
        'the task. ahead of you is <image>.'
    )
    return [
        {'role': 'user', 'content': _content(prompt, [front])},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': '↓'}]},
        {'role': 'user', 'content': _content('ahead of you is <image>.', [lookdown])},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': goal_text}]},
    ], [front, lookdown]


def parse_s2_output(text, width=1920, height=1080):
    if 'STOP' in text.upper():
        return 'stop', None
    values = _COORD_RE.findall(text)
    if len(values) < 2:
        return 'invalid', None
    goal = [int(values[0]), int(values[1])]
    if not (0 <= goal[0] < width and 0 <= goal[1] < height):
        return 'invalid', goal
    return 'coordinate', goal


def physical_trajectories(prediction):
    delta = prediction.float().cpu().numpy().copy()
    delta[..., :2] /= S1_TRAJ_XY_SCALE_VLA_E2E
    delta[..., 2] /= S1_TRAJ_THETA_SCALE
    if S1_USE_AXIS_ROTATION_FIX:
        x, y = unrotate_xy_for_pretrain_alignment(delta[..., 0], delta[..., 1])
        delta[..., 0], delta[..., 1] = x, y
        delta[..., 2] = unrotate_theta_for_pretrain_alignment(delta[..., 2])
    poses = np.concatenate((np.cumsum(delta[..., :2], axis=1),
                            np.cumsum(delta[..., 2], axis=1)[..., None]), axis=-1)
    motion_angle = np.arctan2(delta[..., 1], delta[..., 0])
    heading = np.cumsum(delta[..., 2], axis=1)
    mismatch = np.arctan2(np.sin(heading - motion_angle), np.cos(heading - motion_angle))
    moving = np.linalg.norm(delta[..., :2], axis=-1) > 0.05
    reverse = ((np.cos(mismatch) < 0.0) & moving).any(axis=1)
    behind = (-poses[..., -1, 1]) < -0.05
    valid = ~(reverse | behind)
    mean_pose = poses.mean(axis=0)
    if valid.any():
        valid_ids = np.flatnonzero(valid)
        distances = np.linalg.norm(poses[valid_ids, -1, :2] - mean_pose[-1, :2], axis=1)
        selected = int(valid_ids[int(np.argmin(distances))])
    else:
        selected = None
    return {
        'candidates': poses.tolist(),
        'mean': mean_pose.tolist(),
        'valid_candidate_mask': valid.tolist(),
        'selected_index': selected,
        'selected': poses[selected].tolist() if selected is not None else None,
    }


class DualVLNE2ERuntime:
    def __init__(self, system2_path, system1_path, device='cuda:0',
                 num_candidates=4, num_inference_steps=10, s1_device=None,
                 load_s2=True, load_s1=True):
        self.device = torch.device(device)
        self.s1_device = torch.device(s1_device) if s1_device else self.device
        self.num_candidates = num_candidates
        self.num_inference_steps = num_inference_steps
        self.lock = threading.Lock()
        # The async runtime loads S2 and S1 in separate processes (load_s2 / load_s1).
        self.s2_processor = AutoProcessor.from_pretrained(system2_path, local_files_only=True)
        self.s2_processor.tokenizer.padding_side = 'left'
        if load_s2:
            self.s2_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                system2_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                local_files_only=True,
            ).to(self.device).eval()
        if load_s1:
            self._load_s1(system1_path, system2_path)
        self.reset()

    def _load_s1(self, system1_path, system2_path):
        dav2_name = 'depth_anything_v2_metric_hypersim_vits.pth'
        portable_dav2 = BUNDLE_DIR / 'checkpoints' / dav2_name
        project_dav2 = PROJECT_DIR / 'checkpoints' / dav2_name
        dav2_path = portable_dav2 if portable_dav2.is_file() else project_dav2
        if not dav2_path.is_file():
            raise FileNotFoundError(f'required DepthAnything checkpoint missing: {dav2_path}')
        s1_arch.MODEL_PATH_TO = str(dav2_path.parent)
        repair_utils.DAV2_CKPT = str(dav2_path)
        cfg = InternVLAN1ModelConfig.from_pretrained(system1_path, local_files_only=True)
        self.s1_model = InternVLAN1ForCausalLM.from_pretrained(
            system1_path, config=cfg, torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, local_files_only=True,
        )
        force_reload_s1_submodules(self.s1_model, ckpt_dir=system1_path, fallback_ckpt_dir=None)
        fix_nan_params(self.s1_model, ckpt_dir=system1_path, fallback_ckpt_dir=None)
        self.s1_model.config.use_cache = False
        self.s1_model = self.s1_model.to(self.s1_device).eval()
        self.s1_processor = AutoProcessor.from_pretrained(system2_path, local_files_only=True)
        self.s1_processor.tokenizer = AutoTokenizer.from_pretrained(
            system1_path, padding_side='left', use_fast=False, local_files_only=True)

    def reset(self):
        self.history = []
        self.previous_lookdown = None
        self.previous_goal = None
        self.active_instruction = None
        self.safe_stop_instruction = None
        self._pixel_cache = {}

    def _pixels(self, image):
        # Per-image preprocessing is bit-identical to the batched processor call, so each
        # front is processed once and reused while it stays in the history. S1 shares the
        # S2 image processor config, so the same entry also serves S1 inputs.
        hit = self._pixel_cache.get(id(image))
        if hit is None or hit[0] is not image:
            out = self.s2_processor.image_processor(images=[image], return_tensors='pt')
            hit = (image, out['pixel_values'], out['image_grid_thw'])
            self._pixel_cache[id(image)] = hit
        return hit[1], hit[2]

    def _prune_pixel_cache(self, history=None):
        live = {id(image) for image in (self.history if history is None else history)}
        if self.previous_lookdown is not None:
            live.add(id(self.previous_lookdown))
        self._pixel_cache = {k: v for k, v in self._pixel_cache.items() if k in live}

    def _processor_inputs(self, processor, messages, images, generation_prompt,
                          device=None, pixels=None):
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt)
        pixels = pixels or [self._pixels(image) for image in images]
        grids = torch.cat([grid for _, grid in pixels])
        # Same image-token expansion as Qwen2_5_VLProcessor.__call__.
        merge = processor.image_processor.merge_size ** 2
        parts = text.split(processor.image_token)
        if len(parts) != len(images) + 1:
            raise ValueError(f'image token mismatch: {len(parts)-1} != {len(images)}')
        text = parts[0] + ''.join(processor.image_token * int(grid.prod() // merge) + part
                                  for grid, part in zip(grids, parts[1:]))
        inputs = processor.tokenizer([text], return_tensors='pt')
        return BatchFeature({**inputs, 'pixel_values': torch.cat([pv for pv, _ in pixels]),
                             'image_grid_thw': grids}).to(device or self.device)

    def _s2_goal(self, instruction, history, front, lookdown,
                 previous_lookdown=None, previous_goal=None):
        messages, images = build_s2_messages(
            instruction, history, front, lookdown, previous_lookdown, previous_goal)
        s2_inputs = self._processor_inputs(self.s2_processor, messages, images, True)
        generated = self.s2_model.generate(
            **s2_inputs, max_new_tokens=48, do_sample=False, return_dict_in_generate=True)
        new_ids = generated.sequences[0, s2_inputs.input_ids.shape[1]:]
        text = self.s2_processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        status, goal = parse_s2_output(text)
        return status, goal, text

    def _s1_trajectory(self, instruction, front, lookdown, goal, pixel_device=None):
        s1_messages, s1_images = build_s1_messages(
            instruction, front, lookdown, f'{goal[0]} {goal[1]}')
        pixels = None
        if pixel_device is not None:
            # Bypasses the shared pixel cache (owned by the S2 worker in async mode).
            # On a GPU device the resize is not bit-identical to the CPU path.
            pixels = []
            for image in s1_images:
                out = self.s1_processor.image_processor(
                    images=[image], return_tensors='pt', device=pixel_device)
                pixels.append((out['pixel_values'], out['image_grid_thw']))
        s1_inputs = self._processor_inputs(self.s1_processor, s1_messages, s1_images, False,
                                           device=self.s1_device, pixels=pixels)
        latents = self.s1_model.generate_latents(
            s1_inputs.input_ids, s1_inputs.pixel_values, s1_inputs.image_grid_thw)
        image = torch.from_numpy(np.asarray(lookdown.resize((224, 224))).copy()).to(self.s1_device)
        image = (image.float() / 255.0).to(torch.bfloat16)
        images_dp = torch.stack([image, image], dim=0).unsqueeze(0)
        pred = self.s1_model.generate_traj(
            latents, images_dp, predict_step_nums=self.s1_model.config.n_query,
            num_sample_trajs=self.num_candidates,
            num_inference_steps=self.num_inference_steps,
        )
        return physical_trajectories(pred)

    @torch.inference_mode()
    def infer(self, front, instruction, command, lookdown=None, reset=False):
        started = time.perf_counter()
        with self.lock:
            if reset:
                self.reset()
            self._prune_pixel_cache()
            self.active_instruction = instruction
            front = front.convert('RGB')
            if command == 'stop':
                self.history.append(front)
                self.history = self.history[-64:]
                return {'status': 'stop', 'control_decision': 'stop', 'discrete_action': 0,
                        's2_text': 'STOP', 'pixel_goal_lookdown': None,
                        'latency_ms': round((time.perf_counter() - started) * 1000, 1)}
            lookdown = lookdown.convert('RGB') if lookdown is not None else pseudo_lookdown_crop(front)
            track_prior = command == 'safe_stop'
            if track_prior and self.safe_stop_instruction != instruction:
                self.previous_lookdown = None
                self.previous_goal = None
            self.safe_stop_instruction = instruction if track_prior else None
            status, goal, text = self._s2_goal(
                instruction, self.history, front, lookdown,
                self.previous_lookdown if track_prior else None,
                self.previous_goal if track_prior else None,
            )
            self.history.append(front)
            self.history = self.history[-64:]
            result = {'status': status, 's2_text': text, 'pixel_goal_lookdown': goal}
            if status == 'stop':
                result.update(control_decision='stop', discrete_action=0,
                              latency_ms=round((time.perf_counter() - started) * 1000, 1))
                return result
            if status != 'coordinate':
                result.update(control_decision='stop', discrete_action=0,
                              latency_ms=round((time.perf_counter() - started) * 1000, 1))
                return result
            if track_prior:
                self.previous_lookdown = lookdown
                self.previous_goal = goal
            result['trajectory'] = self._s1_trajectory(instruction, front, lookdown, goal)
            if result['trajectory']['selected'] is None:
                result.update(status='unsafe_trajectory', control_decision='stop', discrete_action=0)
            else:
                result['control_decision'] = 'trajectory'
            result['latency_ms'] = round((time.perf_counter() - started) * 1000, 1)
            return result
