"""CoView-VLN System 3 dataset.

Terminology
-----------
ESV (Exocentric Scene View): any fixed, non-ego external view of the environment.
  Covers overhead CCTV cameras, architectural floor plans, satellite/aerial maps, etc.
  At this stage we use an overhead rendered BEV as the ESV; the abstraction
  generalises to other exocentric modalities without code changes.

System 3 role: ESV + ego → navigation instruction generation
  - Trained to reconstruct the original R2R/RxR instruction from visual context
  - At inference: generates a spatially-grounded instruction for System 2

Input images (2):
  ① ESV  — overhead scene layout + trajectory overlay + goal marker (red cross)
  ② Ego  — agent's current forward-facing RGB frame

GT label: original R2R/RxR instruction text

Inference pipeline:
  S3(ESV+goal, ego) → instruction → S2(ego history + instruction) → pixel-goal → S1 → action

Graceful degradation (ESV/communication unavailable):
  S3 skipped → original instruction fed directly to S2  (= DualVLN baseline)
"""

import copy
import math
import os
import random
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from .internvla_n1_lerobot_dataset import (
    NavPixelGoalDataset,
    preprocess_qwen_2_visual,
)


# ──────────────────────────────────────────────────────────────────────────────
# Overlay palette
# ──────────────────────────────────────────────────────────────────────────────
_C_PAST        = (255, 200, 0)    # yellow  — past trajectory dots
_C_PAST_BORDER = (60,  60,  60)  # dark    — dot outline for contrast
_C_CURR        = (0, 220, 220)   # cyan    — current position
_C_WHITE       = (255, 255, 255)
_C_ARROW       = (0, 180, 255)   # blue    — heading direction arrow
_C_GOAL        = (220, 30, 30)   # red     — final destination
_DOT_R         = 5               # slightly larger for visibility
_CURR_R        = 7
_GOAL_R        = 9
_ARROW_LEN     = 26              # pixels


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extrinsic_to_world_pos(T_world2cam) -> np.ndarray:
    """4×4 T_world2camera → world [x, y, z]."""
    T = np.array(T_world2cam, dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(T)[:3, 3]


def _world_to_bev_pixel(world_pos, bev_meta: dict):
    """World [x, y, z] → BEV pixel (u, v) via pinhole projection.

    Camera at (ref_x, ref_y + h, ref_z) looking straight down.
    Image axes: right = +X world, down = +Z world.
    """
    ref  = bev_meta['ref_pos']
    h    = bev_meta['overhead_height']
    size = bev_meta['img_size']
    hfov = bev_meta.get('hfov', 90.0)

    focal = (size / 2) / math.tan(math.radians(hfov / 2))
    depth = max(ref[1] + h - world_pos[1], 0.1)

    u = int(size / 2 + (world_pos[0] - ref[0]) * focal / depth)
    v = int(size / 2 + (world_pos[2] - ref[2]) * focal / depth)
    return u, v


# ──────────────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────────────
_CONJUNCTIONS = [
    "Based on these views,",
    "Looking at the overhead map and your current perspective,",
    "From the infrastructure map and ego view,",
    "Given the overhead view and your current position,",
]

_PROMPT_TEMPLATE = (
    "You are a navigation instruction generator for an autonomous robot.\n"
    "[Image 1: Exocentric Scene View (ESV)]\n"
    "An external fixed-viewpoint map of the environment "
    "(overhead camera / floor plan / aerial view). "
    "Yellow dots = past trajectory. Cyan circle = current position. "
    "Blue arrow from cyan circle = heading direction. "
    "Red cross = final destination.\n"
    "[Image 2: Ego View]\n"
    "The robot's current first-person forward perspective.\n"
    "{conjunction} generate a natural language navigation instruction that would guide "
    "the robot from its current position (cyan dot) to the final destination (red cross). "
    "Describe landmarks and turns that are visible in the images.\n"
    "<image> <image>"
)


# ──────────────────────────────────────────────────────────────────────────────
# CoViewNavDataset (System 3)
# ──────────────────────────────────────────────────────────────────────────────

class CoViewNavDataset(NavPixelGoalDataset):
    """System 3 training dataset for CoView-VLN.

    Teaches the VLM to generate navigation instructions from an
    Exocentric Scene View (ESV) + ego-view pair.
    Original R2R/RxR instruction text is the GT supervision signal.

    Required extra field in data_args:
        bev_data_path (str): root dir containing per-scene ESV assets.
            At this stage the ESV is an overhead-rendered BEV.
            └── {scene_id}/
                ├── overhead.jpg        # pre-rendered ESV background
                └── overhead_meta.json  # projection parameters
    """

    def __init__(self, tokenizer, data_args):
        super().__init__(tokenizer, data_args)
        self.bev_data_path = getattr(data_args, 'bev_data_path', None)
        if not self.bev_data_path:
            raise ValueError("data_args.bev_data_path must be set for CoViewNavDataset")

    # ── BEV assets ────────────────────────────────────────────────────────

    def _bev_bg(self, scene_id: str) -> Image.Image:
        return Image.open(
            os.path.join(self.bev_data_path, scene_id, "overhead.jpg")
        ).convert('RGB')

    def _bev_meta(self, scene_id: str) -> dict:
        import json as _json
        with open(os.path.join(self.bev_data_path, scene_id, "overhead_meta.json")) as f:
            return _json.load(f)

    # ── Past-trajectory loading ───────────────────────────────────────────

    def _load_past_poses(self, data_path: str, scene_id: str,
                         ep_id: int, pose_key: str, n_steps: int) -> list:
        import pyarrow.parquet as pq
        parquet = os.path.join(
            data_path, scene_id, "data",
            f"chunk-{ep_id // 1000:03d}",
            f"episode_{ep_id:06d}.parquet",
        )
        tbl = pq.read_table(parquet, columns=[pose_key])
        poses = tbl.to_pandas()[pose_key].apply(lambda x: x.tolist()).tolist()
        return poses[:n_steps]

    # ── BEV image with overlays ───────────────────────────────────────────

    def _make_bev_image(self, bev_bg: Image.Image, bev_meta: dict,
                        past_poses: list, current_pose, goal_pose) -> Image.Image:
        """Render BEV with trajectory overlay + goal marker onto background."""
        img  = bev_bg.copy()
        draw = ImageDraw.Draw(img)

        ref   = bev_meta['ref_pos']
        h     = bev_meta['overhead_height']
        size  = bev_meta['img_size']
        hfov  = bev_meta.get('hfov', 90.0)
        focal = (size / 2) / math.tan(math.radians(hfov / 2))

        def to_pix(world_pos):
            depth = max(ref[1] + h - world_pos[1], 0.1)
            u = int(size / 2 + (world_pos[0] - ref[0]) * focal / depth)
            v = int(size / 2 + (world_pos[2] - ref[2]) * focal / depth)
            return u, v

        # Past trajectory (yellow dots with dark outline, down-sampled to ≤ 20)
        if past_poses:
            step = max(1, len(past_poses) // 20)
            for pm in past_poses[::step]:
                wp = _extrinsic_to_world_pos(pm)
                u, v = to_pix(wp)
                draw.ellipse([u - _DOT_R - 1, v - _DOT_R - 1, u + _DOT_R + 1, v + _DOT_R + 1],
                             fill=_C_PAST_BORDER)
                draw.ellipse([u - _DOT_R, v - _DOT_R, u + _DOT_R, v + _DOT_R],
                             fill=_C_PAST)

        # Current position (cyan circle)
        curr_wp = _extrinsic_to_world_pos(current_pose)
        cu, cv  = to_pix(curr_wp)
        draw.ellipse(
            [cu - _CURR_R, cv - _CURR_R, cu + _CURR_R, cv + _CURR_R],
            fill=_C_CURR, outline=_C_WHITE, width=2,
        )

        # Heading arrow (arrived direction: past[-1] → current)
        if past_poses:
            prev_wp = _extrinsic_to_world_pos(past_poses[-1])
            dx = float(curr_wp[0] - prev_wp[0])
            dz = float(curr_wp[2] - prev_wp[2])
            mag = math.sqrt(dx * dx + dz * dz)
            if mag > 1e-4:
                du = (dx / mag) * _ARROW_LEN
                dv = (dz / mag) * _ARROW_LEN
                draw.line([(cu, cv), (int(cu + du), int(cv + dv))],
                          fill=_C_ARROW, width=3)

        # Goal destination (red star / filled circle)
        goal_wp  = _extrinsic_to_world_pos(goal_pose)
        gu, gv   = to_pix(goal_wp)
        # Draw a star-like marker: outer circle + inner cross
        draw.ellipse(
            [gu - _GOAL_R, gv - _GOAL_R, gu + _GOAL_R, gv + _GOAL_R],
            fill=_C_GOAL, outline=_C_WHITE, width=2,
        )
        # Cross inside circle
        draw.line([(gu - _GOAL_R + 2, gv), (gu + _GOAL_R - 2, gv)],
                  fill=_C_WHITE, width=2)
        draw.line([(gu, gv - _GOAL_R + 2), (gu, gv + _GOAL_R - 2)],
                  fill=_C_WHITE, width=2)

        return img

    # ── Ego view loading ──────────────────────────────────────────────────

    def _load_ego_frame(self, video: str, height: int, pitch_1: int,
                        ep_id: int, frame_id: int) -> Image.Image:
        path = os.path.join(
            video,
            f"observation.images.rgb.{height}cm_{pitch_1}deg",
            f"episode_{ep_id:06d}_{frame_id}.jpg",
        )
        return Image.open(path).convert('RGB')

    # ── Prompt ───────────────────────────────────────────────────────────

    def _build_prompt(self) -> str:
        return _PROMPT_TEMPLATE.format(
            conjunction=random.choice(_CONJUNCTIONS)
        )

    # ── __getitem__ ───────────────────────────────────────────────────────

    def __getitem__(self, i):
        (ep_id, data_path, video, height, pitch_1, pitch_2, instruction,
         (start_frame_id, end_frame_id), action, pose) = self.list_data_dict[i]

        # Skip samples without a pixel-goal (STOP or turn-only steps)
        # System 3 needs a defined goal pose to mark on the BEV
        if pose is None:
            # Fall back to a neighbouring pixel-goal sample
            return self.__getitem__((i + 1) % len(self.list_data_dict))

        # ── Scene ID ──────────────────────────────────────────────────────
        rel_video = os.path.relpath(video, data_path)
        scene_id  = rel_video.split(os.sep)[0]
        pose_key  = f"pose.{height}cm_{pitch_2}deg"

        # ── BEV image (Image 1) ───────────────────────────────────────────
        bev_bg   = self._bev_bg(scene_id)
        bev_meta = self._bev_meta(scene_id)

        past_poses = []
        if start_frame_id > 0:
            try:
                past_poses = self._load_past_poses(
                    data_path, scene_id, ep_id, pose_key, start_frame_id
                )
            except Exception:
                past_poses = []

        current_pose = pose[0]   # agent position at start_frame_id
        goal_pose    = pose[-1]  # final destination of this episode segment

        bev_img = self._make_bev_image(
            bev_bg, bev_meta, past_poses, current_pose, goal_pose
        )

        if self.data_args.transform_train is not None:
            bev_img = self.data_args.transform_train(bev_img)

        bev_tensor, bev_thw = self.process_image_unified(bev_img)

        # ── Ego view (Image 2) ────────────────────────────────────────────
        ego_img = self._load_ego_frame(video, height, pitch_1, ep_id, start_frame_id)

        if self.data_args.transform_train is not None:
            ego_img = self.data_args.transform_train(ego_img)

        ego_tensor, ego_thw = self.process_image_unified(ego_img)

        images    = [bev_tensor, ego_tensor]
        grid_thws = [bev_thw,   ego_thw]

        # ── Prompt & GT ───────────────────────────────────────────────────
        # GT: original R2R/RxR instruction text (System 2 will receive this)
        chat_sources = [[
            {'from': 'human', 'value': self._build_prompt()},
            {'from': 'gpt',   'value': instruction},
        ]]

        # ── Tokenization ─────────────────────────────────────────────────
        grid_thw_merged = copy.deepcopy(grid_thws)
        if not isinstance(grid_thw_merged, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thws       = [grid_thws]
        grid_thw_merged = [
            t.prod() // self.data_args.image_processor.merge_size ** 2
            for t in grid_thw_merged
        ]

        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
        )

        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thws, dim=0) if grid_thws else None,
        )

        data_dict["position_ids"]   = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]
        data_dict["pixel_values"]   = torch.cat(images, dim=0)
        data_dict["image_grid_thw"] = torch.cat(
            [thw.unsqueeze(0) for thw in grid_thws], dim=0
        )

        return data_dict
