import copy
import itertools
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import transformers
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
from transformers.image_utils import to_numpy_array

from .rope2d import get_rope_index_2, get_rope_index_25
from .vlln_lerobot_dataset import VLLNDataset

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",  # noqa: F541
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}


R2R_125CM_0_30 = {
    "data_path": "traj_data/r2r",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}

R2R_125CM_0_45 = {
    "data_path": "traj_data/r2r",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 45,
}

R2R_60CM_15_15 = {
    "data_path": "traj_data/r2r",
    "height": 60,
    "pitch_1": 15,
    "pitch_2": 15,
}

R2R_60CM_30_30 = {
    "data_path": "traj_data/r2r",
    "height": 60,
    "pitch_1": 30,
    "pitch_2": 30,
}

RxR_125CM_0_30 = {
    "data_path": "traj_data/rxr",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}

RxR_125CM_0_45 = {
    "data_path": "traj_data/rxr",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 45,
}

RxR_60CM_15_15 = {
    "data_path": "traj_data/rxr",
    "height": 60,
    "pitch_1": 15,
    "pitch_2": 15,
}

RxR_60CM_30_30 = {
    "data_path": "traj_data/rxr",
    "height": 60,
    "pitch_1": 30,
    "pitch_2": 30,
}

SCALEVLN_125CM_0_30 = {
    "data_path": "traj_data/scalevln",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 30,
}

SCALEVLN_125CM_0_45 = {
    "data_path": "traj_data/scalevln",
    "height": 125,
    "pitch_1": 0,
    "pitch_2": 45,
}

SCALEVLN_60CM_30_30 = {
    "data_path": "traj_data/scalevln",
    "height": 60,
    "pitch_1": 30,
    "pitch_2": 30,
}

VLA_E2E_FRONT = {
    "data_path": "/home/work/.datasets/VLA-E2E-converted",  # vla_e2e/ 가 scene_id
    "height": 92,       # camera height 0.92m
    "pitch_1": 0,       # front camera
    "pitch_2": 0,       # no lookdown (single front camera)
}

# train/val 은 episodes.jsonl의 'split' 필드(step03_to_lerobot.py가 S2와 동일한
# md5 해시 split_for()로 기록)에서 그대로 읽는다 — 예전엔 68-episode 시절 기준
# 하드코딩 id set(_VLA_E2E_VAL_IDS={4,12,...} 등)을 썼는데, 이후 원본 에피소드
# 구성이 40->42로 바뀌면서 조용히 어긋난 적이 있어(패키징이 재실행될 때마다
# episode_index 배정이 바뀔 수 있음) 패키징 산출물에서 직접 읽도록 바꿈.
# (2026-07 park_forward 전량 제외 결정은 더 이상 적용하지 않음 — 당시 887샘플
# 기준으로는 park_forward가 22%였지만 지금 42-episode 기준으로는 38%라 통째로
# 빼면 데이터 손실이 너무 크다; 필요하면 sample_category 기반 다운웨이트로 대응.)
def _load_vla_e2e_split_ids():
    _meta = os.path.join(VLA_E2E_FRONT["data_path"], "vla_e2e", "meta", "episodes.jsonl")
    train_ids, val_ids = set(), set()
    if os.path.exists(_meta):
        with open(_meta) as _fp:
            for _line in _fp:
                _ep = json.loads(_line)
                (val_ids if _ep.get("split") == "val" else train_ids).add(_ep["episode_index"])
    return train_ids, val_ids


_VLA_E2E_TRAIN_IDS, _VLA_E2E_VAL_IDS = _load_vla_e2e_split_ids()

VLA_E2E_FRONT_TRAIN = {**VLA_E2E_FRONT, "episode_ids": _VLA_E2E_TRAIN_IDS}
VLA_E2E_FRONT_VAL   = {**VLA_E2E_FRONT, "episode_ids": _VLA_E2E_VAL_IDS}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "r2r_125cm_0_30": R2R_125CM_0_30,
    "r2r_125cm_0_45": R2R_125CM_0_45,
    "r2r_60cm_15_15": R2R_60CM_15_15,
    "r2r_60cm_30_30": R2R_60CM_30_30,
    "rxr_125cm_0_30": RxR_125CM_0_30,
    "rxr_125cm_0_45": RxR_125CM_0_45,
    "rxr_60cm_15_15": RxR_60CM_15_15,
    "rxr_60cm_30_30": RxR_60CM_30_30,
    "scalevln_125cm_0_30": SCALEVLN_125CM_0_30,
    "scalevln_125cm_0_45": SCALEVLN_125CM_0_45,
    "scalevln_60cm_30_30": SCALEVLN_60CM_30_30,
    "vla_e2e_front":       VLA_E2E_FRONT,
    "vla_e2e_front_train": VLA_E2E_FRONT_TRAIN,
    "vla_e2e_front_val":   VLA_E2E_FRONT_VAL,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
TRAJ_TOKEN_INDEX = 151667
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_TRAJ_TOKEN = "<traj>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:  # noqa: E722
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:  # noqa: E722
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>" * grid_thw_image[visual_replicate_index_image]  # noqa: F541
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + "<|video_pad|>" * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def prepare_sample(self, sample):
        """Per-fetch hook for task-specific online augmentation.

        The default keeps the loaded annotation untouched. Subclasses may return
        an augmented deep copy; the item loader then consistently uses that object
        for both visual inputs and conversations.
        """
        return sample

    def process_video(self, video_file):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 3
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(video_file)
                return decord_video
                if decord_video:
                    break
            except Exception as e:
                print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_file)
            return torchcodec_video
        except Exception as e:
            print(f"torchcodec attempt failed: {e}")

    def video_decord(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(images=None, videos=video, return_tensors="pt")
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [self.data_args.image_processor.temporal_patch_size / fps] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3  # noqa: F841
        num_final_retries = 30  # noqa: F841

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.prepare_sample(self.list_data_dict[i])
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        if "image" in sources[0]:
            image_folder = sources[0]["data_path"]
            image_file = sources[0]["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [os.path.join(image_folder, file) for file in image_file]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2 for merged_thw in grid_thw_merged
            ]
        if "video" in sources[0]:
            video_file = sources[0]["video"]
            video_folder = sources[0]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [os.path.join(video_folder, file) for file in video_file]
                    results = [self.process_video(file) for file in video_file]
                    video, video_grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, video_grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, video_grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]
            video_grid_thw_merged = copy.deepcopy(video_grid_thw)
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw_merged = [video_grid_thw_merged]
                video_grid_thw = [video_grid_thw]
            video_grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in video_grid_thw_merged
            ]
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(torch.stack(video_grid_thw, dim=0) if video_grid_thw else None),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        if "image" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer, grid_thw=grid_thw_merged)
            position_ids = torch.arange(0, data_dict["input_ids"].size(1)).view(1, -1).unsqueeze(0).expand(3, -1, -1)

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if "image" in sources[0]:
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat([thw.unsqueeze(0) for thw in grid_thw], dim=0)
        # video exist in the data
        elif "video" in sources[0]:
            data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
            data_dict["video_grid_thw"] = torch.cat([thw.unsqueeze(0) for thw in video_grid_thw], dim=0)

        return data_dict


def interpolate_and_resample_trajectory(absolute_trajectories, predict_step_num=None, step_threshold=0.05,
                                         resample_interval=0.1, xy_norm_scale=4.0):
    """step_threshold: 프레임 간 이동으로 인정할 최소 거리의 제곱값.
    r2r/rxr/scalevln처럼 이산적인 VLN-CE 액션(한 스텝 ~0.25m급) 기준으로 잡힌
    기본값(0.05)이며, VLA-E2E처럼 연속적이고 촘촘한 저속 주행 궤적(프레임당
    이동거리가 대부분 0.05 미만)에는 너무 커서 실제 이동이 전부 걸러지고
    상대 pose가 0으로 붕괴한다 — 호출부에서 데이터셋 스케일에 맞게 낮춰야 한다.

    xy_norm_scale: 원본 R2R(resample_interval=0.1m) 기준으로 캘리브레이션된 "*4"
    정규화 상수. VLA-E2E(resample_interval=1.0m, 10배 큰 스텝)에 그대로 적용하면
    diffusion 학습 타깃의 수치 범위가 R2R 사전학습 때보다 ~10배 커진다 — 실측:
    R2R per-step |xy|(*4 후) mean=0.4545 vs VLA-E2E(동일 *4) mean=4.93, 비율
    10.85배로 resample_interval 비율(10배)과 거의 정확히 일치. 모델이 사전학습
    때 배운 "이 정도가 정상 출력 크기"라는 감각과 우리가 요구하는 크기가 10배
    차이 나면서, 실제 관측된 예측 크기(ratio*GT reach ≈ 2.78m)가 공교롭게도
    R2R 기대치(0.4545)와 우리 목표(4.93)의 산술 평균(2.69m)에 가깝게 나옴 —
    사전학습 스케일 감각과 파인튜닝 목표 사이에서 타협하는 것으로 보이는 정황
    증거. VLA-E2E 호출부에서 xy_norm_scale=0.4(≈4/10)로 낮춰 학습 타깃 자체를
    R2R이 익숙한 수치 범위로 맞추고, 평가 시에는 그만큼 다시 나눠 실제 미터로
    환산한다 (S1_TRAJ_XY_SCALE_VLA_E2E 참고)."""
    start_point = np.array([[0.0, 0.0]])  # Avoid creating arrays repeatedly

    traj = absolute_trajectories[..., :2]
    # Vectorized filtering of valid steps (distance squared > step_threshold)
    steps = traj[1:] - traj[:-1]  # (T, 2)
    steps_sq = (steps**2).sum(axis=-1)  # (T,)
    mask = steps_sq > step_threshold  # (T,)

    # Filter and concatenate starting point
    filtered_traj = traj[1:][mask]  # (T, 2), where M is the number of filtered steps
    filtered_traj = np.concatenate([start_point, filtered_traj], axis=0)  # (T+1, 2)

    resampled_trajectories = smooth_and_resample_trajectory(
        filtered_traj, sample_length=predict_step_num + 1, interval=resample_interval
    )
    resampled_relative_poses = xy_to_delta_xyt(resampled_trajectories)

    resampled_relative_poses[:, 0:2] *= xy_norm_scale  # norm

    return resampled_trajectories, resampled_relative_poses


def get_trajectory_relative_to_frame(extrinsics, camera_deg=0):
    """
    Calculate trajectory poses (x, y, yaw) relative to a reference frame

    Args:
        extrinsics: Sequence of 4x4 extrinsic matrices [T_world2camera], shape: (n, 4, 4), numpy array
        camera_deg: Camera pitch angle

    Returns:
        relative_xyyaw: Pose sequence relative to the reference frame (x, y, yaw), shape: (n, 3), numpy array
    """
    # T_world2camera
    # Coordinate transformation matrices
    T_camera2robot = np.array(
        [[[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )

    T_robot2camera = np.array(
        [[[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )

    # Apply camera pitch angle transformation (30 degrees downward)
    if camera_deg is not None:
        camera_rad = np.radians(camera_deg)
        # Clockwise rotation around the x-axis of the level camera is downward view
        T_deg = np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, np.cos(-camera_rad), -np.sin(-camera_rad), 0.0],
                    [0.0, np.sin(-camera_rad), np.cos(-camera_rad), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=np.float32,
        )
        T_robot2camera = np.matmul(T_robot2camera, T_deg)
        T_camera2robot = np.linalg.inv(T_robot2camera)

    # Transform to robot coordinate system
    extrinsics_robot = np.matmul(extrinsics, T_camera2robot)

    # Get the transformation matrix of the reference frame and compute its inverse
    T_ref = extrinsics_robot[0]
    T_ref_inv = np.linalg.inv(T_ref)

    # Calculate transformations of all frames relative to the reference frame
    # T_relative = T_ref^{-1} * T_current
    relative_to_ref = np.matmul(T_ref_inv[np.newaxis, :, :], extrinsics_robot)

    # Extract relative poses
    relative_translations = relative_to_ref[:, :2, 3]  # (x, y), only take the xy-plane
    relative_yaws = np.arctan2(relative_to_ref[:, 1, 0], relative_to_ref[:, 0, 0])

    relative_xyyaw = np.concatenate((relative_translations, relative_yaws.reshape(-1, 1)), axis=-1)

    return relative_xyyaw


def get_trajectory_relative_to_frame_vla_e2e(pose_seq):
    """VLA-E2E(front, pitch=0) 전용 대체 구현.

    VLA-E2E의 `pose`(lerobot parquet의 `pose.92cm_0deg`)는 get_trajectory_relative_to_frame이
    기대하는 "프레임별 절대 world2cam"이 아니라, data_conversion/step03_to_lerobot.py에서
    `relative_pose(T_cur, T_next) = T_cur^{-1} @ T_next` (raw ego x,y,theta 기반)로 만든
    **"현재 프레임 -> 다음 프레임" 상대 SE(3) 변환**이다 (실측으로 확인: pose[i] 역행렬의
    translation 크기가 실제 프레임간 이동거리와 정확히 일치, world2cam 절대좌표와는 전혀 다름).

    절대좌표인 것처럼 get_trajectory_relative_to_frame(camera_deg=0)에 넣으면 이중으로
    잘못 계산되고, camera_deg=0 자체의 축퇴 문제(x/yaw가 0으로 나옴)까지 겹쳐서 궤적이
    완전히 깨진다.

    올바른 방법: pose[i] = T_i^{-1} @ T_{i+1} 이므로, 순서대로 그냥 행렬곱하면
    텔레스코핑으로 pose[ref] @ pose[ref+1] @ ... @ pose[j-1] = T_ref^{-1} @ T_j
    (frame j를 frame ref=pose_seq[0] 기준으로 본 상대 pose)가 나온다.
    raw ego (x,y) 직접 비교로 소수점까지 정확히 일치함을 검증함.

    Args:
        pose_seq: 상대 SE(3) 변환 시퀀스, shape (n, 4, 4) (pose_seq[i] = frame i -> i+1)
    Returns:
        relative_xyyaw: (n, 3), pose_seq[0] 프레임 기준 상대 (x, y, yaw)
    """
    n = len(pose_seq)
    cum = np.eye(4)
    out = np.zeros((n, 3))
    for i in range(n - 1):
        cum = cum @ pose_seq[i]
        out[i + 1, 0] = cum[0, 3]
        out[i + 1, 1] = cum[1, 3]
        out[i + 1, 2] = np.arctan2(cum[1, 0], cum[0, 0])
    return out


def filter_backward_points(discrete_traj_pose, angle_thresh=np.pi / 2, min_step=0.05):
    """연속된 두 스텝의 '이동 방향'끼리 비교해서, 직전 스텝 방향 대비 갑자기
    반대(디폴트 90도 초과)로 튀는 점을 제거한다.

    처음엔 이동 방향을 heading(yaw)과 비교하는 방식을 시도했으나, yaw의
    기준축이 위치(x,y) 기준축과 체계적으로 ~90도 어긋나 있어(raw anno의
    coordinate_schema=bench2drive_left_hand 때문으로 추정) 정상 주행도
    대부분 걸러지는 문제가 있었다. yaw를 아예 쓰지 않고 위치만으로 판단하면
    이 축 오프셋 문제를 피할 수 있다. 전체 시작->끝 직선 기준(progress) 방식도
    시도했으나 U턴/훅처럼 실제로 휘어지는 경로가 대량으로 잘려나가 폐기했다
    — 스텝끼리의 로컬 비교라 완만한 커브는 걸리지 않고 급격한 역전만 잡는다.
    항상 시작점(index 0)은 유지한다.

    min_step(기존 1e-9)이 실제 GPS/위치 노이즈 스케일(mm~cm급)보다 훨씬 작아서,
    정지 구간(에피소드 시작 직후 등)의 노이즈 스텝이 그대로 prev_step_dir의
    기준으로 채택되는 버그가 있었다 — 그 뒤로는 실제 주행 벡터들이 이 노이즈
    방향과 비교되어 대부분 '역전'으로 걸러지고, 한참 뒤 우연히 그 노이즈
    방향과 90도 이내로 맞아떨어지는 지점 하나만 살아남아 물리적으로 불가능한
    큰 점프(실측 43m)가 생겼다(val idx=88, episode 20260521/data9에서 발견 —
    GT 궤적이 역주행하는 것처럼 보이는 원인이었음). interpolate_and_resample_
    trajectory의 노이즈 컷오프(0.05, 실측 프레임당 이동 0.5~0.7m 대비 mm급
    노이즈를 거르는 값)와 동일한 값을 재사용한다."""
    xy = discrete_traj_pose[:, :2]
    keep = np.ones(len(xy), dtype=bool)
    prev_step_dir = None
    prev_idx = 0
    for i in range(1, len(xy)):
        step = xy[i] - xy[prev_idx]
        step_norm = np.linalg.norm(step)
        if step_norm < min_step:
            keep[i] = False
            continue
        step_dir = np.arctan2(step[1], step[0])
        if prev_step_dir is not None:
            diff = (step_dir - prev_step_dir + np.pi) % (2 * np.pi) - np.pi
            if abs(diff) > angle_thresh:
                keep[i] = False
                continue
        prev_step_dir = step_dir
        prev_idx = i
    return discrete_traj_pose[keep]


# ── VLA-E2E 카메라 투영 (S1 로컬 pixel-goal 재계산용) ───────────────────────────
# scripts/qual_s1_s2_val.py의 실측 캘리브레이션과 동일 (고정 하드웨어, 여러 날짜/
# 에피소드에서 동일함을 확인함). 원본 pixel_goal(최종 목적지 또는 화면에 보이는
# 가장 먼 지점)은 S1이 실제로 학습하는 로컬 궤적 타깃
# (resample_interval*predict_step_num, VLA-E2E는 최대 ~5m)과 거리 스케일이 전혀
# 안 맞아(수십 m 차이) S1 conditioning용 goal로는 부적합 — 로컬 타깃 지점을 직접
# 카메라에 투영해 action(pixel-goal)을 다시 계산한다.
# S1(nextdit_async) DiT의 실제 물리적 궤적 horizon — n_query=4, resample_interval=1.0m
# (총 ~5m). 이 값은 S1의 아키텍처 상수이지, 데이터셋을 로드하는 스크립트마다 다른
# self.predict_step_num(예: S2 학습은 N_QUERY=32, pose 리샘플링용 서로 다른 개념)과
# 무관하게 항상 고정이어야 한다 — S2/S1 모두 "같은 물리적 지점"을 pixel-goal로
# 학습하게 하려면 반드시 이 상수를 써야 한다 (self.predict_step_num을 쓰면 S2는
# 엉뚱하게 32m 앞을 가리키는 라벨로 학습됨).
S1_TRAJ_PREDICT_STEP_NUM = 4
S1_TRAJ_RESAMPLE_INTERVAL = 1.0

# interpolate_and_resample_trajectory의 `resampled_relative_poses[:, 0:2] *= 4`는
# 원본 DualVLN(R2R, resample_interval=0.1m) 기준 xy/theta 스케일 균형을 맞춘
# 상수라 VLA-E2E(resample_interval=1.0m, xy가 10배 더 넓게 퍼짐)에는 안 맞는다.
# 실측(val onscreen_long 200스텝): |xy|(*4 적용 후) mean~4.93 vs |dtheta|(raw rad)
# mean~0.46 -> ~10.7배 격차. Diffusion/flow-matching의 noise가 채널별로 동일한
# unit-variance라 dtheta는 signal-to-noise比가 xy 대비 훨씬 나빠 사실상 학습이
# 안 되고(예측 웨이포인트 heading이 실제 이동 방향과 평균 45~53도 어긋남,
# eval_s1_heading_channel_consistency.py로 확인) 화살표가 엉뚱한 방향을 가리키는
# 원인이 된다. xy와 comparable한 스케일로 올려 SNR을 맞춘다 (VLA-E2E 전용).
#
# *** S1_TRAJ_XY_SCALE_VLA_E2E 도입에 따라 10.0 -> 1.0으로 재조정 ***
# (interpolate_and_resample_trajectory의 xy_norm_scale 주석 참고) R2R 사전학습
# 스케일(resample_interval=0.1m)과 VLA-E2E(1.0m, 10배 큰 스텝)의 절대 크기 차이를
# 없애려고 xy_norm_scale을 4.0->0.4로(10배 축소) 낮췄다. 기존 THETA_SCALE=10은
# 그 10배 커진 xy(4.93)에 맞춰 theta를 끌어올린 값이었으므로, xy를 10배 줄이면
# theta도 같은 비율로 10배 줄여야 둘의 상대적 SNR 균형(약 1.07:1, 실측으로 검증된
# 값)이 그대로 유지된다 — xy_norm_scale과 THETA_SCALE을 항상 같은 비율로
# 맞춰야 함(둘 다 10배 축소 = xy:theta 균형 불변, R2R 스케일에도 맞춰짐).
S1_TRAJ_THETA_SCALE = 1.0
# VLA-E2E 전용 xy 정규화 상수 — R2R용 기본값(4.0)의 1/10. 실측: R2R per-step
# |xy|(*4 적용 후) mean=0.4545, VLA-E2E per-step raw |xy| mean~1.23(추정, *4 적용시
# 4.93이 나오므로) -> *0.4 적용 시 mean~0.493, R2R과 거의 동일한 수치 범위가 된다.
S1_TRAJ_XY_SCALE_VLA_E2E = 0.4

# Exp A(train/inference 프레임 정렬) on/off 스위치 — raw orientation 기반 theta 수정을
# 단독으로 검증하기 위해 일시적으로 False로 둠.
S1_USE_FRAME_ALIGNMENT_FIX = True

# R2R(원본 사전학습)과 VLA-E2E(우리 데이터)의 forward/lateral 축이 정반대임을 실측으로 확인함:
#   R2R:      dx(forward, 지배적) mean=+0.1227  |  dy(lateral) mean≈0
#   VLA-E2E:  dx(lateral) mean≈0                |  dy(forward, 지배적) mean=-1.19
# ego_to_se3(raw x,y,theta 기반)의 표준 회전행렬 자체는 정상이지만, raw anno의 theta
# 기준축이 위치축과 90도 어긋나 있어(S1_RAW_YAW_CORRECTION_RAD 참고) 그 여파로 합성된 (x,y)의
# "전진 축"도 R2R과 반대로(y축) 나온 것으로 보인다.
#
# *** 중요: lateral(좌우) 축은 회전이 아니라 반전(reflection)이 필요함 ***
# 처음엔 steer(CARLA 컨벤션 양수=우회전)와만 대조해서 "90도 순수 회전(new_x=-old_y,
# new_y=old_x)으로 충분하다"고 결론 냈었다. 하지만 이건 우리 데이터 안에서의
# "자체 일관성"만 확인한 것이었고, R2R 사전학습이 실제로 y채널에 어떤 부호를
# 쓰는지는 대조하지 않은 상태였다. VLN-CE의 진짜 R2R 원본 데이터(Habitat
# activeview_v0, position+quaternion+next_waypoint)로 get_trajectory_relative_to_
# frame()에 깨끗한 synthetic 케이스(순수 좌/우 이동만)를 넣어 직접 검증한 결과:
# R2R 컨벤션은 "우측 이동 -> y=+1, 좌측 이동 -> y=-1"이다 (habitat 우수 좌표계,
# forward=-Z, up=+Y, right=+X 표준 컨벤션으로 재구성해 확인).
# 그런데 위 90도 순수 회전(new_y=old_x)을 실측 steer와 대조하면 우회전 구간에서
# new_dy<0, 좌회전 구간에서 new_dy>0 — 이건 R2R과 정확히 반대 부호다. 순수 회전
# 2가지(new_y=+old_x 또는 new_y=-old_x, 둘 다 forward 축은 new_x=-old_y로 동일)
# 중 어느 쪽도 "전진 축 정렬 + lateral 부호 일치"를 동시에 만족할 수 없다는 것도
# 확인함 — CARLA/UE4(raw anno의 coordinate_schema=bench2drive_left_hand, 즉
# 좌수 좌표계)와 Habitat/R2R(우수 좌표계)의 handedness 자체가 다르기 때문에
# 순수 회전이 아니라 반사(reflection, det=-1)가 필요하다. 수정된 변환
# new_x=-old_y, new_y=-old_x 를 실측 steer로 재검증: 우회전 -> new_dy>0(100%),
# 좌회전 -> new_dy<0(94%) — R2R과 부호 일치 확인함.
S1_USE_AXIS_ROTATION_FIX = True


def rotate_xy_for_pretrain_alignment(dx, dy):
    """VLA-E2E의 (전진=y, lateral=x) 축을 R2R의 (전진=x, lateral=y) 컨벤션에 맞게 정렬.
    forward는 90도 회전(new_x=-old_y)이지만 lateral은 반사(new_y=-old_x)가 필요하다
    — CARLA(좌수 좌표계)와 R2R/Habitat(우수 좌표계)의 handedness가 다르기 때문
    (위 S1_USE_AXIS_ROTATION_FIX 주석의 실측 검증 참고).

    dx/dy가 같은 배열의 컬럼 뷰(예: arr[:,0], arr[:,1])일 수 있으므로 반드시 값을 복사한
    뒤 반환한다 — 호출부에서 `arr[:,0]=nx; arr[:,1]=ny` 식으로 순차 대입하면, 복사 없이
    뷰를 그대로 반환할 경우 첫 대입이 두 번째 반환값이 참조하던 메모리를 덮어써 버려
    dx==dy로 뒤섞이는 aliasing 버그가 생긴다 (실측으로 확인함: 90도 회전 도입 직후
    모든 GT dx가 -dy와 정확히 같아지는 버그 발생 — 이 함수의 return이 원인이었음)."""
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    return (-dy).copy(), (-dx).copy()


def unrotate_xy_for_pretrain_alignment(new_dx, new_dy):
    """rotate_xy_for_pretrain_alignment의 역변환 — 학습/추론에서 나온 (dx,dy)를 실제
    카메라 투영/시각화에 쓰는 원래 VLA-E2E 좌표계로 되돌릴 때 사용. 반사 변환은 스스로가
    자기 자신의 역변환이 아니라는 점에 주의(forward/lateral이 다른 부호 규칙이라 완전히
    대칭은 아님) — 실제로 검산하면 (-new_dy, -new_dx)가 정확한 역변환이다. 위와 동일한
    이유로 항상 복사본을 반환한다."""
    new_dx = np.asarray(new_dx)
    new_dy = np.asarray(new_dy)
    return (-new_dy).copy(), (-new_dx).copy()


def rotate_theta_for_pretrain_alignment(delta_theta):
    """Map VLA-E2E heading deltas to the reflected InternVLN/R2R XY frame.

    XY uses (x_pre, y_pre)=(-y_vla, -x_vla). Therefore the first absolute
    heading maps as theta_pre=-theta_vla-pi/2, while later heading deltas
    change sign under the reflection.
    """
    theta = np.asarray(delta_theta).copy()
    if theta.shape[-1]:
        theta[..., 0] = (-theta[..., 0] - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
        theta[..., 1:] *= -1
    return theta


def unrotate_theta_for_pretrain_alignment(delta_theta):
    """Inverse heading mapping; this reflection is self-inverse."""
    return rotate_theta_for_pretrain_alignment(delta_theta)


# VLA-E2E 학습 데이터의 좌/우회전 불균형(steer 기준 실측: 우회전 444 vs 좌회전 52,
# 약 8.5:1) 때문에 모델이 좌회전을 잘 못 배우고, 특히 재계획(t,t+k) 상황에서 GT가
# 명확히 왼쪽인데도 오른쪽으로 예측하는 사례가 반복 관찰됨(val idx=49 등). 좌우
# 반전 augmentation으로 대응: 이미지+궤적+pixel-goal+지시문을 전부 거울상으로
# 뒤집으면, 원래 우회전이던 샘플이 "진짜" 좌회전 샘플이 된다(가짜 데이터를 만드는
# 게 아니라 동일한 물리 현상을 반대편에서 본 것과 동치) — 원본 비율과 무관하게
# 증강 후 분포가 정확히 50:50이 된다.
S1_USE_LR_FLIP_AUGMENT = True

_LR_FLIP_WORD_RE = re.compile(r'\b(left|right)\b', re.IGNORECASE)


def flip_left_right_text(text):
    """지시문의 'left'/'right' 단어를 서로 맞바꾼다 (대소문자 보존, 단어 경계 기준
    — 'leftover' 같은 부분 문자열 오탐 방지). 두 단어가 한 문장에 여러 번 나와도
    치환 순서에 따라 이중 치환되지 않도록 매치마다 원래 대소문자 패턴을 보고 반대
    단어를 그 패턴에 맞춰 생성한다."""
    def _swap(m):
        word = m.group(0)
        replacement = 'right' if word.lower() == 'left' else 'left'
        if word.isupper():
            return replacement.upper()
        if word[0].isupper():
            return replacement.capitalize()
        return replacement
    text = re.sub(r'좌회전|우회전', lambda m: '우회전' if m.group(0) == '좌회전' else '좌회전', text)
    return _LR_FLIP_WORD_RE.sub(_swap, text)


# get_trajectory_relative_to_frame_vla_e2e()가 SE(3) 회전행렬에서 뽑는 진짜
# orientation(out[:,2])은 raw anno의 coordinate_schema 문제로 위치차분 방향(atan2(dy,dx))과
# 축이 어긋나 있다 — 실측(4308 스텝): raw_yaw - pos_dir = mean 89.57deg, std 1.60deg.
# std가 1.6도로 매우 작아 노이즈가 아니라 거의 정확히 고정된 90도 오프셋임을 확인함.
# xy_to_delta_xyt는 지금까지 이 raw orientation을 버리고 위치차분(atan2)만으로 theta를
# 다시 만들었는데, 이건 저속/정지 근처 구간에서 위치가 조금만 흔들려도 각도가 크게
# 튀는 수치적 불안정성이 있다. raw orientation(90도 보정)을 쓰면 그 불안정성을 피할 수 있다.
S1_RAW_YAW_CORRECTION_RAD = -np.pi / 2


def resample_theta_from_raw_orientation(discrete_traj_pose, predict_step_num, resample_interval=1.0):
    """xy_to_delta_xyt가 위치차분(atan2)으로 theta를 재계산하는 대신, discrete_traj_pose[:,2]에
    이미 있는 진짜 orientation(SE(3) 회전행렬 기반, S1_RAW_YAW_CORRECTION_RAD로 축 보정)을
    각 리샘플 지점(xy_to_delta_xyt와 동일한 예정 거리)에 가장 가까운 원본 프레임에서
    최근접 탐색으로 가져와 delta_yaw를 만든다.

    Args:
        discrete_traj_pose: (n,3) get_trajectory_relative_to_frame_vla_e2e + filter_backward_points 결과
        predict_step_num: 리샘플 스텝 수 (xy_to_delta_xyt와 동일하게 이 값+1개 지점 기준 delta)
        resample_interval: 스텝당 거리(m)
    Returns:
        delta_yaw: (predict_step_num,) — xy_to_delta_xyt의 delta_yaw 컬럼과 동일한 컨벤션
                   (index 0=첫 지점의 절대 각도, 이후는 이전 대비 회전량, [-pi,pi] wrap)
    """
    xy = discrete_traj_pose[:, :2]
    raw_yaw = discrete_traj_pose[:, 2] + S1_RAW_YAW_CORRECTION_RAD

    if len(xy) < 2:
        return np.zeros(predict_step_num)

    seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_len)])

    target_distances = np.linspace(0, resample_interval * predict_step_num, predict_step_num + 1)
    resampled_yaw = np.zeros(predict_step_num + 1)
    resampled_xy = np.zeros((predict_step_num + 1, 2))
    for i, td in enumerate(target_distances):
        nearest_idx = int(np.argmin(np.abs(cum_dist - td)))
        resampled_yaw[i] = raw_yaw[nearest_idx]
        resampled_xy[i] = xy[nearest_idx]

    # resampled_yaw[0]은 "기준 프레임이 자기 자신을 보는 각도"라 항상 0(+보정값)인 인공물이지,
    # 차량의 실제 초기 진행방향과는 무관하다 — 첫 delta는 xy_to_delta_xyt와 동일하게 위치
    # 기반 절대방향(atan2)으로 앵커링하고, 이후(index>=1)만 실제 orientation 증분을 쓴다.
    first_vec = resampled_xy[1] - resampled_xy[0]
    if np.linalg.norm(first_vec) < 1e-6:
        first_abs = 0.0
    else:
        first_abs = np.arctan2(first_vec[1], first_vec[0])

    delta_yaw = np.diff(resampled_yaw)  # length predict_step_num
    delta_yaw[0] = first_abs
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
    return delta_yaw


VLA_E2E_IMG_W, VLA_E2E_IMG_H = 1920, 1080
_VLA_E2E_CAM_K = np.array([
    [983.4798, 0.0,     1006.9],
    [0.0,      984.759, 490.492],
    [0.0,      0.0,     1.0],
])
_VLA_E2E_T_EGO2CAM = np.array([
    [0.0, -0.9998, 0.0218, -1.0498],
    [1.0,  0.0,     0.0,    0.0],
    [0.0,  0.0218,  0.9998, -0.8973],
    [0.0,  0.0,     0.0,    1.0],
])


def project_relative_xy_vla_e2e(x, y, z=0.0):
    """get_trajectory_relative_to_frame_vla_e2e 컨벤션의 상대 (x,y)[meter, ego-local]를
    기준 프레임 이미지 픽셀 (u,v)로 투영. 카메라 뒤(투영 불가)면 None."""
    p_ego = np.array([x, y, z, 1.0])
    p_cam = _VLA_E2E_T_EGO2CAM @ p_ego
    if p_cam[0] <= 1e-3:
        return None
    pt_ocv = np.array([p_cam[1], -p_cam[2], p_cam[0]])
    pt_img = _VLA_E2E_CAM_K @ pt_ocv
    return pt_img[0] / pt_img[2], pt_img[1] / pt_img[2]


def compute_local_pixel_goal_vla_e2e(pose_seq, predict_step_num, resample_interval=1.0):
    """S1이 실제로 학습하는 로컬 궤적 타깃(최대 resample_interval*predict_step_num, ~5m)
    지점을 카메라에 투영해 (action, stop_like) 반환.

    5m 지점 자체가 화면 밖(옆)이면, step02_gen_goal_pixels.py의 "pulled" 전략과 동일하게
    5m 지점에서 현재 프레임 쪽으로 거꾸로 훑어 화면에 보이는 가장 먼 지점으로 대체한다
    (원본이 에피소드 전체 스케일에서 하던 걸 S1의 로컬 5m 윈도우 스케일로 그대로 적용).

    action: (u,v) 튜플(화면 안, pull 포함) 또는 0 (윈도우 내 어떤 지점도 화면에 안 보임 -> STOP 취급)
    stop_like=True면 호출부에서 sample_category도 near_zero로 맞춰야 한다.

    step02_gen_goal_pixels.py의 "pulled" 전략은 프레임 단위로 촘촘하게 훑는다 —
    resample된 4개 점만으로 훑으면 후보가 너무 성겨서 실제로는 보이는 점이 있는데도
    놓치는 경우가 생김. raw discrete(프레임 단위) 점으로 훑되, filter_backward_points는
    쓰지 않는다(역방향 제거가 오히려 근처 유효 프레임을 지워 오탐을 늘림 — 확인됨)."""
    discrete = get_trajectory_relative_to_frame_vla_e2e(pose_seq)
    xy_full = discrete[:, :2]
    dist_from_start = np.linalg.norm(xy_full - xy_full[0], axis=1)
    max_dist = resample_interval * predict_step_num
    local_xy = xy_full[dist_from_start <= max_dist]
    if len(local_xy) == 0:
        return 0, True

    # 가장 먼 지점부터 거꾸로 훑어 화면에 보이는 첫 지점 사용 ("pulled"와 동일한 아이디어)
    for x, y in local_xy[::-1]:
        proj = project_relative_xy_vla_e2e(x, y)
        if proj is None:
            continue  # 카메라 뒤 -> 더 가까운 점 시도
        u, v = proj
        if 0 <= u < VLA_E2E_IMG_W and 0 <= v < VLA_E2E_IMG_H:
            return (int(round(u)), int(round(v))), False
    return 0, True  # 윈도우 내 어떤 지점도 화면에 안 보임 -> STOP 취급


VLA_E2E_LOOKDOWN_BOTTOM_FRAC = 0.6  # 이미지 아래쪽 몇 %를 "lookdown"으로 쓸지


def pseudo_lookdown_crop(pil_image):
    """VLA-E2E는 실제 아래방향(lookdown) 카메라가 없어 pitch_2==pitch_1(같은 전방 이미지)이다
    — 그대로 두면 memory_encoder가 "두 뷰"를 완전히 동일한 이미지로 받아 아무 정보도 못 얻는다.
    새 카메라 없이, 같은 이미지의 아래쪽(근거리 지면 위주)만 크롭+리사이즈해 pitch_1(전방
    히스토리)과 실제로 다른 시야를 인위적으로 만든다."""
    w, h = pil_image.size
    top = int(h * (1 - VLA_E2E_LOOKDOWN_BOTTOM_FRAC))
    return pil_image.crop((0, top, w, h)).resize((w, h), Image.BILINEAR)


from scipy.interpolate import CubicSpline


def smooth_and_resample_trajectory(points, sample_length=33, interval=0.1):
    total_distance = sample_length * interval  # Total sampling length

    if len(points) == 0:
        return np.zeros((sample_length, 2))

    if len(points) == 1:
        return np.tile(points[0], (sample_length, 1))

    # Calculate cumulative distance of the original trajectory
    diff = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diff**2, axis=1))
    cumulative_distances = np.cumsum(segment_lengths)
    cumulative_distances = np.insert(cumulative_distances, 0, 0)  # Starting point distance is 0

    # Use cubic spline interpolation for smoothing
    if len(points) > 3:  # At least 4 points are needed for cubic spline interpolation
        # Construct cubic splines using cumulative distance as the parameter
        cs_x = CubicSpline(cumulative_distances, points[:, 0])
        cs_y = CubicSpline(cumulative_distances, points[:, 1])

        # Perform dense sampling within the original cumulative distance range
        dense_distances = np.linspace(0, cumulative_distances[-1], max(50, len(points) * 2))
        x_smooth = cs_x(dense_distances)
        y_smooth = cs_y(dense_distances)
        smoothed_points = np.column_stack((x_smooth, y_smooth))

        # Recalculate cumulative distance of the smoothed trajectory
        smooth_diff = np.diff(smoothed_points, axis=0)
        smooth_segment_lengths = np.sqrt(np.sum(smooth_diff**2, axis=1))
        smooth_cumulative_distances = np.cumsum(smooth_segment_lengths)
        smooth_cumulative_distances = np.insert(smooth_cumulative_distances, 0, 0)
    else:
        # Too few points for cubic spline interpolation, use original points directly
        smoothed_points = points
        smooth_cumulative_distances = cumulative_distances

    # Target sampling point distances
    target_distances = np.linspace(0, total_distance, sample_length)

    # Initialize result array
    resampled = np.zeros((sample_length, 2))

    # Interpolate for each target distance
    for i, target_dist in enumerate(target_distances):
        # If target distance exceeds total trajectory length, use the last point
        if target_dist >= smooth_cumulative_distances[-1]:
            resampled[i] = smoothed_points[-1]
            continue

        # Find the line segment where the target distance is located
        segment_idx = np.searchsorted(smooth_cumulative_distances, target_dist, side='right') - 1

        # Calculate interpolation ratio
        start_dist = smooth_cumulative_distances[segment_idx]
        end_dist = smooth_cumulative_distances[segment_idx + 1]
        t = (target_dist - start_dist) / (end_dist - start_dist)

        # Linear interpolation
        resampled[i] = smoothed_points[segment_idx] + t * (
            smoothed_points[segment_idx + 1] - smoothed_points[segment_idx]
        )

    return resampled


def xy_to_delta_xyt(xy_actions):
    """
    Compute (dx, dy, delta_yaw) where dx, dy in global frame and delta_yaw is heading difference.

    Args:
        xy_actions: [N, 2] array of absolute positions

    Returns:
        delta_xyt: [N-1, 3] array
    """
    vectors = np.diff(xy_actions, axis=0)  # [N-1, 2]
    yaw = np.arctan2(vectors[:, 1], vectors[:, 0])  # [N-1] yaw angles w.r.t x-axis

    delta_yaw = np.diff(yaw)  # [N-2]
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]

    # prepend first yaw (absolute angle of first segment) as delta_yaw[0]
    delta_yaw = np.concatenate([[yaw[0]], delta_yaw])  # now length = N-1

    delta_xyt = np.concatenate([vectors, delta_yaw[:, None]], axis=1)
    return delta_xyt


def clip_or_pad(arr, fixed_len):
    T, D = arr.shape
    if T >= fixed_len:
        return arr[:fixed_len]
    else:
        pad = np.zeros((fixed_len - T, D), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)


def get_annotations_from_lerobot_data(data_path, setting):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pyarrow.parquet as pq

    annotations = {
        "axis_align_matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "episodes": [],
    }
    scene_ids = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    def process_scene(scene_id):
        scene_path = os.path.join(data_path, scene_id)
        episodes = read_jsonl(os.path.join(scene_path, "meta", "episodes.jsonl"))
        scene_annotations = []

        for ep in episodes:
            ep_id = ep["episode_index"]
            ep_instructions = ep["tasks"][0].split("<INSTRUCTION_SEP>")
            ep_len = ep["length"]
            parquet_path = os.path.join(
                scene_path, "data", f"chunk-{ep_id // 1000:03d}", f"episode_{ep_id:06d}.parquet"
            )

            table = pq.read_table(parquet_path)
            df = table.to_pandas()

            ep_actions = df["action"].tolist()

            pose_key = f"pose.{setting}"
            goal_key = f"goal.{setting}"
            relative_goal_frame_id_key = f"relative_goal_frame_id.{setting}"

            if pose_key in df.columns and goal_key in df.columns and relative_goal_frame_id_key in df.columns:
                ep_poses = df[pose_key].apply(lambda x: x.tolist()).tolist()
                ep_pixel_goals = [
                    [df[relative_goal_frame_id_key][idx].tolist(), df[goal_key][idx].tolist()] for idx in range(len(df))
                ]
            else:
                print(f"Warning: Missing data for setting {setting} in episode {ep_id}, filling with defaults.")

            assert len(ep_actions) == ep_len, f"Action length mismatch in episode {ep_id}"

            for ep_instruction in ep_instructions:
                episode = {
                    "id": ep_id,
                    "video": f"{data_path}/{scene_id}/videos/chunk-{ep_id // 1000:03d}",
                    "instructions": ep_instruction,
                    "actions": ep_actions,
                    "length": ep_len,
                    f"poses_{setting}": ep_poses,
                    "pixel_goals": ep_pixel_goals,
                }
                scene_annotations.append(episode)

        return scene_annotations

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_scene, scene_id): scene_id for scene_id in scene_ids}
        for future in as_completed(futures):
            scene_id = futures[future]
            try:
                scene_annotations = future.result()
                annotations["episodes"].extend(scene_annotations)
            except Exception as e:
                print(f"Error processing scene {scene_id}: {e}")

    return annotations


class NavPixelGoalDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(NavPixelGoalDataset, self).__init__()
        dataset = data_args.vln_dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        self.sample_step = data_args.sample_step
        self.predict_step_num = data_args.predict_step_num
        self.pixel_goal_only = data_args.pixel_goal_only
        self.num_future_steps = data_args.num_future_steps
        # pixel_goal_only=True에서도 STOP 샘플(stop_list)을 함께 학습에 포함할지 여부.
        # turn_list는 건드리지 않음 — pixel_goal_only 자체를 끄면(=False) 기존처럼
        # turn_list+stop_list가 모두 섞여 다른(S2 등) 소비자의 동작이 바뀌므로,
        # S1 전용으로 stop_list만 추가하는 별도 플래그를 둔다.
        self.include_stop = getattr(data_args, 'include_stop', False)

        self.list_data_dict = []

        for data in dataset_list:
            sampling_rate = data.get("sampling_rate", 1.0)
            height = data.get("height", None)
            pitch_1 = data.get("pitch_1", None)
            pitch_2 = data.get("pitch_2", None)

            data_path = data['data_path']
            setting = f'{height}cm_{pitch_2}deg'
            episode_ids = data.get("episode_ids", None)  # None이면 전체 사용
            annotations = get_annotations_from_lerobot_data(data_path, setting)

            pixel_goal_list = []
            turn_list = []
            stop_list = []
            list_data_dict = []
            for item in annotations['episodes']:
                ep_id = item['id']
                if episode_ids is not None and ep_id not in episode_ids:
                    continue
                instruction = item['instructions']
                video = item['video']
                actions = item['actions'][1:] + [0]
                pixel_goals = item['pixel_goals']
                poses = item[f'poses_{height}cm_{pitch_2}deg']

                actions_len = len(actions)
                if actions_len < 4:
                    continue

                if height == 92:
                    # VLA-E2E: 프레임 수(sample_step) 대신 실거리 기준으로 윈도우 시작점을
                    # 뽑는다 — S1의 실제 로컬 타깃(~5m)만큼 떨어지게 해서 연속 윈도우끼리
                    # 과도하게 겹치는 것을 줄인다. poses[i] = "frame i -> i+1" 상대 SE(3)라
                    # 각 스텝의 translation 크기를 그대로 누적하면 실제 주행 거리가 나온다.
                    step_dists = np.array([np.linalg.norm(np.asarray(p)[:2, 3]) for p in poses])
                    cum_dist = np.concatenate([[0.0], np.cumsum(step_dists)])
                    target_interval = S1_TRAJ_RESAMPLE_INTERVAL * S1_TRAJ_PREDICT_STEP_NUM  # ~5m
                    start_frame_ids = [0]
                    last_dist = 0.0
                    for _idx in range(1, len(cum_dist)):
                        if cum_dist[_idx] - last_dist >= target_interval:
                            start_frame_ids.append(_idx)
                            last_dist = cum_dist[_idx]
                else:
                    num_rounds = actions_len // self.sample_step
                    start_frame_ids = [n * self.sample_step for n in range(num_rounds + 1)]

                for start_frame_id in start_frame_ids:
                    if start_frame_id == actions_len or start_frame_id == actions_len - 1:
                        continue
                    action_flag = actions[start_frame_id]
                    pixel_goal = pixel_goals[start_frame_id]
                    if pixel_goal[0] == -1:
                        if action_flag == 1:
                            continue
                        else:
                            end_frame_id = min(actions_len, start_frame_id + self.num_future_steps)
                            turn_actions = []
                            for id in range(start_frame_id, end_frame_id):
                                if actions[id] == 1:
                                    break
                                turn_actions.append(actions[id])
                            turn_list.append(
                                (
                                    ep_id,
                                    data_path,
                                    video,
                                    height,
                                    pitch_1,
                                    pitch_2,
                                    instruction,
                                    (start_frame_id, start_frame_id + 1),
                                    turn_actions,
                                    None,
                                )
                            )
                    else:
                        goal_len = pixel_goal[0]
                        if goal_len < 3:
                            continue
                        action = pixel_goal[1]
                        pose = poses[start_frame_id : start_frame_id + goal_len + 1]
                        pixel_goal_list.append(
                            (
                                ep_id,
                                data_path,
                                video,
                                height,
                                pitch_1,
                                pitch_2,
                                instruction,
                                (start_frame_id, start_frame_id + goal_len + 1),
                                action,
                                pose,
                            )
                        )

                # include_stop=True일 때만 실제 pose 윈도우를 채워 STOP 직전 구간의
                # (거의 0에 가까운) 궤적을 S1 DiT 타깃으로 쓸 수 있게 한다.
                # include_stop=False(기본값)면 기존과 동일하게 pose=None 유지.
                if self.include_stop:
                    stop_window = min(self.predict_step_num, actions_len - 1)
                    stop_start = actions_len - 1 - stop_window
                    stop_window_range = (stop_start, actions_len)
                    stop_pose = poses[stop_start:actions_len]
                else:
                    stop_window_range = (actions_len - 1, actions_len)
                    stop_pose = None
                stop_list.append(
                    (
                        ep_id,
                        data_path,
                        video,
                        height,
                        pitch_1,
                        pitch_2,
                        instruction,
                        stop_window_range,
                        0,
                        stop_pose,
                    )
                )

            list_data_dict = pixel_goal_list
            rank0_print(len(turn_list), len(pixel_goal_list), len(stop_list))
            if not self.pixel_goal_only:
                list_data_dict += turn_list
                list_data_dict += stop_list * 5
            elif self.include_stop:
                list_data_dict += stop_list * 5
            if sampling_rate < 1.0:
                list_data_dict = random.sample(list_data_dict, int(len(list_data_dict) * sampling_rate))
                print(f"sampling {len(list_data_dict)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")

            self.list_data_dict.extend(list_data_dict)

        self.num_history = data_args.num_history
        self.idx2actions = {0: 'STOP', 1: "↑", 2: "←", 3: "→", 5: "↓"}
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]
        self.data_args = data_args
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.list_data_dict)

    def process_image_unified(self, image):
        processor = copy.deepcopy(self.data_args.image_processor)
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def preprocess_depth_image_v2(
        self, depth_image, do_depth_scale=True, depth_scale=1000, target_height=None, target_width=None
    ):
        if target_height is None:
            target_height = self.image_processor.crop_size['height']
            target_width = self.image_processor.crop_size['width']

        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)

        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale
        img[img > 5.0] = 5.0
        return img, (target_width, target_height)

    def __getitem__(self, i):
        (
            ep_id,
            data_path,
            video,
            height,
            pitch_1,
            pitch_2,
            instruction,
            (start_frame_id, end_frame_id),
            action,
            pose,
        ) = self.list_data_dict[i]

        # VLA-E2E(height==92)의 pixel_goal_list 항목(action이 정확히 (u,v) 2-튜플인
        # 경우만 — turn_list/stop_list는 건드리지 않음)은 원본 action이 "최종 목적지
        # 또는 화면에 보이는 가장 먼 지점"이라 S1의 실제 로컬 궤적 타깃과 거리가
        # 전혀 안 맞는다. S1이 실제로 학습하는 지점으로 다시 투영해 덮어쓴다.
        behind_camera = False
        if (height == 92 and pose is not None and isinstance(action, (tuple, list)) and len(action) == 2
                and not getattr(self.data_args, 'preserve_s2_goal', False)):
            # self.predict_step_num이 아니라 S1의 실제 물리적 horizon 상수를 써야 함
            # (S2 학습(N_QUERY=32)에서도 항상 같은 ~5m 지점을 가리키게 하기 위함).
            action, behind_camera = compute_local_pixel_goal_vla_e2e(
                np.asarray(pose), S1_TRAJ_PREDICT_STEP_NUM,
                resample_interval=S1_TRAJ_RESAMPLE_INTERVAL,
            )  # behind_camera=True면 action은 이미 0(STOP 취급)

        # 좌우 반전 augmentation(S1_USE_LR_FLIP_AUGMENT 주석 참고) — VLA-E2E 전용,
        # 샘플당 한 번만 동전 던지기로 결정해서 이미지/궤적/pixel-goal/지시문에
        # 전부 일관되게 적용한다. val split에는 절대 적용하지 않는다 — 안 그러면
        # 평가할 때마다 무작위로 다른 절반이 뒤집혀서 재현 불가능해지고, GT
        # self-consistency 검증(eval_s1_heading_channel_consistency.py) 결과도
        # 실행마다 달라진다(실측으로 확인: val에도 적용됐을 때 sanity check std가
        # ~130도까지 치솟았음 — 뒤집힌 절반과 안 뒤집힌 절반이 섞여서 생긴 현상).
        _is_val_split = 'val' in getattr(self.data_args, 'vln_dataset_use', '')
        flip_probability = getattr(self.data_args, 'lr_flip_probability', 0.5)
        do_lr_flip = (
            (height == 92) and S1_USE_LR_FLIP_AUGMENT and not _is_val_split
            and getattr(self.data_args, 'data_augmentation', True)
            and random.random() < flip_probability
        )
        if do_lr_flip:
            instruction = flip_left_right_text(instruction)
            if isinstance(action, tuple) and len(action) == 2:
                action = (VLA_E2E_IMG_W - 1 - action[0], action[1])

        if start_frame_id != 0 and self.num_history > 0:
            history_id = np.unique(np.linspace(0, start_frame_id - 1, self.num_history, dtype=np.int32)).tolist()
        else:
            history_id = []

        images = []
        grid_thws = []
        traj_images = []
        traj_depths = []  # optional

        skip_traj = getattr(self.data_args, 'skip_traj_images', False)
        for id in range(0, end_frame_id):
            image_file = os.path.join(
                video, f"observation.images.rgb.{height}cm_{pitch_1}deg", f"episode_{ep_id:06d}_{id}.jpg"
            )
            if hasattr(self, 'get_frame_path'):
                image_file = self.get_frame_path(ep_id, id)
            if id in history_id or id == start_frame_id:
                image = Image.open(image_file).convert('RGB')
                if do_lr_flip:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if self.data_args.transform_train is not None:
                    image = self.data_args.transform_train(image)
                image, grid_thw = self.process_image_unified(image)
                images.append(image)
                grid_thws.append(grid_thw)
                if id == start_frame_id and pose is not None:
                    lookdown_image = Image.open(image_file.replace(f'_{pitch_1}deg', f'_{pitch_2}deg')).convert('RGB')
                    if do_lr_flip:
                        lookdown_image = lookdown_image.transpose(Image.FLIP_LEFT_RIGHT)
                    if height == 92:  # 실제 lookdown 카메라 없음 -> 의사-lookdown 크롭
                        lookdown_image = pseudo_lookdown_crop(lookdown_image)
                    image, grid_thw = self.process_image_unified(lookdown_image)
                    images.append(image)
                    grid_thws.append(grid_thw)
                    if not skip_traj:
                        depth_image = Image.open(
                            image_file.replace(f'_{pitch_1}deg', f'_{pitch_2}deg').replace('rgb', 'depth').replace('.jpg', '.png')
                        )
                        depth_image, _ = self.preprocess_depth_image_v2(
                            depth_image, do_depth_scale=True, depth_scale=1000, target_height=224, target_width=224
                        )
                        depth_image = torch.as_tensor(np.ascontiguousarray(depth_image)).float()
                        if do_lr_flip:
                            depth_image = torch.flip(depth_image, dims=[-1])
                        traj_images.append(lookdown_image)
                        traj_depths.append(depth_image)
            elif id > start_frame_id:
                if not skip_traj:
                    lookdown_image = Image.open(image_file.replace(f'_{pitch_1}deg', f'_{pitch_2}deg')).convert('RGB')
                    if do_lr_flip:
                        lookdown_image = lookdown_image.transpose(Image.FLIP_LEFT_RIGHT)
                    if height == 92:  # 실제 lookdown 카메라 없음 -> 의사-lookdown 크롭
                        lookdown_image = pseudo_lookdown_crop(lookdown_image)
                    depth_image = Image.open(
                        image_file.replace(f'_{pitch_1}deg', f'_{pitch_2}deg').replace('rgb', 'depth').replace('.jpg', '.png')
                    )
                    depth_image, _ = self.preprocess_depth_image_v2(
                        depth_image, do_depth_scale=True, depth_scale=1000, target_height=224, target_width=224
                    )
                    depth_image = torch.as_tensor(np.ascontiguousarray(depth_image)).float()
                    if do_lr_flip:
                        depth_image = torch.flip(depth_image, dims=[-1])
                    traj_images.append(lookdown_image)
                    traj_depths.append(depth_image)

        history_imgs = "<image>\n" * len(history_id)

        if start_frame_id != 0 and self.num_history > 0:
            chat_sources = [
                [
                    {
                        'from': 'human',
                        'value': f"You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task. These are your historical observations: <history>. {(self.conjunctions[0] if getattr(self.data_args, 'deterministic_prompt', False) else random.choice(self.conjunctions))}<image>.",
                    }
                ]
            ]
            chat_sources[0][0]['value'] = (
                chat_sources[0][0]['value'].replace('<instruction>', instruction).replace('<history>', history_imgs)
            )
        else:
            chat_sources = [
                [
                    {
                        'from': 'human',
                        'value': f"You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task. {(self.conjunctions[0] if getattr(self.data_args, 'deterministic_prompt', False) else random.choice(self.conjunctions))}<image>.",
                    }
                ]
            ]
            chat_sources[0][0]['value'] = chat_sources[0][0]['value'].replace('<instruction>', instruction)

        # 이미지 로딩 루프(위)는 `id == start_frame_id and pose is not None`일 때
        # lookdown/goal 이미지를 하나 더 추가한다. 텍스트 쪽 <image> placeholder
        # 개수가 여기에 정확히 맞아야 하므로, 분기는 반드시 pose 유무를 먼저 봐야
        # 한다 (action==0/STOP이어도 include_stop=True면 pose가 채워져 있음).
        if pose is not None:
            if action == 0:
                # include_stop=True로 채워진 STOP 직전 구간: 궤적(traj_poses)은
                # 학습하되 텍스트 타깃은 그대로 STOP.
                stop_gpt_value = self.idx2actions[action]
            else:
                stop_gpt_value = f'{action[0]} {action[1]}'
            chat_sources[0].extend(
                [
                    {'from': 'gpt', 'value': self.idx2actions[5]},
                    {'from': 'human', 'value': f'{(self.conjunctions[0] if getattr(self.data_args, 'deterministic_prompt', False) else random.choice(self.conjunctions))}<image>.'},
                    {'from': 'gpt', 'value': stop_gpt_value},
                ]
            )
        elif action == 0:
            chat_sources[0].extend([{'from': 'gpt', 'value': self.idx2actions[action]}])
        else:
            turn_action_text = ''.join([self.idx2actions[idx] for idx in action])
            chat_sources[0].extend([{'from': 'gpt', 'value': turn_action_text}])

        grid_thw_merged = copy.deepcopy(grid_thws)

        if not isinstance(grid_thws, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thws = [grid_thws]

        grid_thw_merged = [
            merged_thw.prod() // self.data_args.image_processor.merge_size**2 for merged_thw in grid_thw_merged
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

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]
        data_dict["pixel_values"] = torch.cat(images, dim=0)
        data_dict["image_grid_thw"] = torch.cat([thw.unsqueeze(0) for thw in grid_thws], dim=0)

        if self.pixel_goal_only and pose is not None:
            goal_len = end_frame_id - start_frame_id - 1
            interval = 2
            frame_ids = np.arange(0, goal_len, interval)
            max_len = 12
            if len(frame_ids) > max_len:
                interval = int(np.ceil(goal_len / max_len))
                frame_ids = np.arange(0, goal_len, interval)

            # (예전엔 pose를 절대좌표로 잘못 해석해 실제 이동량이 작다고 착각,
            # 그래서 1e-6으로 필터링을 사실상 꺼뒀었다. 올바르게 계산하면 이동
            # 프레임당 0.5~0.7m급이라 r2r 기본값(0.05, 약 22cm)이 그대로 적절함
            # — 노이즈(mm급)는 걸러내고 실제 정지 구간은 자동으로 0에 고정되며,
            # 실제 이동 구간은 영향 없음.
            step_threshold = 0.05
            # VLA-E2E의 pose는 get_trajectory_relative_to_frame이 기대하는 절대
            # world2cam이 아니라 "frame i -> i+1" 상대 SE(3) 변환이라 전용 함수로
            # 순차 누적한다 (get_trajectory_relative_to_frame_vla_e2e 참고).
            is_vla_e2e = height == 92
            # 기본 0.1m 간격(predict_step_num=4 기준 총 0.5m)은 이미지에 투영하면
            # 차량 바로 앞이라 화면 밖/발밑에 찍혀 시각적으로 의미가 없다.
            # VLA-E2E는 1.0m 간격(총 5.0m)으로 늘려 더 먼 거리를 예측하게 한다.
            resample_interval = 1.0 if is_vla_e2e else 0.1

            # eval_s1_val_split.py와 동일한 기준으로 샘플 카테고리를 매겨
            # 학습 loss 가중치(onscreen_long 업웨이트)에 사용할 수 있게 한다.
            if is_vla_e2e:
                _discrete_full = get_trajectory_relative_to_frame_vla_e2e(np.asarray(pose))
            else:
                _discrete_full = get_trajectory_relative_to_frame(pose, camera_deg=pitch_2)
            _discrete_full = filter_backward_points(_discrete_full)
            _xy_full = _discrete_full[:, :2] if len(_discrete_full) > 0 else np.zeros((0, 2))
            reach = float(np.linalg.norm(_xy_full[-1] - _xy_full[0])) if len(_xy_full) > 0 else 0.0
            offscreen = isinstance(action, tuple) and action == (-1, -1)
            if behind_camera:
                # 로컬 pixel-goal이 카메라 뒤로 나와 STOP 취급된 경우 — 정지씬과 동일하게 분류
                data_dict["sample_category"] = "near_zero"
            elif reach < 0.5:
                data_dict["sample_category"] = "near_zero"
            elif offscreen:
                data_dict["sample_category"] = "offscreen_short"
            else:
                data_dict["sample_category"] = "onscreen_long"

            traj_poses_gt = []
            for cid in frame_ids:
                pose_seq = pose[cid:]
                if is_vla_e2e:
                    discrete_traj_pose = get_trajectory_relative_to_frame_vla_e2e(np.asarray(pose_seq))
                    discrete_traj_pose = filter_backward_points(discrete_traj_pose)
                else:
                    discrete_traj_pose = get_trajectory_relative_to_frame(pose_seq, camera_deg=pitch_2)
                rel_trajectory, rel_pose_resample = interpolate_and_resample_trajectory(
                    discrete_traj_pose, self.predict_step_num, step_threshold=step_threshold,
                    resample_interval=resample_interval,
                    xy_norm_scale=S1_TRAJ_XY_SCALE_VLA_E2E if is_vla_e2e else 4.0,
                )
                rel_pose_resample = clip_or_pad(rel_pose_resample, self.predict_step_num)
                if is_vla_e2e:
                    rel_pose_resample = rel_pose_resample.copy()
                    # xy_to_delta_xyt의 위치차분(atan2) 기반 theta 대신, raw pose의 SE(3)
                    # 회전행렬에서 뽑은 진짜 orientation을 쓴다 (저속/정지 근처 구간에서
                    # atan2가 위치 노이즈에 취약한 문제를 피함 — S1_RAW_YAW_CORRECTION_RAD 주석 참고).
                    raw_theta = resample_theta_from_raw_orientation(
                        discrete_traj_pose, self.predict_step_num, resample_interval=resample_interval,
                    )
                    if do_lr_flip:
                        # raw_theta[0]은 절대각(anchor — resample_theta_from_raw_orientation의
                        # first_abs, "이 프레임의 실제 초기 heading"), raw_theta[1:]는 진짜
                        # step-to-step delta다. 거울상으로 뒤집을 때 절대각은 "pi - theta"로
                        # 반사되지만 delta(차분)는 단순 부호 반전과 동치라 — 이 둘을 구분 안
                        # 하고 전체를 부호 반전하면 anchor가 180도 어긋나 그 뒤로 이어지는
                        # cumsum 전체가 밀리는 버그가 생긴다(실측으로 발견: flip된 샘플의 GT
                        # arrow-vs-motion self-consistency sanity check가 깨져서 std가 ~130도
                        # 까지 치솟음 — eval_s1_heading_channel_consistency.py로 확인).
                        raw_theta = raw_theta.copy()
                        if len(raw_theta) > 0:
                            raw_theta[0] = (np.pi - raw_theta[0] + np.pi) % (2 * np.pi) - np.pi
                        raw_theta[1:] *= -1
                    if S1_USE_AXIS_ROTATION_FIX:
                        raw_theta = rotate_theta_for_pretrain_alignment(raw_theta)
                    n_theta = min(len(raw_theta), rel_pose_resample.shape[0])
                    rel_pose_resample[:n_theta, 2] = raw_theta[:n_theta]
                    rel_pose_resample[:, 2] *= S1_TRAJ_THETA_SCALE
                    if S1_USE_AXIS_ROTATION_FIX:
                        _rx, _ry = rotate_xy_for_pretrain_alignment(
                            rel_pose_resample[:, 0], rel_pose_resample[:, 1],
                        )
                        rel_pose_resample[:, 0] = _rx
                        rel_pose_resample[:, 1] = _ry
                    # lateral(우측=양수) 채널만 부호 반전 — forward는 불변, theta는 위에서
                    # 이미 raw_theta 단계에서 정확히 처리했다.
                    if do_lr_flip:
                        rel_pose_resample[:, 1] *= -1
                traj_poses_gt.append(torch.tensor(rel_pose_resample))

            n_frames = len(frame_ids)
            if skip_traj:
                # dummy 텐서: collator가 video_frame_num을 정상 계산할 수 있도록 shape만 맞춤
                data_dict["traj_images"] = torch.zeros(n_frames, 224, 224, 3)
                data_dict["traj_depths"] = torch.zeros(n_frames, 224, 224)
            else:
                traj_images = torch.tensor(np.stack([np.asarray(timg.resize((224, 224))) for timg in traj_images])) / 255.0
                sel_images = traj_images[:goal_len][::interval]
                sel_depths = torch.stack(traj_depths[:goal_len][::interval])
                # S1 command inference is current-frame (s=0).  Keep exactly the
                # matching s=0 image/trajectory pair.  Repeating frame0 for later
                # replanning targets would attach different trajectories to an
                # identical condition; overlapping command windows already expose
                # those later frames as their own s=0 samples.
                if is_vla_e2e and S1_USE_FRAME_ALIGNMENT_FIX:
                    sel_images = sel_images[0:1]
                    sel_depths = sel_depths[0:1]
                    traj_poses_gt = traj_poses_gt[0:1]
                    n_frames = 1
                data_dict["traj_images"] = sel_images
                data_dict["traj_depths"] = sel_depths
            data_dict["traj_poses"] = torch.stack(traj_poses_gt)

        # NaN 디버깅용 메타데이터 (S1Trainer가 NaN 발견 시 이걸로 원인 샘플 추적)
        data_dict["debug_meta"] = {
            "ep_id": ep_id, "video": video, "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id, "action": action,
            "pose": np.asarray(pose).tolist() if pose is not None else None,
        }
        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def process_input_with_traj_tokens(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        max_seq_len: int = None,
        traj_token_length: int = 4,  # TODO hard-code
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
        if max_seq_len is None:
            max_seq_len = self.tokenizer.model_max_length - traj_token_length

        batch_size = len(input_ids)
        multi_input_ids = [None] * batch_size
        multi_labels = [None] * batch_size
        t_s_pos = [0] * batch_size

        traj_token_template = torch.full(
            (traj_token_length,), TRAJ_TOKEN_INDEX, dtype=input_ids[0].dtype, device=input_ids[0].device
        )

        for i in range(batch_size):
            truncated_input = input_ids[i][:max_seq_len]
            truncated_label = labels[i][:max_seq_len]

            t_s_pos[i] = len(truncated_input)

            multi_input_ids[i] = torch.cat([truncated_input, traj_token_template])
            multi_labels[i] = torch.cat([truncated_label, traj_token_template.clone()])

        return multi_input_ids, multi_labels, t_s_pos

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]

        if "traj_images" in instances[0]:
            input_ids, labels, t_s_pos = self.process_input_with_traj_tokens(input_ids, labels)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        if input_ids.shape[1] > self.tokenizer.model_max_length:
            print(
                f"Warning input with length {input_ids.shape[1]} is longer than max length {self.tokenizer.model_max_length}"
            )

        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
        videos = list(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids

        if "traj_images" in instances[0]:
            traj_images, traj_depths, traj_poses = tuple(
                [instance[key] for instance in instances] for key in ("traj_images", "traj_depths", "traj_poses")
            )
            video_frame_num = []
            max_len = max(img.shape[0] for img in traj_images)
            traj_image_batch = []
            traj_depth_batch = []
            traj_pose_batch = []
            # import pdb; pdb.set_trace()
            for idx in range(len(traj_images)):
                t_img = traj_images[idx]
                t_depth = traj_depths[idx]
                t_pose = traj_poses[idx]
                n_frames = t_img.shape[0]
                video_frame_num.append(n_frames)
                if n_frames < max_len:
                    pad_len = max_len - n_frames
                    last_img = t_img[-1]
                    last_depth = t_depth[-1]
                    last_pose = t_pose[-1]
                    padding_img = last_img.unsqueeze(0).repeat(pad_len, 1, 1, 1)
                    padding_depth = last_depth.unsqueeze(0).repeat(pad_len, 1, 1)
                    padding_pose = last_pose.unsqueeze(0).repeat(pad_len, 1, 1)
                    padded_img = torch.cat([t_img, padding_img], dim=0)
                    padded_depth = torch.cat([t_depth, padding_depth], dim=0)
                    padded_pose = torch.cat([t_pose, padding_pose], dim=0)
                else:
                    padded_img = t_img
                    padded_depth = t_depth
                    padded_pose = t_pose
                traj_image_batch.append(padded_img)
                traj_depth_batch.append(padded_depth)
                traj_pose_batch.append(padded_pose)
            batch['position_ids'] = None
            batch['t_s_pos'] = t_s_pos
            batch['traj_images'] = torch.stack(traj_image_batch)
            batch['traj_depths'] = torch.stack(traj_depth_batch)
            batch['traj_poses'] = torch.stack(traj_pose_batch)
            batch['video_frame_num'] = torch.tensor(video_frame_num)

        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(*(instance["attention_mask"] for instance in instances if "attention_mask" in instance))
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
        videos = list(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


class CombinedDataset(Dataset):
    """
    Combine multiple datasets into a single dataset interface.

    This class is used to merge different datasets for joint training.
    It concatenates samples from all provided datasets and optionally shuffles
    the global index mapping (without changing the underlying datasets).
    """

    def __init__(self, datasets, shuffle=False):
        super(CombinedDataset, self).__init__()
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cum_lengths = np.cumsum(self.lengths)
        self.total_length = self.cum_lengths[-1]
        self.shuffle_enabled = shuffle
        self.indices = np.arange(self.total_length)
        if self.shuffle_enabled:
            self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indices)

    def _map_index(self, idx):
        return self.indices[idx]

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, i):
        real_idx = self._map_index(i)
        for idx, cum_len in enumerate(self.cum_lengths):
            if real_idx < cum_len:
                return self.datasets[idx][real_idx - cum_len + self.lengths[idx]]
        raise ValueError(f"Index {real_idx} out of bound")


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_datasets = []
    if data_args.iign_dataset_use:
        train_datasets.append(VLLNDataset(tokenizer=tokenizer, data_args=data_args))
    if data_args.vln_dataset_use:
        if getattr(data_args, 'use_coview', False):
            from internnav.dataset.coview_nav_dataset import CoViewNavDataset
            train_datasets.append(CoViewNavDataset(tokenizer=tokenizer, data_args=data_args))
        else:
            train_datasets.append(NavPixelGoalDataset(tokenizer=tokenizer, data_args=data_args))
    train_dataset = CombinedDataset(train_datasets, shuffle=False)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


if __name__ == "__main__":
    pass
