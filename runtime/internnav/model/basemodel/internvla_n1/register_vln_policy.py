"""
RegisterVLN eval policy — wraps RegisterVLNForCausalLM with the same
s2_step / s1_step_latent interface as InternVLAN1Net.
"""

import sys
import os

# project root so we can import models.register_vln
_PROJECT = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../../../'))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

import copy
import itertools
import re
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PreTrainedModel

from internnav.configs.model.base_encoders import ModelCfg
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ModelConfig
from internnav.model.utils.vln_utils import (
    S1Output,
    S2Output,
    chunk_token,
    split_and_clean,
    traj_to_actions,
)

from transformers import AutoConfig, AutoModelForCausalLM
from models.register_vln import RegisterVLNForCausalLM, RegisterVLNModelConfig

_QWEN_BASE = '/home/work/models/Qwen2.5-VL-7B-Instruct'

# register custom model type so AutoConfig/AutoModel can resolve it
AutoConfig.register("register_vln", RegisterVLNModelConfig)
AutoModelForCausalLM.register(RegisterVLNModelConfig, RegisterVLNForCausalLM)


class RegisterVLNNet(PreTrainedModel):
    config_class = InternVLAN1ModelConfig

    def __init__(self, config: Union[InternVLAN1ModelConfig, ModelCfg]):
        super().__init__(config)
        self.model_config = ModelCfg(**config.model_cfg['model'])

        ckpt_path = self.model_config.model_path
        cfg = RegisterVLNModelConfig.from_pretrained(_QWEN_BASE)
        cfg.system1          = 'nextdit_singleview'
        cfg.n_query          = 32
        cfg.sv_num_registers     = 16   # per image; ×2 images = 32 total (DualVLN-aligned)
        cfg.sv_cross_attn_layers = 3
        cfg.dav2_ckpt        = '/home/work/vla_ws/vla_project/checkpoints/depth_anything_v2_metric_hypersim_vits.pth'

        self.model = RegisterVLNForCausalLM.from_pretrained(
            ckpt_path,
            config=cfg,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
            device_map={"": self.model_config.device},
        )
        self.model.config.use_cache = True
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(_QWEN_BASE, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(_QWEN_BASE)
        self.processor.tokenizer = self.tokenizer
        self.processor.tokenizer.padding_side = 'left'

        self.init_prompts()

        self.num_history  = self.model_config.num_history
        self.resize_w     = self.model_config.resize_w
        self.resize_h     = self.model_config.resize_h
        self.continuous_traj = getattr(self.model_config, 'continuous_traj', True)

        self.rgb_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""

    # ── identical to InternVLAN1Net ────────────────────────────────────────

    def init_prompts(self):
        self.DEFAULT_IMAGE_TOKEN = "<image>"
        prompt = (
            "You are an autonomous navigation assistant. Your task is to <instruction>. "
            "Where should you go next to stay on track? "
            "Please output the next waypoint\'s coordinates in the image. "
            "Please output STOP when you have successfully completed the task."
        )
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
        self.conjunctions = [
            'you can see ', 'in front of you is ', 'there is ',
            'you can spot ', 'you are toward the ', 'ahead of you is ', 'in your sight is ',
        ]
        self.actions2idx = OrderedDict({
            'STOP': [0], "↑": [1], "←": [2], "→": [3], "↓": [5],
        })

    def reset(self):
        self.rgb_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(a) for a in self.actions2idx)
        matches = re.compile(action_patterns).findall(output)
        return list(itertools.chain.from_iterable(self.actions2idx[m] for m in matches))

    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
            self.conversation_history = []
            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            cur_images = self.rgb_list[-1:]
            if self.episode_idx == 0:
                history_id = []
            else:
                history_id = np.unique(
                    np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)
                ).tolist()
                placeholder = (self.DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'
            self.input_images = [self.rgb_list[i] for i in sorted(history_id)] + cur_images
            input_img_id = 0
            self.episode_idx += 1
        else:
            self.input_images.append(image)
            input_img_id = -1
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )

        prompt = self.conjunctions[0] + self.DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" {prompt}."
        parts = split_and_clean(sources[0]["value"])

        content = []
        for part in parts:
            if part == "<image>":
                content.append({"type": "image", "image": self.input_images[input_img_id]})
                input_img_id += 1
            else:
                content.append({"type": "text", "text": part})

        self.conversation_history.append({'role': 'user', 'content': content})
        text = self.processor.apply_chat_template(
            self.conversation_history, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                use_cache=True, past_key_values=None, return_dict_in_generate=True,
            ).sequences
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        print(f"============ output {self.episode_idx}  {self.llm_output}")

        output = S2Output()
        if bool(re.search(r'\d', self.llm_output)):
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            output.output_pixel = np.array([int(coord[1]), int(coord[0])])
            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            with torch.no_grad():
                traj_latents = self.model.generate_latents(
                    output_ids, inputs.pixel_values, image_grid_thw
                )
            output.output_latent = traj_latents
        else:
            output.output_action = self.parse_actions(self.llm_output)

        return output

    def s1_step_latent(self, rgb, depth, latent):
        with torch.no_grad():
            dp_actions = self.model.generate_traj(
                traj_latents=latent, images_dp=rgb, depths_dp=depth
            )
        if self.continuous_traj:
            action_list = traj_to_actions(dp_actions)
        else:
            random_choice = np.random.choice(dp_actions.shape[0])
            action_list = chunk_token(dp_actions[random_choice])

        action_list = [x for x in action_list if x != 0]
        return S1Output(idx=action_list[:4])
