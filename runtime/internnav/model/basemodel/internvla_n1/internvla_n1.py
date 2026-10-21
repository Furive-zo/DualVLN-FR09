from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
)
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from .internvla_n1_arch import InternVLAN1MetaForCausalLM, InternVLAN1MetaModel

TRAJ_TOKEN_INDEX = 151667
IMAGE_TOKEN_INDEX = 151655
_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class InternVLAN1ModelConfig(Qwen2_5_VLConfig):
    model_type = "internvla_n1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)


class InternVLAN1Model(InternVLAN1MetaModel, Qwen2_5_VLModel):
    config_class = InternVLAN1ModelConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(InternVLAN1Model, self).__init__(config)

    @property
    def embed_tokens(self):
        return self.language_model.embed_tokens

    @property
    def norm(self):
        return self.language_model.norm

    @property
    def layers(self):
        return self.language_model.layers


class InternVLAN1ForCausalLM(Qwen2_5_VLForConditionalGeneration, InternVLAN1MetaForCausalLM):
    config_class = InternVLAN1ModelConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type == "internvla_n1"

        self.model = InternVLAN1Model(config)
        self.rope_deltas = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    def get_model(self):
        return self.model

    @property
    def visual(self):
        return self.model.visual

    def get_rope_index(self, *args, **kwargs):
        return self.model.get_rope_index(*args, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        t_s_pos: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        traj_images: Optional[torch.Tensor] = None,
        traj_depths: Optional[torch.Tensor] = None,
        video_frame_num: Optional[torch.Tensor] = None,
        traj_poses: Optional[torch.Tensor] = None,
        cached_traj_hs: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
        xy_loss_weight: float = 1.0,
        theta_loss_weight: float = 1.0,
        lateral_loss_weight: float = 0.0,
        reverse_loss_weight: float = 0.0,
        endpoint_loss_weight: float = 0.0,
        horizon_loss_strength: float = 0.0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # S1 cached mode: LLM/visual forward 생략, 사전 계산된 features 직접 사용
        if cached_traj_hs is not None:
            cached_memory_tokens = kwargs.pop('cached_memory_tokens', None)
            return self._forward_s1_cached(
                cached_traj_hs=cached_traj_hs,
                cached_memory_tokens=cached_memory_tokens,
                traj_images=traj_images,
                traj_poses=traj_poses,
                video_frame_num=video_frame_num,
                return_dict=return_dict,
            )

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            n_traj_tokens = (input_ids == TRAJ_TOKEN_INDEX).sum().item()
            traj_idx = input_ids == TRAJ_TOKEN_INDEX
            latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
            H = latent_queries.shape[-1]
            latent_queries = latent_queries.contiguous().view(-1, H)
            if n_traj_tokens != 0:
                inputs_embeds[traj_idx] = latent_queries

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None and t_s_pos is None:
            # S2 mode: LM cross-entropy loss only (no trajectory DiT)
            vocab_size = logits.shape[-1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

        elif labels is not None:
            # S1 mode: DiT diffusion loss
            traj_hidden_states = []
            for b in range(hidden_states.shape[0]):
                traj_hidden_states.append(hidden_states[b, t_s_pos[b] : t_s_pos[b] + self.config.n_query, :])

            traj_hidden_states = torch.stack(traj_hidden_states, dim=0)
            traj_hidden_states = traj_hidden_states.unsqueeze(1).repeat(1, traj_poses.size(1), 1, 1).flatten(0, 1)
            loss_mask = torch.arange(traj_images.size(1), device=self.device).expand(
                traj_images.size(0), traj_images.size(1)
            ) < video_frame_num.unsqueeze(1)

            if 'nextdit' in self.get_system1_type():
                if 'async' in self.get_system1_type():
                    cur_images = traj_images.flatten(0, 1)
                    pix_goal_images = traj_images[:, 0:1].repeat(1, traj_images.size(1), 1, 1, 1).flatten(0, 1)
                    bsz = cur_images.size(0)
                    images_dp = torch.stack([pix_goal_images, cur_images], dim=1).permute(0, 1, 4, 2, 3)
                    images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std

                    images_dp_feat = (
                        self.get_model()
                        .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1))[0]
                        .unflatten(dim=0, sizes=(bsz, -1))
                    )

                    memory_feat = self.get_model().memory_encoder(
                        images_dp_feat.flatten(1, 2)
                    )  # [bs*select_size,512,384]
                    memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                    memory_tokens = self.get_model().rgb_resampler(memory_feat)

                    traj_hidden_states = self.get_model().cond_projector(traj_hidden_states)
                    latents = torch.cat([memory_tokens, traj_hidden_states], dim=1)
                else:
                    traj_hidden_states = self.get_model().cond_projector(traj_hidden_states)
                    latents = traj_hidden_states

                relative_poses = traj_poses.flatten(0, 1)
                bsz = relative_poses.shape[0]
                noise = torch.randn(relative_poses.shape, device=relative_poses.device, dtype=relative_poses.dtype)
                u = torch.rand(size=(bsz,), device="cpu")
                indices = (u * self.get_model().noise_scheduler.config.num_train_timesteps).long()
                timesteps = self.get_model().noise_scheduler.timesteps[indices].to(device=latents.device)
                sigmas = self.get_sigmas(
                    timesteps, latents.device, n_dim=relative_poses.shape[-1], dtype=relative_poses.dtype
                )

                noisy_trajectory = (1 - sigmas) * relative_poses + sigmas * noise
                action_features = self.get_model().action_encoder(noisy_trajectory)
                pos_ids = torch.arange(relative_poses.shape[1]).reshape(1, -1).repeat(bsz, 1).to(relative_poses.device)
                pos_embed = self.get_model().pos_encoding(pos_ids)
                action_features += pos_embed

                noise_pred = self.get_model().traj_dit(
                    x=action_features,
                    timestep=timesteps,
                    z_latents=latents,
                )
                noise_pred = self.get_model().action_decoder(noise_pred)
                target = noise - relative_poses
                flow_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                mask = loss_mask.flatten(0, 1)[:, None, None]
                if sample_weight is not None:
                    # sample_weight: (B,) — traj_poses.size(1)개 프레임에 동일 가중치.
                    w = sample_weight.to(flow_loss.device, dtype=flow_loss.dtype)
                    w = w.view(-1, 1).repeat(1, traj_poses.size(1)).flatten(0, 1)[:, None, None]
                    mask = mask * w

                # Later waypoints had the largest reverse/mismatch error.  Apply a
                # normalized linear ramp so the global loss scale stays comparable.
                step_weights = torch.linspace(
                    1.0, 1.0 + horizon_loss_strength, flow_loss.shape[1],
                    device=flow_loss.device, dtype=flow_loss.dtype,
                )
                step_weights = step_weights / step_weights.mean()
                weighted_mask = mask * step_weights[None, :, None]
                denom = mask.sum().clamp_min(1e-6) * step_weights.sum()
                xy_loss = (flow_loss[..., :2] * weighted_mask).sum() / denom / 2
                theta_loss = (flow_loss[..., 2:] * weighted_mask).sum() / denom

                # Flow matching predicts v = noise - x0 for
                # x_sigma = x0 + sigma * v, hence x0_hat = x_sigma - sigma*v_hat.
                clean_pred = noisy_trajectory.float() - sigmas.float() * noise_pred.float()
                clean_gt = relative_poses.float()
                pred_heading = torch.cumsum(clean_pred[..., 2], dim=1)
                cos_h, sin_h = torch.cos(pred_heading), torch.sin(pred_heading)
                dx, dy = clean_pred[..., 0], clean_pred[..., 1]
                speed = torch.sqrt(dx.square() + dy.square()).clamp_min(0.05)
                longitudinal = (dx * cos_h + dy * sin_h) / speed
                lateral_ratio = (-dx * sin_h + dy * cos_h) / speed

                # Ignore zero-padded/stationary GT.  Global lateral displacement is
                # allowed.  Match the predicted body-frame side-slip ratio to the
                # demonstrated one, preserving normal turning curvature.
                gt_heading = torch.cumsum(clean_gt[..., 2], dim=1)
                gt_cos_h, gt_sin_h = torch.cos(gt_heading), torch.sin(gt_heading)
                gt_dx, gt_dy = clean_gt[..., 0], clean_gt[..., 1]
                gt_speed = torch.sqrt(gt_dx.square() + gt_dy.square()).clamp_min(0.05)
                gt_lateral_ratio = (-gt_dx * gt_sin_h + gt_dy * gt_cos_h) / gt_speed
                gt_longitudinal = (gt_dx * gt_cos_h + gt_dy * gt_sin_h) / gt_speed

                sample_mask = mask[:, 0, 0].to(flow_loss.dtype)
                moving_gt = torch.linalg.vector_norm(clean_gt[..., :2], dim=-1) > 0.02
                motion_mask = sample_mask[:, None] * moving_gt.to(flow_loss.dtype)
                motion_denom = motion_mask.sum().clamp_min(1.0)
                lateral_raw = F.smooth_l1_loss(
                    lateral_ratio, gt_lateral_ratio, reduction="none", beta=0.05,
                )
                lateral_loss = (lateral_raw * motion_mask).sum() / motion_denom
                longitudinal_raw = F.smooth_l1_loss(
                    longitudinal, gt_longitudinal, reduction="none", beta=0.05,
                )
                reverse_raw = longitudinal_raw + F.relu(-longitudinal).square()
                reverse_loss = (reverse_raw * motion_mask).sum() / motion_denom

                pred_endpoint = torch.cumsum(clean_pred[..., :2], dim=1)[:, -1]
                gt_endpoint = torch.cumsum(clean_gt[..., :2], dim=1)[:, -1]
                endpoint_raw = F.smooth_l1_loss(
                    pred_endpoint, gt_endpoint, reduction="none", beta=0.1,
                )
                endpoint_loss = (
                    endpoint_raw * sample_mask[:, None]
                ).sum() / sample_mask.sum().clamp_min(1.0) / 2

                loss = (
                    xy_loss * xy_loss_weight
                    + theta_loss * theta_loss_weight
                    + lateral_loss * lateral_loss_weight
                    + reverse_loss * reverse_loss_weight
                    + endpoint_loss * endpoint_loss_weight
                )
                self._last_s1_loss_components = {
                    "xy": float(xy_loss.detach()),
                    "theta": float(theta_loss.detach()),
                    "lateral": float(lateral_loss.detach()),
                    "reverse": float(reverse_loss.detach()),
                    "endpoint": float(endpoint_loss.detach()),
                    "total": float(loss.detach()),
                }
            elif 'navdp' in self.get_system1_type():
                if 'async' in self.get_system1_type():
                    cur_images = traj_images.flatten(0, 1)
                    cur_depths = traj_depths.flatten(0, 1)
                    pix_goal_images = traj_images[:, 0:1].repeat(1, traj_images.size(1), 1, 1, 1).flatten(0, 1)
                    pix_goal_depths = traj_depths[:, 0:1].repeat(1, traj_depths.size(1), 1, 1).flatten(0, 1)
                    images_dp = torch.stack([pix_goal_images, cur_images], dim=1)  # (bs*select_size, 2, 224, 224, 3)
                    depths_dp = torch.stack([pix_goal_depths, cur_depths], dim=1).unsqueeze(
                        -1
                    )  # (bs*select_size, 2, 224, 224, 1)
                    pred_pg, noise = self.model.navdp.forward_vlm_traj(
                        traj_hidden_states, images_dp, depths_dp, tensor_label_actions=traj_poses
                    )
                    pg_action_loss = (pred_pg - noise).square()
                    mask = loss_mask.flatten(0, 1)[:, None, None]
                    masked_loss = pg_action_loss * mask
                    loss = masked_loss.sum() / mask.sum() / (pg_action_loss.shape[1] * pg_action_loss.shape[2])

            else:
                raise NotImplementedError

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _forward_s1_cached(self, cached_traj_hs, traj_poses, video_frame_num,
                            cached_memory_tokens=None, traj_images=None, return_dict=True):
        """LLM/visual forward 생략 — 사전 계산된 features로 DiT 손실만 계산."""
        B = cached_traj_hs.shape[0]
        T = traj_poses.size(1)  # future steps (predict_step_num)
        traj_hidden_states = cached_traj_hs  # (B, n_query, hidden_size)

        loss_mask = torch.arange(T, device=traj_poses.device).expand(B, T) \
                    < video_frame_num.unsqueeze(1)

        traj_hidden_states = traj_hidden_states.unsqueeze(1).repeat(1, T, 1, 1).flatten(0, 1)

        if cached_memory_tokens is not None:
            # visual cache: (B, T_full, 32, 768) → 마지막 T 프레임(미래)만 사용
            memory_tokens = cached_memory_tokens[:, -T:].flatten(0, 1)  # (B*T, 32, 768)
        else:
            # fallback: 실시간 계산 — traj_images의 마지막 T 프레임만 사용
            cur_images      = traj_images[:, -T:].flatten(0, 1)
            pix_goal_images = traj_images[:, 0:1].repeat(1, T, 1, 1, 1).flatten(0, 1)
            bsz = cur_images.size(0)
            images_dp      = torch.stack([pix_goal_images, cur_images], dim=1).permute(0, 1, 4, 2, 3)
            images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std
            images_dp_feat = (
                self.get_model()
                .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1))[0]
                .unflatten(dim=0, sizes=(bsz, -1))
            )
            memory_feat   = self.get_model().memory_encoder(images_dp_feat.flatten(1, 2))
            memory_feat   = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
            memory_tokens = self.get_model().rgb_resampler(memory_feat)

        traj_hidden_states = self.get_model().cond_projector(traj_hidden_states)
        latents = torch.cat([memory_tokens, traj_hidden_states], dim=1)

        # eval 경로에서 traj_poses가 float64(Double)로 들어와 action_encoder(bf16)와
        # dtype이 어긋나 crash하는 경우가 있었다(train 경로는 우연히 이미 bf16으로
        # 들어옴) — 모델 compute dtype으로 항상 맞춰준다.
        relative_poses = traj_poses.flatten(0, 1).to(self.dtype)
        bsz_dit = relative_poses.shape[0]
        noise   = torch.randn(relative_poses.shape, device=relative_poses.device,
                              dtype=relative_poses.dtype)
        u       = torch.rand(size=(bsz_dit,), device="cpu")
        indices = (u * self.get_model().noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.get_model().noise_scheduler.timesteps[indices].to(device=latents.device)
        sigmas  = self.get_sigmas(timesteps, latents.device,
                                  n_dim=relative_poses.shape[-1], dtype=relative_poses.dtype)

        noisy_trajectory = (1 - sigmas) * relative_poses + sigmas * noise
        action_features  = self.get_model().action_encoder(noisy_trajectory)
        pos_ids = torch.arange(relative_poses.shape[1]).reshape(1, -1).repeat(bsz_dit, 1).to(relative_poses.device)
        action_features += self.get_model().pos_encoding(pos_ids)

        noise_pred = self.get_model().traj_dit(x=action_features, timestep=timesteps, z_latents=latents)
        noise_pred = self.get_model().action_decoder(noise_pred)
        target     = noise - relative_poses

        raw_loss    = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        mask        = loss_mask.flatten(0, 1)[:, None, None]
        loss        = (raw_loss * mask).sum() / mask.sum() / (raw_loss.shape[1] * raw_loss.shape[2])

        if not return_dict:
            return (loss,)
        return CausalLMOutputWithPast(loss=loss)

    def generate_latents(self, input_ids, pixel_values, image_grid_thw):
        input_ids.to(self.get_model().device)
        with torch.no_grad():
            text_embeds = self.get_model().embed_tokens(input_ids)
        latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)
        image_idx = input_ids == IMAGE_TOKEN_INDEX
        N_QUERY = self.get_n_query()
        input_ids = torch.cat([input_ids, torch.tensor([[TRAJ_TOKEN_INDEX] * N_QUERY]).to(input_ids.device)], dim=1)

        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).unsqueeze(0)

        text_embeds[image_idx] = image_embeds.to(text_embeds.device)[: image_idx.sum(), :]

        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)

        position_ids, _ = self.get_rope_index(input_ids, image_grid_thw)
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=text_embeds,
                position_ids=position_ids,
                # attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states[-1][:, -N_QUERY:, :]

        return hidden_states

    def generate_traj(
        self,
        traj_latents,
        images_dp,
        depths_dp=None,
        predict_step_nums=32,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 32,
    ):
        if 'nextdit' in self.get_system1_type():
            scheduler = FlowMatchEulerDiscreteScheduler()
            device = traj_latents.device
            dtype = traj_latents.dtype

            traj_latents = self.get_model().cond_projector(traj_latents)
            if 'async' in self.get_system1_type():
                with torch.no_grad():
                    images_dp = images_dp.permute(0, 1, 4, 2, 3)
                    images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std
                    self.get_model().rgb_model.to(dtype)
                    images_dp_feat = (
                        self.get_model()
                        .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1).to(dtype))[0]
                        .unflatten(dim=0, sizes=(1, -1))
                    )
                    memory_feat = self.get_model().memory_encoder(
                        images_dp_feat.flatten(1, 2)
                    )  # [bs*select_size,512,384]
                    memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                    memory_tokens = self.get_model().rgb_resampler(memory_feat)
                hidden_states = torch.cat([memory_tokens, traj_latents], dim=1)
            else:
                hidden_states = traj_latents
            hidden_states_null = torch.zeros_like(hidden_states, device=device, dtype=dtype)
            hidden_states_input = torch.cat([hidden_states_null, hidden_states], 0)
            batch_size = traj_latents.shape[0]
            latent_size = predict_step_nums
            latent_channels = 3

            latents = randn_tensor(
                shape=(batch_size * num_sample_trajs, latent_size, latent_channels),
                generator=None,
                device=device,
                dtype=dtype,
            )

            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

            hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)

            for t in scheduler.timesteps:
                latent_features = self.get_model().action_encoder(latents)
                pos_ids = (
                    torch.arange(latent_features.shape[1])
                    .reshape(1, -1)
                    .repeat(batch_size, 1)
                    .to(latent_features.device)
                )
                pos_embed = self.get_model().pos_encoding(pos_ids)
                latent_features += pos_embed  # [num_sample_trajs, t, 384]
                latent_model_input = latent_features.repeat(2, 1, 1)
                if hasattr(scheduler, "scale_model_input"):
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict noise model_output
                noise_pred = self.get_model().traj_dit(
                    x=latent_model_input,
                    timestep=t.unsqueeze(0)
                    .expand(latent_model_input.shape[0])
                    .to(latent_model_input.device, torch.long),
                    z_latents=hidden_states_input,
                )

                noise_pred = self.get_model().action_decoder(noise_pred)

                # perform guidance
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # compute previous: x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            return latents

        elif 'navdp' in self.get_system1_type():
            if 'async' in self.get_system1_type():
                all_trajs = self.model.navdp.predict_pointgoal_action_async(
                    traj_latents.to(self.get_model().device), images_dp, depths_dp
                )
            else:
                all_trajs = self.model.navdp.predict_pointgoal_action(traj_latents.to(self.get_model().device))
            return all_trajs
