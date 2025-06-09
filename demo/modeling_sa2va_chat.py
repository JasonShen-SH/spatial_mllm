# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import Any, List, Optional, Tuple, Union

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import torch.utils.checkpoint
import transformers

# from .modeling_internlm2 import InternLM2ForCausalLM
# from .modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers import StoppingCriteriaList, StoppingCriteria
import pdb

# from .configuration_sa2va_chat import Sa2VAChatConfig
# from .modeling_intern_vit import InternVisionModel, has_flash_attn

import sys
sys.path.append('/home/rqshen/Sa2VA/demo')
# from sam2 import SAM2
from sam2 import SAM2

# from .templates import PROMPT_TEMPLATE

import numpy as np
from torchvision.transforms.functional import resize, to_pil_image

from types import MethodType
import torch.nn.functional as F

try:
    # from .flash_attention import FlashAttention
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False

logger = logging.get_logger(__name__)

# def version_cmp(v1, v2, op='eq'):
#     import operator

#     from packaging import version
#     op_func = getattr(operator, op)
#     return op_func(version.parse(v1), version.parse(v2))

class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word

def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

# class Sa2VAChatModel(PreTrainedModel):
#     config_class = Sa2VAChatConfig
#     main_input_name = 'pixel_values'
#     base_model_prefix = 'language_model'
#     _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
#                          'Phi3DecoderLayer', 'Qwen2DecoderLayer', 'SAM2']
#     _supports_flash_attn_2 = True
#     supports_gradient_checkpointing = True

#     def __init__(self, config: Sa2VAChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
#         super().__init__(config)

#         assert version_cmp(transformers.__version__, '4.37.0', 'ge')
#         image_size = config.force_image_size or config.vision_config.image_size
#         patch_size = config.vision_config.patch_size
#         self.patch_size = patch_size
#         self.select_layer = config.select_layer
#         self.template = config.template
#         self.template = self.template.replace('-', '_')
#         self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
#         self.downsample_ratio = config.downsample_ratio
#         self.ps_version = config.ps_version
#         self.llm_arch_name = config.llm_config.architectures[0]

#         use_flash_attn = use_flash_attn if has_flash_attn else False
#         config.vision_config.use_flash_attn = True if use_flash_attn else False
#         config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

#         logger.info(f'num_image_token: {self.num_image_token}')
#         logger.info(f'ps_version: {self.ps_version}')
#         if vision_model is not None:
#             self.vision_model = vision_model
#         else:
#             self.vision_model = InternVisionModel(config.vision_config)
#         if language_model is not None:
#             self.language_model = language_model
#         else:
#             if config.llm_config.architectures[0] == 'LlamaForCausalLM':
#                 self.language_model = LlamaForCausalLM(config.llm_config)
#             elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
#                 self.language_model = InternLM2ForCausalLM(config.llm_config)
#             elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
#                 self.language_model = Phi3ForCausalLM(config.llm_config)
#             elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
#                 self.language_model = Qwen2ForCausalLM(config.llm_config)
#             else:
#                 raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

#         vit_hidden_size = config.vision_config.hidden_size
#         llm_hidden_size = config.llm_config.hidden_size

#         self.mlp1 = nn.Sequential(
#             nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
#             nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
#             nn.GELU(),
#             nn.Linear(llm_hidden_size, llm_hidden_size)
#         )

#         self.img_context_token_id = None
#         self.conv_template = PROMPT_TEMPLATE[self.template]
#         self.template = self.conv_template
#         if hasattr(config, 'system_message'):
#             self.system_message = config.system_message
#         self.num_samples = 0

#         if config.use_backbone_lora:
#             self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

#         if config.use_llm_lora:
#             self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

#         self.grounding_encoder = SAM2()
#         out_dim = self.grounding_encoder.hidden_dim
#         in_dim = llm_hidden_size
#         self.text_hidden_fcs = nn.Sequential(
#             nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
#             nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
#         )

#         self.init_prediction_config = False

#     def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
#         lora_config = LoraConfig(
#             r=r,
#             target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
#             lora_alpha=lora_alpha,
#             lora_dropout=lora_dropout,
#         )
#         self.vision_model = get_peft_model(self.vision_model, lora_config)
#         self.vision_model.print_trainable_parameters()

#     def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
#         # Determine the target modules based on the architecture of the language model
#         if self.llm_arch_name == 'InternLM2ForCausalLM':
#             target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
#         elif self.llm_arch_name == 'Phi3ForCausalLM':
#             target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
#         elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
#             target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
#                               'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
#         else:
#             raise NotImplemented
#         lora_config = LoraConfig(
#             r=r,
#             target_modules=target_modules,
#             lora_alpha=lora_alpha,
#             lora_dropout=lora_dropout,
#             task_type='CAUSAL_LM'
#         )
#         self.language_model = get_peft_model(self.language_model, lora_config)
#         self.language_model.enable_input_require_grads()
#         self.language_model.print_trainable_parameters()

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))
    ps_version = 'v2'
    if ps_version == 'v1':
        warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                        'which results in a transposed image.')
    else:
        x = x.permute(0, 2, 1, 3).contiguous()
    return x

def extract_feature(vision_model, mlp1, pixel_values):
    select_layer = -1
    downsample_ratio = 0.5
    vit_hidden_size = 1024
    llm_hidden_size = 4096
    
    # mlp1 = nn.Sequential(
    #     nn.LayerNorm(vit_hidden_size * int(1 / downsample_ratio) ** 2),
    #     nn.Linear(vit_hidden_size * int(1 / downsample_ratio) ** 2, llm_hidden_size),
    #     nn.GELU(),
    #     nn.Linear(llm_hidden_size, llm_hidden_size)
    # )
    
    if select_layer == -1:
        vit_embeds = vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True).last_hidden_state
    else:
        vit_embeds = vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True).hidden_states[select_layer]
    vit_embeds = vit_embeds[:, 1:, :]

    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = pixel_shuffle(vit_embeds, scale_factor=downsample_ratio)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    # pdb.set_trace()
    vit_embeds = mlp1(vit_embeds)
    return vit_embeds

    # @property
    # def lm_head(self):
    #     return self.language_model.get_output_embeddings()

    # def get_input_embeddings(self):
    #     return self.language_model.get_input_embeddings()

    # def get_output_embeddings(self):
    #     return self.language_model.get_output_embeddings()

    # def forward(self, data, data_samples=None, mode='loss'):
    #     pixel_values = data['pixel_values']

    #     if type(pixel_values) is list or pixel_values.ndim == 5:
    #         if type(pixel_values) is list:
    #             pixel_values = [
    #                 x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
    #             ]
    #         # b*n, c, h, w
    #         concat_images = torch.cat(
    #             [image.to(self.vision_model.dtype) for image in pixel_values], dim=0)
    #     else:
    #         raise NotImplementedError()

    #     input_ids = data['input_ids']
    #     position_ids = data['position_ids']
    #     attention_mask = data['attention_mask']
    #     # sum is 0 are text
    #     image_flags = torch.sum(concat_images, dim=(1, 2, 3)) != 0
    #     image_flags = image_flags.long()

    #     labels = data['labels']
    #     use_cache = False

    #     if 'vp_overall_mask' not in data.keys():
    #         vp_overall_mask = None
    #     else:
    #         vp_overall_mask = data['vp_overall_mask']

    #     if 'prompt_masks' in data.keys():
    #         prompt_masks = data['prompt_masks']
    #     else:
    #         prompt_masks = None

    #     outputs = self._llm_forward(
    #         input_ids=input_ids,
    #         position_ids=position_ids,
    #         attention_mask=attention_mask,
    #         image_flags=image_flags,
    #         pixel_values=concat_images,
    #         labels=labels,
    #         use_cache=use_cache,
    #         output_hidden_states=True,
    #         vp_overall_mask=vp_overall_mask,
    #         prompt_masks=prompt_masks,
    #     )

    #     return outputs

    # def _llm_forward(
    #         self,
    #         pixel_values: torch.FloatTensor,
    #         input_ids: torch.LongTensor = None,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         position_ids: Optional[torch.LongTensor] = None,
    #         image_flags: Optional[torch.LongTensor] = None,
    #         past_key_values: Optional[List[torch.FloatTensor]] = None,
    #         labels: Optional[torch.LongTensor] = None,
    #         use_cache: Optional[bool] = None,
    #         output_attentions: Optional[bool] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         return_dict: Optional[bool] = None,
    #         vp_overall_mask=None,
    #         prompt_masks=None,
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
    #     return_dict = return_dict if return_dict is not None \
    #         else self.config.use_return_dict

    #     image_flags = image_flags.squeeze(-1)
    #     # We only added the clone code here to avoid the error.
    #     input_embeds = self.language_model.get_input_embeddings()(
    #         input_ids).clone()

    #     vit_embeds = self.extract_feature(pixel_values)
    #     vit_embeds = vit_embeds.to(input_embeds.dtype)  # FIXME: why vit_embeds is float16?
    #     fast_vit_embeds = None

    #     vit_embeds = vit_embeds[image_flags == 1]
    #     vit_batch_size = pixel_values.shape[0]

    #     B, N, C = input_embeds.shape
    #     input_embeds = input_embeds.reshape(B * N, C)

    #     self._count += 1

    #     if vp_overall_mask is not None and prompt_masks is not None:
    #         vp_embeds = []
    #         vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
    #         prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

    #         vp_overall_mask = vp_overall_mask[image_flags == 1]
    #         overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

    #         i_vp_img = 0
    #         for i_img in range(len(vit_embeds)):
    #             vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
    #             if vp_overall_mask[i_img]:
    #                 tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
    #                 objects_prompt_masks = prompt_masks[i_vp_img]
    #                 n_obj = len(objects_prompt_masks)
    #                 tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
    #                 objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
    #                 vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
    #                 i_vp_img += 1
    #         vp_embeds = torch.cat(vp_embeds, dim=0)
    #     else:
    #         vp_embeds = None

    #     input_ids = input_ids.reshape(B * N)
    #     selected = (input_ids == self.img_context_token_id)

    #     if vp_embeds is None:
    #         try:
    #             input_embeds[selected] = vit_embeds.reshape(-1, C)
    #         except Exception as e:
    #             vit_embeds = vit_embeds.reshape(-1, C)
    #             print(f'warning: {e}, input_embeds[selected].shape='
    #                   f'{input_embeds[selected].shape}, '
    #                   f'vit_embeds.shape={vit_embeds.shape}')
    #             n_token = selected.sum()
    #             if n_token > len(vit_embeds):
    #                 print(f"Wrong !!! {n_token} image tokens in text but only {len(vit_embeds)} vit embeds !!!")
    #                 expand_ratio = n_token // len(vit_embeds) + 1
    #                 vit_embeds = torch.cat([vit_embeds] * expand_ratio, dim=0)

    #             input_embeds[selected] = vit_embeds[:n_token]
    #     else:
    #         try:
    #             input_embeds[selected] = vp_embeds.reshape(-1, C)
    #         except Exception as e:
    #             vp_embeds = vp_embeds.reshape(-1, C)
    #             print(f'warning: {e}, input_embeds[selected].shape='
    #                   f'{input_embeds[selected].shape}, '
    #                   f'vp_embeds.shape={vp_embeds.shape}')
    #             n_token = selected.sum()
    #             if n_token > len(vp_embeds):
    #                 print(f"Wrong !!! {n_token} image tokens in text but only {len(vp_embeds)} vit embeds !!!")
    #                 expand_ratio = n_token // len(vp_embeds) + 1
    #                 vp_embeds = torch.cat([vp_embeds] * expand_ratio, dim=0)

    #             input_embeds[selected] = vp_embeds[:n_token]

    #     input_embeds = input_embeds.reshape(B, N, C)

    #     outputs = self.language_model(
    #         inputs_embeds=input_embeds,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #     logits = outputs.logits

    #     loss = None
    #     if labels is not None:
    #         # Shift so that tokens < n predict n
    #         shift_logits = logits[..., :-1, :].contiguous()
    #         shift_labels = labels[..., 1:].contiguous()
    #         # Flatten the tokens
    #         loss_fct = CrossEntropyLoss()
    #         shift_logits = shift_logits.view(
    #             -1, self.language_model.config.vocab_size)
    #         shift_labels = shift_labels.view(-1)
    #         # Enable model parallelism
    #         shift_labels = shift_labels.to(shift_logits.device)
    #         loss = loss_fct(shift_logits, shift_labels)

    #     if not return_dict:
    #         output = (logits,) + outputs[1:]
    #         return (loss,) + output if loss is not None else output

    #     return CausalLMOutputWithPast(
    #         loss=loss,
    #         logits=logits,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
        # )


@torch.no_grad()
def generate(
        model,
        language_model,
        vision_model,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        prompt_masks=None,
        vp_overall_mask=None,
        **generate_kwargs,
) -> torch.LongTensor:
    device = "cuda"
    # pdb.set_trace()
    img_context_token_id = 151667

    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            if type(pixel_values) is list or pixel_values.ndim == 5:
                if type(pixel_values) is list:
                    pixel_values = [
                        x.unsqueeze(0) if x.ndim == 3 else x for x in pixel_values
                    ]
                # b*n, c, h, w
                pixel_values = torch.cat(
                    [image.to(torch.bfloat16) for image in pixel_values], dim=0)

            # pdb.set_trace()
            vit_embeds = extract_feature(vision_model, model.mlp1, pixel_values.to(device))
        image_flags = torch.sum(pixel_values, dim=(1, 2, 3)) != 0
        image_flags = image_flags.long()
        vit_embeds = vit_embeds[image_flags == 1]

        input_embeds = language_model.get_input_embeddings()(input_ids.to(device))
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if vp_overall_mask is not None and prompt_masks is not None:
            vp_embeds = []
            vp_overall_mask = vp_overall_mask.to(vit_embeds.device).bool()
            prompt_masks = [item.to(vit_embeds.device).bool() for item in prompt_masks]

            vp_overall_mask = vp_overall_mask[image_flags == 1]
            overall_tile_vit_embeds = vit_embeds[vp_overall_mask]  # (n_img, hw, c)

            i_vp_img = 0
            for i_img in range(len(vit_embeds)):
                vp_embeds.append(vit_embeds[i_img].reshape(-1, C))
                if vp_overall_mask[i_img]:
                    tile_vit_embeds = overall_tile_vit_embeds[i_vp_img].reshape(-1, C)  # (hw, C)
                    objects_prompt_masks = prompt_masks[i_vp_img]
                    n_obj = len(objects_prompt_masks)
                    tile_vit_embeds = tile_vit_embeds.unsqueeze(0).repeat(n_obj, 1, 1)
                    objects_prompt_masks = objects_prompt_masks.reshape(n_obj, -1)
                    vp_embeds.append(tile_vit_embeds[objects_prompt_masks])
                    i_vp_img += 1

            vp_embeds = torch.cat(vp_embeds, dim=0)
        else:
            vp_embeds = None

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == img_context_token_id)
        assert selected.sum() != 0
        if vp_embeds is None:
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        else:
            if len(input_embeds[selected]) != len(vp_embeds.reshape(-1, C)):
                print("Shape mismatch, selected is {}, vp embeds is {} !!!" \
                        .format(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C))))
                min_tokens = min(len(input_embeds[selected]), len(vp_embeds.reshape(-1, C)))
                input_embeds[selected][:min_tokens] = vp_embeds.reshape(-1, C)[:min_tokens].to(input_embeds.device)
            else:
                input_embeds[selected] = vp_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = language_model.get_input_embeddings()(input_ids)

    # pdb.set_trace()
    print("hey")
    outputs = language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask.to(device),
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        # return_dict=return_dict,
        use_cache=True,
        **generate_kwargs,
    )
    # pdb.set_trace()
    return outputs

# def preparing_for_generation(self, tokenizer, max_new_tokens=2048, torch_dtype=torch.bfloat16):
#     # set stop criteria and generation configs for model
#     if not hasattr(self, 'tokenizer'):
#         self.tokenizer = tokenizer
#     self.bot_name = 'BOT'
#     stop_words = []
#     stop_words += self.template.get('STOP_WORDS', [])
#     stop_criteria = get_stop_criteria(
#         tokenizer=self.tokenizer, stop_words=stop_words)
#     self.stop_criteria = stop_criteria

#     default_generation_kwargs = dict(
#         max_new_tokens=max_new_tokens,
#         do_sample=False,
#         eos_token_id=self.tokenizer.eos_token_id,
#         pad_token_id=(
#             self.tokenizer.pad_token_id
#             if self.tokenizer.pad_token_id is not None
#             else self.tokenizer.eos_token_id
#         ),
#     )

#     self.gen_config = GenerationConfig(**default_generation_kwargs)
#     self.init_prediction_config = True
#     self.torch_dtype = torch_dtype
#     self.to(torch_dtype)
#     self.extra_image_processor = DirectResize(target_length=1024, )
#     # for multi image process
#     self.min_dynamic_patch = 1
#     self.max_dynamic_patch = 12
#     self.downsample_ratio = 0.5
#     self.image_size = 448
#     self.use_thumbnail = True
#     patch_size = 14
#     self.patch_size = patch_size

#     self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
#     self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
#     self.IMAGENET_STD = (0.229, 0.224, 0.225)
#     self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
#     self.IMG_START_TOKEN = '<img>'
#     self.IMG_END_TOKEN = '</img>'

#     self.transformer = T.Compose([
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
#     ])
#     self.VP_START_TOKEN = '<vp>'
#     self.VP_END_TOKEN = '</vp>'

#     # change phi3 prepare for generation fuction
#     if self.config.llm_config.architectures[0] == 'Phi3ForCausalLM':
#         self.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_phi3, self.language_model)

#     img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
#     self.img_context_token_id = img_context_token_id
#     self.seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')
#     return


# def preparing_for_generation(tokenizer, max_new_tokens=2048, torch_dtype=torch.bfloat16):
#     # set stop criteria and generation configs for model
#     bot_name = 'BOT'
#     stop_words = []
#     stop_words += template.get('STOP_WORDS', [])
#     stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words=stop_words)

#     default_generation_kwargs = dict(
#         max_new_tokens=max_new_tokens,
#         do_sample=False,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=(
#             tokenizer.pad_token_id
#             if tokenizer.pad_token_id is not None
#             else tokenizer.eos_token_id
#         ),
#     )

#     gen_config = GenerationConfig(**default_generation_kwargs)
#     init_prediction_config = True
#     torch_dtype = torch_dtype
#     to(torch_dtype)  # This seems like a method call, but it's unclear. Replace accordingly
#     extra_image_processor = DirectResize(target_length=1024)

#     # for multi image process
#     min_dynamic_patch = 1
#     max_dynamic_patch = 12
#     downsample_ratio = 0.5
#     image_size = 448
#     use_thumbnail = True
#     patch_size = 14
#     patch_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
#     IMAGENET_MEAN = (0.485, 0.456, 0.406)
#     IMAGENET_STD = (0.229, 0.224, 0.225)
#     IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
#     IMG_START_TOKEN = '<img>'
#     IMG_END_TOKEN = '</img>'

#     transformer = T.Compose([
#         T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#         T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
#     ])
#     VP_START_TOKEN = '<vp>'
#     VP_END_TOKEN = '</vp>'

#     img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
#     img_context_token_id = img_context_token_id
#     seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')
    
#     return gen_config, transformer, extra_image_processor, img_context_token_id, seg_token_idx


def predict_forward(
        model,
        image=None,
        video=None,
        text=None,
        past_text='',
        mask_prompts=None,
        tokenizer=None,
        start=None,
        end=None,
):
    assert tokenizer
    # preparing_for_generation(tokenizer=tokenizer)
    
    #######################################################################
    max_new_tokens= 2048
    torch_dtype=torch.bfloat16
    
    grounding_encoder = SAM2()
    # grounding_encoder = model.grounding_encoder
    grounding_encoder.load_state_dict(model.grounding_encoder.state_dict(), strict=False)
    grounding_encoder.to(model.device).to(torch.bfloat16)
    # pdb.set_trace()
    
    in_dim = 2048
    out_dim = 256
    # text_hidden_fcs = nn.Sequential(
    #         nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
    #         nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
    #     )
    # text_hidden_fcs.load_state_dict(model.text_hidden_fcs.state_dict(), strict=False)
    # text_hidden_fcs.to(model.device).to(torch.bfloat16)
    # pdb.set_trace()
    
    template = {
        'INSTRUCTION': 'You are a helpful assistant. Please answer the following question: {input}.',
        'STOP_WORDS': ['<image>', '<img>', '</img>', '<vp>', '</vp>', '<IMG_CONTEXT>', '[SEG]']
    }
    bot_name = 'BOT'
    stop_words = []
    stop_words += template.get('STOP_WORDS', [])
    stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words=stop_words)

    default_generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )

    gen_config = GenerationConfig(**default_generation_kwargs)
    init_prediction_config = True
    extra_image_processor = DirectResize(target_length=1024)

    # for multi image process
    min_dynamic_patch = 1
    max_dynamic_patch = 12
    downsample_ratio = 0.5
    image_size = 448
    use_thumbnail = True
    patch_size = 14
    patch_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
    print("patch_token", patch_token)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    transformer = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    VP_START_TOKEN = '<vp>'
    VP_END_TOKEN = '</vp>'

    img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    img_context_token_id = img_context_token_id
    seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')
    ####################################################################
    
    input_dict = {}
    # if video is not None:
    #     pixel_values = []
    #     extra_pixel_values = []
    #     ori_image_size = video[0].size
    #     for frame_idx, frame_image in enumerate(video):
    #         assert ori_image_size == frame_image.size
    #         g_image = np.array(frame_image)  # for grounding
    #         g_image = extra_image_processor.apply_image(g_image)
    #         g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
    #         extra_pixel_values.append(g_image)
    #         # if frame_idx < 5:
    #         #     img = transformer(frame_image)
    #         #     pixel_values.append(img)
    #         img = transformer(frame_image)
    #         pixel_values.append(img)

    #     pixel_values = torch.stack(pixel_values, dim=0).to(torch_dtype)  # (n_f, 3, h, w)
    #     g_pixel_values = torch.stack([
    #         grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
    #     ]).to(torch_dtype)
    #     num_image_tokens = patch_token
    #     num_frames = len(pixel_values)
    #     print("num_frames", num_frames)

    #     input_dict['vp_overall_mask'] = None
    
    if video is not None:
        pixel_values = []
        extra_pixel_values = []
        ori_image_size = video[0].size
        
        total_frames = len(video)
        num_sample_frames = 10
        if start is not None and end is not None:
            frame_indices = np.linspace(start, end, num_sample_frames, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_sample_frames, dtype=int)
        
        # pdb.set_trace()
        
        for frame_idx in frame_indices:
            frame_image = video[frame_idx]
            assert ori_image_size == frame_image.size
            
            g_image = np.array(frame_image)  # for grounding
            g_image = extra_image_processor.apply_image(g_image)
            g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            extra_pixel_values.append(g_image)
            
            img = transformer(frame_image)
            pixel_values.append(img)

        pixel_values = torch.stack(pixel_values, dim=0).to(torch_dtype)  # (64, 3, h, w)
        g_pixel_values = torch.stack([
            grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
        ]).to(torch_dtype)
        
        num_image_tokens = patch_token
        num_frames = len(pixel_values)
        print(f"Sampled {num_frames} frames from original {total_frames} frames")

        input_dict['vp_overall_mask'] = None
        
    else:
        ori_image_size = image.size

        # prepare grounding images
        g_image = np.array(image)  # for grounding
        g_image = extra_image_processor.apply_image(g_image)
        g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(torch_dtype)
        extra_pixel_values = [g_pixel_values]
        g_pixel_values = torch.stack([
            grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
        ]).to(torch_dtype)

        images = dynamic_preprocess(image, min_dynamic_patch, max_dynamic_patch, image_size, use_thumbnail)

        if mask_prompts is not None:
            vp_overall_mask = torch.Tensor([False] * (len(images) - 1) + [True])
            input_dict['vp_overall_mask'] = vp_overall_mask
        else:
            input_dict['vp_overall_mask'] = None

        pixel_values = [transformer(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(torch_dtype)
        num_image_tokens = pixel_values.shape[0] * patch_token
        num_frames = 1
        
    input_dict['g_pixel_values'] = g_pixel_values
    input_dict['pixel_values'] = pixel_values

    if mask_prompts is not None:
        # reshape mask prompts to feature size
        mask_prompts = [torch.Tensor(item).to(pixel_values.device) for item in mask_prompts]
        mask_prompts = [F.interpolate(
            item.unsqueeze(0),
            size=(int(image_size // patch_size * downsample_ratio),
                    int(image_size // patch_size * downsample_ratio)),
            mode='nearest').squeeze(0) for item in mask_prompts]
        region_pixels = []
        for mask_prompt in mask_prompts[0]:
            region_pixels.append(mask_prompt.bool().to(torch.int64).sum())

        vp_token_str = '\nThere are {} part regions in the picture: '.format(len(mask_prompts[0]))
        for i in range(len(mask_prompts[0])):
            vp_token_str = vp_token_str + \
                            f"region{i + 1}" + VP_START_TOKEN + \
                            IMG_CONTEXT_TOKEN * region_pixels[i] + \
                            VP_END_TOKEN
            if i == len(mask_prompts[0]) - 1:
                vp_token_str = vp_token_str + '.\n'
            else:
                vp_token_str = vp_token_str + ', '
    else:
        vp_token_str = ''

    image_token_str = f'{IMG_START_TOKEN}' \
                        f'{IMG_CONTEXT_TOKEN * num_image_tokens}' \
                        f'{IMG_END_TOKEN}'
    # per frame has {num_image_tokens=256} tokens
    image_token_str = image_token_str + '\n'
    image_token_str = image_token_str * num_frames
    image_token_str = image_token_str.strip()

    ret_masks = []

    if '<image>' in text or mask_prompts is not None:
        assert past_text is None or len(past_text) == 0
    text = text.replace('<image>', image_token_str + vp_token_str)
    input_text = ''
    input_text += template['INSTRUCTION'].format(
        input=text, round=1, bot_name=bot_name)
    input_text = past_text + input_text
    ids = tokenizer.encode(input_text)
    ids = torch.tensor(ids).cuda().unsqueeze(0)

    attention_mask = torch.ones_like(ids, dtype=torch.bool)

    # pdb.set_trace()
    mm_inputs = {
        'model': model,
        'language_model': model.language_model,
        'vision_model': model.vision_model,
        'pixel_values': input_dict['pixel_values'],
        'input_ids': ids,
        'attention_mask': attention_mask,
        'position_ids': None,
        'past_key_values': None,
        'labels': None,
        'prompt_masks': mask_prompts,
        'vp_overall_mask': input_dict['vp_overall_mask'],
        }

    # pdb.set_trace()
    generate_output = generate(
        **mm_inputs,
        generation_config=gen_config,
        streamer=None,
        bos_token_id=tokenizer.bos_token_id,
        stopping_criteria=stop_criteria,
        output_hidden_states=True,
        return_dict_in_generate=True
    )
    predict = tokenizer.decode(
        generate_output.sequences[0], skip_special_tokens=False).strip()

    # pdb.set_trace()
    
    if image is None and video is None and '<image>' not in past_text:
        return {'prediction': predict, 'prediction_masks': ret_masks, }

    # if have seg result, find the seg hidden states
    hidden_states = generate_output.hidden_states
    last_hidden_states = [item[-1][0] for item in hidden_states]
    last_hidden_states = torch.cat(last_hidden_states, dim=0)
    # pdb.set_trace()
    seg_hidden_states = get_seg_hidden_states(
        tokenizer, last_hidden_states, generate_output.sequences[0][:-1],
        # tokenizer, last_hidden_states, generate_output.sequences[0],
        seg_id=seg_token_idx
    )
    
    pdb.set_trace()
    all_seg_hidden_states = text_hidden_fcs(seg_hidden_states)

    for seg_hidden_states in all_seg_hidden_states:
        seg_hidden_states = seg_hidden_states.unsqueeze(0)
        g_pixel_values = input_dict['g_pixel_values']
        # pdb.set_trace()
        sam_states = grounding_encoder.get_sam2_embeddings(g_pixel_values)
        pred_masks = grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * num_frames)
        w, h = ori_image_size
        masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
        masks = masks[:, 0]
        masks = masks.sigmoid() > 0.5
        masks = masks.cpu().numpy()
        ret_masks.append(masks)
    
    print(f"The prediction is:\n{predict}")
    
    if video is not None:
        return {'prediction': predict, 'prediction_masks': ret_masks, 'frame_indices': frame_indices}
    else:
        return {'prediction': predict, 'prediction_masks': ret_masks}


def get_seg_hidden_states(tokenizer, hidden_states, output_ids, seg_id):
    # pdb.set_trace()
    seg_mask = output_ids == seg_id
    # tokenizer.decode(output_ids.cpu())
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

from transformers.cache_utils import Cache, DynamicCache

# def prepare_inputs_for_generation_phi3(
#         self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
# ):
#     if past_key_values is not None:
#         if isinstance(past_key_values, Cache):
#             cache_length = past_key_values.get_seq_length()
#             past_length = past_key_values.seen_tokens
#             max_cache_length = past_key_values.get_max_length()
#         else:
#             cache_length = past_length = past_key_values[0][0].shape[2]
#             max_cache_length = None

#         # Keep only the unprocessed tokens:
#         # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
#         # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
#         # input)
#         if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
#             input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
#         # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
#         # input_ids based on the past_length.
#         elif past_length < input_ids.shape[1]:
#             input_ids = input_ids[:, past_length:]
#         # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

#         # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
#         if (
#                 max_cache_length is not None
#                 and attention_mask is not None
#                 and cache_length + input_ids.shape[1] > max_cache_length
#         ):
#             attention_mask = attention_mask[:, -max_cache_length:]

#     position_ids = kwargs.get('position_ids', None)
#     if attention_mask is not None and position_ids is None:
#         # create position_ids on the fly for batch generation
#         position_ids = attention_mask.long().cumsum(-1) - 1
#         position_ids.masked_fill_(attention_mask == 0, 1)
#         if past_key_values:
#             position_ids = position_ids[:, -input_ids.shape[1]:]

#     # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
#     if inputs_embeds is not None and (past_key_values is None or len(past_key_values)==0):
#         model_inputs = {'inputs_embeds': inputs_embeds}
#     else:
#         model_inputs = {'input_ids': input_ids}

#     model_inputs.update(
#         {
#             'position_ids': position_ids,
#             'past_key_values': past_key_values,
#             'use_cache': kwargs.get('use_cache'),
#             'attention_mask': attention_mask,
#         }
#     )
#     return model_inputs

# def language_embd_inference(self, inference_state, language_embd):
#     num_frame = len(language_embd)
#     num_obj = len(language_embd[0])
#     # mask_out = []
#     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#         for frame_idx in range(num_frame):
#             frame_mask_out = []

#             for obj_idx in range(num_obj):
#                 _language_embd = language_embd[frame_idx][obj_idx][None][None]
#                 _, _, out_mask_logits = self.sam2_model.add_language_embd(
#                     inference_state,
#                     frame_idx,
#                     obj_idx + 100,
#                     _language_embd,
#                     inference=True,
#                 )
#                 frame_mask_out.append(out_mask_logits)
#             # frame_mask_out = torch.cat(frame_mask_out, dim=1)
#             # mask_out.append(frame_mask_out)

#         mask_out = []
#         for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(inference_state):
#             mask_out.append(out_mask_logits)
#         mask_out = torch.cat(mask_out, dim=0)
        
#     return mask_out

# def get_sam2_embeddings(self, images):
#     return self.sam2_model.init_state(images)