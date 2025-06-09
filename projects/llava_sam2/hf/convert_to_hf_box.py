import argparse
import copy
import os.path as osp
import torch
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict
import os
import pdb

def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')
    parser.add_argument('--config', default='projects/llava_sam2/configs/sa2va_8b_box.py', help='config file name or path.')
    # parser.add_argument('--pth-model', default='debug_system_prompt/iter_16600.pth', help='pth model file')
    parser.add_argument('--pth-model', default='debug_refcoco/iter_2000.pth', help='pth model file')
    parser.add_argument('--save-path', type=str, default='work_dirs/hf_model_refcoco', help='save folder name')
    args = parser.parse_args()
    return args

@master_only
def master_print(msg):
    print(msg)

def main():
    args = parse_args()

    # build model
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    model = BUILDER.build(cfg.model)
    backend = get_file_backend(args.pth_model)

    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)
        
    keys_to_remove = [key for key in state_dict.keys() if 'ca_layer' in key]
    for key in keys_to_remove:
        del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    model._merge_lora()
    model.mllm.transfer_to_hf = True
    
    pdb.set_trace()

    # build the hf format model
    from projects.llava_sam2.hf.models.configuration_sa2va_chat import Sa2VAChatConfig
    from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel

    internvl_config = Sa2VAChatConfig.from_pretrained(cfg.path)
    config_dict = internvl_config.to_dict()
    
    config_dict['auto_map'] = \
        {'AutoConfig': 'configuration_sa2va_chat.Sa2VAChatConfig',
         'AutoModel': 'modeling_sa2va_chat.Sa2VAChatModel',
         'AutoModelForCausalLM': 'modeling_sa2va_chat.Sa2VAChatModel'}

    config_dict["llm_config"]["vocab_size"] = len(model.tokenizer)
    config_dict["template"] = cfg.template
    print("template", config_dict["template"])
    
    sa2va_hf_config = Sa2VAChatConfig(
        **config_dict
    )
    
    hf_sa2va_model = Sa2VAChatModel(
        sa2va_hf_config, vision_model=model.mllm.model.vision_model,
        language_model=model.mllm.model.language_model,
    )
    hf_sa2va_model.load_state_dict(state_dict, strict=False)

    hf_sa2va_model.save_pretrained(args.save_path)
    model.tokenizer.save_pretrained(args.save_path)
    print(f"Save the hf model into {args.save_path}")

    # copy the files
    os.system(f"cp -pr ./projects/llava_sam2/hf/models/* {args.save_path}")

if __name__ == '__main__':
    main()


# model = Sa2VAChatModel.from_pretrained('work_dirs/hf_model',torch_dtype=torch.float16,device_map="auto",trust_remote_code=True,local_files_only=True)

# tokenizer = AutoTokenizer.from_pretrained(
#     'output',
#     trust_remote_code=True
# )