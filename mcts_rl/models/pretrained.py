# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
# from peft import PeftModelForCausalLM

from mcts_rl.configs import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_UNK_TOKEN,
)
from mcts_rl.models.score_model import AutoModelForScore
from mcts_rl.utils import is_main_process


# Reference: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
def resize_tokenizer_embedding(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel) -> None:
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

    def verify_vocabulary_embedding_sizes(
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        format_message: Callable[[Any, Any], str],
    ) -> None:
        input_embeddings = model.get_input_embeddings()
        if (
            input_embeddings is not None
            and input_embeddings.num_embeddings != len(tokenizer)
            and is_main_process()
        ):
            warnings.warn(
                format_message(len(tokenizer), input_embeddings.num_embeddings),
                category=RuntimeWarning,
                stacklevel=3,
            )

    def init_new_embeddings(embeddings: nn.Embedding | None) -> None:
        if embeddings is None:
            return

        embeddings_data = embeddings.weight.data
        embeddings_mean = embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        embeddings_data[-num_new_tokens:] = embeddings_mean

    verify_vocabulary_embedding_sizes(
        tokenizer=tokenizer,
        model=model,
        format_message=(
            'The tokenizer vocabulary size ({}) is different from '
            'the model embedding size ({}) before resizing.'
        ).format,
    )

    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        # special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        # special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
        tokenizer.unk_token_id = tokenizer.eos_token_id

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if num_new_tokens > 0:
        hf_device_map = getattr(model, 'hf_device_map', {})
        devices = {
            torch.device(device)
            for device in hf_device_map.values()
            if device not in {'cpu', 'disk'}
        }
        is_model_parallel = len(devices) > 1

        if not is_model_parallel:
            model.resize_token_embeddings(len(tokenizer))
            init_new_embeddings(model.get_input_embeddings())
            init_new_embeddings(model.get_output_embeddings())

    verify_vocabulary_embedding_sizes(
        tokenizer=tokenizer,
        model=model,
        format_message=(
            'The tokenizer vocabulary size ({}) is different from '
            'the model embedding size ({}) after resizing.'
        ).format,
    )


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)


def load_pretrained_models(  # pylint: disable=too-many-arguments
    model_name_or_path: str | os.PathLike,
    /,
    model_max_length: int = 512,
    padding_side: Literal['left', 'right'] = 'right',
    auto_device_mapping: bool = False,
    dtype: torch.dtype | str | None = 'auto',
    *,
    cache_dir: str | os.PathLike | None = None,
    trust_remote_code: bool = False,
    auto_model_type: type[AutoModelForCausalLM | AutoModelForScore] = AutoModelForCausalLM,
    auto_model_args: tuple[Any, ...] = (),
    auto_model_kwargs: dict[str, Any] | None = None,
    auto_tokenizer_args: tuple[Any, ...] = (),
    auto_tokenizer_kwargs: dict[str, Any] | None = None,
    lora_enable: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_weight_path: str = '',
    lora_bias: str = 'none',
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load pre-trained model and tokenizer from a given path.

    Args:
        model_name_or_path (str or os.PathLike): Path to the model or its name.
        model_max_length (int, optional): The maximum sequence length of the model. Defaults to 512.
        padding_side (str, optional): The side to pad by the tokenizer. Defaults to 'right'.
        auto_device_mapping (bool, optional): Whether to automatically map the model to the multiple
            devices. Defaults to False.
        dtype (torch.dtype or str or None, optional): The parameter dtype while loading the model.
            Defaults to 'auto'.
        cache_dir (str or os.PathLike or None, optional): The directory to cache the model. Defaults
            to None.
        trust_remote_code (bool, optional): Whether to trust the remote code. Defaults to False.
        auto_model_type (type[AutoModelForCausalLM] or type[AutoModelForScore], optional): The type
            of the model to load. Defaults to AutoModelForCausalLM.
    """
    model_name_or_path = os.path.expanduser(model_name_or_path)
    cache_dir = os.path.expanduser(cache_dir) if cache_dir is not None else None
    device_map = 'auto' if auto_device_mapping else None
    if auto_model_kwargs is None:
        auto_model_kwargs = {}
    if auto_tokenizer_kwargs is None:
        auto_tokenizer_kwargs = {}

    model = auto_model_type.from_pretrained(
        model_name_or_path,
        *auto_model_args,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        **auto_model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        *auto_tokenizer_args,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        trust_remote_code=trust_remote_code,
        use_fast=(model.config.model_type != 'llama'),
        **auto_tokenizer_kwargs,
    )
    resize_tokenizer_embedding(tokenizer=tokenizer, model=model)
    
    # if lora_enable and auto_model_type != PeftModelForCausalLM:
    #     from peft import LoraConfig, get_peft_model
    #     lora_config = LoraConfig(
    #         r=lora_r,
    #         lora_alpha=lora_alpha,
    #         target_modules=['q_proj', 'v_proj'],    # find_all_linear_names(model),
    #         lora_dropout=lora_dropout,
    #         bias=lora_bias,
    #         task_type="FEATURE_EXTRACTION" if auto_model_type == AutoModelForScore else "CAUSAL_LM",
    #     )
    #     model = get_peft_model(model, lora_config)
    
    return model, tokenizer
