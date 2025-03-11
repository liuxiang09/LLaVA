#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig
    # 模型初始化
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)    # 设置基础模型
        self.pretraining_tp = config.pretraining_tp # 配置预训练参数
        self.vocab_size = config.vocab_size # 词表大小
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # 语言模型头部 

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    # 向前传播方法：支持多模态输入处理，在处理文本输入前会先处理图像输入，通过 prepare_inputs_labels_for_multimodal 方法处理多模态输入
    def forward(
        self,
        input_ids: torch.LongTensor = None,                             # 输入文本的 token id
        attention_mask: Optional[torch.Tensor] = None,                  # 注意力掩码
        position_ids: Optional[torch.LongTensor] = None,                # 位置编码
        past_key_values: Optional[List[torch.FloatTensor]] = None,      # 用于存储过去的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,              # 输入的嵌入向量
        labels: Optional[torch.LongTensor] = None,                      # 标签
        use_cache: Optional[bool] = None,                               # 是否使用缓存     
        output_attentions: Optional[bool] = None,                       # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,                    # 是否输出隐藏状态
        images: Optional[torch.FloatTensor] = None,                     # 图像输入
        image_sizes: Optional[List[List[int]]] = None,                  # 图像尺寸
        return_dict: Optional[bool] = None,                             # 是否返回字典
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    # 生成方法：实现了条件文本生成功能，支持图文多模态输入，处理位置编码和注意力掩码
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,   # 其他参数
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        # 检查是否有inputs_embeds参数，如果有则抛出异常
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(  # 处理多模态输入：提取图像特征，将图像特征与文本输入进行对齐，生成相应的位置编码和注意力掩码
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:       # 如果没有图像输入，则直接使用embed_tokens方法获取文本的嵌入向量
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        # 调用父类的generate方法生成文本
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)     # 获取图像输入
        image_sizes = kwargs.pop("image_sizes", None)       # 获取图像尺寸
        inputs = super().prepare_inputs_for_generation(     # 调用父类的prepare_inputs_for_generation方法
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:          # 如果有图像输入，则将图像输入添加到inputs中
            inputs['images'] = images
        if image_sizes is not None:     # 如果有图像尺寸，则将图像尺寸添加到inputs中
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)                     # 注册配置类
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)   # 注册模型类
