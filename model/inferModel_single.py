import re
import torch
import torch.nn as nn
from builder.LLM_builder import build_llm
from builder.ocr_builder import build_ocr
from transformers import  AutoProcessor
from qwen_vl_utils import process_vision_info
from abc import ABC, abstractmethod

def has_repeated_sequence(text):
    # 确保字符串长度足够
    if len(text) < 60:
        return False

    # 使用一个字典存储出现过的子串
    seen = {}
    
    for length in range(4, len(text) // 2 + 1):
        for i in range(len(text) - length * 9 + 1):
            subsequence = text[i:i + length]
            # 如果当前子串已经出现过并且出现次数超过9次，返回True
            if subsequence in seen:
                if seen[subsequence] >= 9:
                    return True
            # 记录当前子串
            seen[subsequence] = seen.get(subsequence, 0) + 1
    return False

class InferModel_single(nn.Module):
    def __init__(self,path_llm='/home/pubw/proj/Qwen2-VL-7B-Instruct'):
        super().__init__()
        self.llm=build_llm(path_llm)
        self.processor = AutoProcessor.from_pretrained("/home/pubw/proj/Qwen2-VL-7B-Instruct")


    # messages:[messages1, messages2,]
    @torch.no_grad()
    def infer_llm(self,messages):
        
        texts = [
        self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )       
        inputs = inputs.to("cuda")
        output_ids = self.llm.generate(**inputs, max_new_tokens=8192)
        
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text
    
    
    
    def forward(self,image_path_list,text):
        item_messages = []
        for image_path in image_path_list:
            item_messages.append( {
                    "type": "image",
                    "image": image_path  
                })
        item_messages.append({
                "type": "text",
                "text": text
            })
        input_mes=[[
            {
                "role": "user",
                "content": item_messages
            }]]
        #print(input_mes)
        output=self.infer_llm(input_mes)
        #while has_repeated_sequence(output):
         #   output=self.infer_llm(input_mes)
        return output