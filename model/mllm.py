import torch
import torch.nn as nn
from builder.LLM_builder import build_llm
from builder.ocr_builder import build_ocr
from transformers import  AutoProcessor
from qwen_vl_utils import process_vision_info
from abc import ABC, abstractmethod



class InferModel_multi(nn.Module):
    def __init__(self,vl_path='/home/pubw/proj/Qwen2-VL-7B-Instruct'):
        super().__init__()
        self.llm_=build_llm(path)
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
        output_ids = self.llm.generate(**inputs, max_new_tokens=1024)
        
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return output_text
    
    
    def forward(self,batch):
        messages_7b=[]
        for sub_i,data_item in enumerate(batch):
            img_list=data_item['image']
            item_messages=[]
            for i,img_item in enumerate(img_list):
                image_message = {
                    "type": "image",
                    "image": img_item  
                }
                item_messages.append(image_message)
            
            item_messages.append({
                "type": "text",
                "text": data_item["instruction"]
            })
        #形成batch
            messages_7b.append([{
                "role": "user",
                "content": item_messages
            }])
        
        output=self.infer_llm(messages_7b)
        return output