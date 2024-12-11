import re
import torch
import torch.nn as nn
from builder.LLM_builder import build_llm
from transformers import  AutoTokenizer,AutoModelForCausalLM




class NLP_model(nn.Module):
    def __init__(self,path_llm='/home/pubw/proj/Qwen2.5-7B-Instruct'):
        super().__init__()
        self.llm=AutoModelForCausalLM.from_pretrained(
            path_llm,
            torch_dtype="auto",
            device_map='balanced_low_0',attn_implementation="flash_attention_2",
        )
        #self.processor = AutoProcessor.from_pretrained("/home/pubw/proj/Qwen2.5-7B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(path_llm)


    # messages:[messages1, messages2,]
    @torch.no_grad()
    def infer_llm(self,prompt):
        
        #prompt = "Give me a short introduction to large language model."
        messages = [
            {"role": "system", "content": " You are a helpful assistant, 用中文回复我的问题"},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)

        generated_ids = self.llm.generate(
            **model_inputs,
            max_new_tokens=2048,
            #do_sample=True,           # 启用采样而不是贪心搜索或束搜索
            #temperature=0.8,          # 控制输出的随机性（值越高，输出越随机）
            #top_p=0.9,                # 核采样：只从累积概率达到此值的词汇子集中选择下一个词
            #top_k=5,                 # top-k 采样：仅考虑最有可能的 k 个词进行采样
            #repetition_penalty=1.2,   # 对重复出现的词语施加惩罚，防止模型生成重复内容
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    
    
    
    def forward(self,text):

        return self.infer_llm(text)