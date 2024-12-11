import argparse
import json
import math
import os
import queue
import time

import yaml
from datetime import datetime
import logging


from dataset import create_data_loader
from tqdm import tqdm

from pbw_RAGprompt.NLPModel import NLP_model
from pbw_RAGprompt.prompt import PROMPTS,task2_label_list
from pbw_RAGprompt.utis import *
import re

def open_jsonl(jsonl_path):
    items={}
    with open(jsonl_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            json_item=json.loads(line)
            #print(json_item['caption4label'])
            sp_str=split_string_by_multi_markers(json_item['caption4label'][0],['##','<|完成|>'])
            if len (sp_str)>15:
                continue
            attribuate_list=[]
            for sub_str in sp_str:
                sub_str=re.search(r"\((.*)\)", sub_str)
                if sub_str is None:
                    continue
                sub_str=sub_str.group(1)
                #print(sub_str)
                attribuate_list.append(split_string_by_multi_markers(sub_str,['<|>']))
            if json_item['output'] not in items:
                items[json_item['output']]=[]
            items[json_item['output']].append(attribuate_list)
    return items
def rebuild_info(caption_list):
    local_list=["中部","顶部","底部","页面某处"]
    def set_local(local_input,local_list):
        for local in local_list[:-1]:
            if local in local_input:
                return local
        return "页面某处"
    attribute_list={}
    for label in caption_list:
        i=0
        prompt_list={}
        for item in caption_list[label]:
            i+=1
            sample_classication={}
            for class_local in local_list:
                sample_classication[class_local]=[]
            for attribute_item in item:
                local=set_local(attribute_item[2],local_list)
                #print(attribute_item[2])
                sample_classication[local].append(attribute_item)

            for class_local in local_list:
                if class_local not in prompt_list:
                    prompt_list[class_local]=[]
                if len(sample_classication[class_local])>0:
                    prompt_list[class_local].append(sample_classication[class_local])          
        attribute_list[label]=prompt_list
    return attribute_list
def setup(config):
    save_path, log_path = set_save_path(config, args.num_chunks, args.chunk_idx)
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')    
    model=NLP_model(path_llm=config['llm_path'])
    caption_list=open_jsonl(config['caption_path'])
    if os.path.exists(save_path + ".lock"):
        os.remove(save_path + ".lock")

    return save_path,model,caption_list


#def llm_infer(save_path,model,dataloader,template):
    

def update_summary(cur_summary,new_sum):
    pass
def get_middle_num(samples_text,return_num=10):
    # 按照字符串长度排序
    sorted_samples = sorted(samples_text, key=len)
    
    # 计算中间位置
    n = len(sorted_samples)
    middle_pos = n // 2
    
    # 如果列表长度小于等于return_num，则返回整个列表
    if n <= return_num:
        return sorted_samples
    
    # 计算开始和结束的位置以获取中间的return_num个元素
    start_pos = max(0, middle_pos - return_num//2)  # 不低于列表起始位置
    end_pos = min(n, middle_pos + return_num//2)    # 不超过列表结束位置
    
    # 获取并返回中间的return_num个元素
    middle_sub = sorted_samples[start_pos:end_pos]
    
    # 如果获取到的元素少于return_num个，可以尝试调整范围
    # 这里简单处理，如果不足return_num个则返回尽可能多的中间元素
    if len(middle_sub) < return_num and start_pos > 0:
        # 尝试从左边补充元素
        while len(middle_sub) < return_num and start_pos > 0:
            start_pos -= 1
            middle_sub = sorted_samples[start_pos:end_pos]
    
    return middle_sub
def summury_caption(caption_list,save_path,model):
    local_summary=PROMPTS['local_summary']
    all_input=[]
    caption_list=rebuild_info(caption_list)
    for label in caption_list:
        if label=="其他类别图片" : 
            continue
        for local in caption_list[label]:
            i=0
            samples_text=''
            samples_text_list=[]
            samples_list= caption_list[label][local]
            samples_list=get_middle_num(samples_list)
            #print(samples_list)
            for attribute_list in  samples_list:
                i+=1
                formal_list=[]
                for sample in  attribute_list:
                    del sample[1]
                    formal_list.append("("+"<|>".join(sample)+")")
                samples_text=f"样本{i}中有元素:"+"##\n".join(formal_list)
                samples_text_list.append(samples_text)
            input_text=local_summary.format(label=label,local=local,attribute="\n".join(samples_text_list))
            all_input.append((label,local,input_text))

    for label,local,summary_input in all_input:
        print('summary_input----',summary_input)
        output=model(summary_input)
        print('output-----',output)
        with open(save_path, "a", encoding='utf-8') as ans_file:
            ans_file.write(json.dumps({
                label+'-'+local:output
            }, ensure_ascii=False) + "\n")
            ans_file.flush()
        


def run(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file) 
    save_path,model,caption_list=setup(config)

    #step2 各个类元素总结
    summury_caption(caption_list,save_path,model)
    #step3 各个类元素去重
    #step4 生成分类依据

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/pubw/proj/UserIntent/pbw_RAGprompt/rag_prompt_gen.yaml")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=4)
    args = parser.parse_args()
    run(args)