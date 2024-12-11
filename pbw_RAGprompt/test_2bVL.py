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

from pbw_RAGprompt.prompt import PROMPTS,task2_label_list
from model.inferModel_single import InferModel_single as infer_model

from pbw_RAGprompt.utis import *

def setup(config):
    save_path, log_path = set_save_path(config, args.num_chunks, args.chunk_idx)
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')    
    model=infer_model(path_llm=config['llm_path'])
    json_file=json.load(open(config['json_path']))
    json_file_chunk=get_chunk(json_file, args.num_chunks, args.chunk_idx)
    if os.path.exists(save_path + ".lock"):
        os.remove(save_path + ".lock")
    dataloader=create_data_loader(json_file_chunk,config['image_path'],batch_size=1)
    return save_path,model,dataloader


#def llm_infer(save_path,model,dataloader,template):
    


def creat_caption(save_path,model,dataloader):
    template=PROMPTS['get_attribute']
    for batch_index,batch in enumerate(tqdm(dataloader)):
            batch=batch[0]
            #print(items[batch['id']]['caption4label'])
            text=template
            #print('input',text)
            try:
                output=model(image_path_list=batch["image"],text=text)
                lock_file(save_path)
                print(output)
                with open(save_path, "a", encoding='utf-8') as ans_file:
                    
                    batch['attributes']=output
                    del batch['instruction']
                    ans_file.write(json.dumps(batch, ensure_ascii=False) + "\n")
                    ans_file.flush()
                unlock_file(save_path)
                logging.info(f"Processed batch {batch_index}: {batch}")
            except Exception as e:
                logging.info(f"error batch {batch_index}: {e}")




def creat_prompt(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file) 
    save_path,model,dataloader=setup(config)
    
    #step1 总结分类原因
    creat_caption(save_path,model,dataloader)
    #step2 各个类元素总结
    #step3 各个类元素去重
    #step4 生成分类依据

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/pubw/proj/UserIntent/pbw_RAGprompt/rag_prompt_gen.yaml")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=4)
    args = parser.parse_args()
    creat_prompt(args)