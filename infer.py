import argparse
import json
import math
import os
import queue
import time

import yaml
from datetime import datetime
import logging

from builder.inferModel_builder import build as model_builder
from dataset import create_data_loader
from tqdm import tqdm
# Get the current time
def set_save_path(config, num_chunks, chunk_idx):
    now = datetime.now()
    timestamp = now.strftime("%m%d%H%M")
    base_dir = os.path.join(config['save_path'], config['id'])
    save_dir = os.path.join(base_dir, f"{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    log_filename = f"{num_chunks}-{chunk_idx}.log"
    log_path = os.path.join(save_dir, log_filename)
    return os.path.join(save_dir, "output.jsonl"), log_path

def lock_file(file_path):
    # 检查锁文件是否存在，如果存在则等待
    while os.path.exists(file_path + ".lock"):
        time.sleep(0.1)
    # 创建锁文件
    open(file_path + ".lock", "w").close()

def unlock_file(file_path):
    if os.path.exists(file_path + ".lock"):
        try:
            os.remove(file_path + ".lock")
        except:
            pass
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file) 
   
    save_path, log_path = set_save_path(config, args.num_chunks, args.chunk_idx)
    
    # 配置日志
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"strat")
    model=model_builder(config)
    json_file=json.load(open(config['json_path']))
    json_file_chunk=get_chunk(json_file, args.num_chunks, args.chunk_idx)

    if os.path.exists(save_path + ".lock"):
        os.remove(save_path + ".lock")
    dataloader=create_data_loader(json_file_chunk,config['image_path'],batch_size=args.batchsize)

    for batch_index,batch in enumerate(tqdm(dataloader)):
        #try:
            output=model(batch)
            print(output)
            lock_file(save_path)
            with open(save_path, "a", encoding='utf-8') as ans_file:
                for i,item in enumerate(batch):
                    item['predict']=output[i]
                    ans_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    ans_file.flush()
            unlock_file(save_path)
            logging.info(f"Processed batch {batch_index}: {batch}")
        #except Exception as e:
        #    logging.info(f"error batch {batch_index}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=4)
    args = parser.parse_args()
    eval_model(args)