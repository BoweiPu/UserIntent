import argparse
import json
import math
import os
import queue
import re
import time

import yaml
from datetime import datetime
import logging


from dataset import create_data_loader
from tqdm import tqdm
def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]

# Get the current time
def set_save_path(config, num_chunks, chunk_idx):
    save_dir = os.path.join(config['save_path'], config['id'])
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
