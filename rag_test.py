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
import json
import logging
import re

import jieba
import torch
from model.inferModel_base import InferModel_base
from builder.ocr_builder import build_ocr, extract_ocr_info
from pbw_RAGprompt.prompt import PROMPTS
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
import cv2
from PIL import Image
import torch.nn as nn
import cv2
import numpy as np
def mask_and_fill_with_external_values(image_path, polygons):
    """
    使用多边形区域外部值填充多边形区域。
    
    :param image_path: 输入图片路径。
    :param polygons: 多边形坐标列表，例如 [[[x1, y1], [x2, y2], ...], ...]
    :param output_path: 输出图片路径。
    """
    # 1. 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法加载图片，请检查路径是否正确。")
    
    # 2. 创建掩膜
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for polygon_coords in polygons:
        polygon_coords = np.array([polygon_coords], dtype=np.int32)
        cv2.fillPoly(mask, polygon_coords, 255)
    
    # 3. 使用掩膜外部值进行填充
    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    

    return result

def global_blur(image_path, blur_kernel=(89, 89)):
    """
    对图片进行全局模糊处理。
    
    :param image_path: 输入图片路径。
    :param blur_kernel: 模糊核大小，例如 (25, 25)。
    :param output_path: 输出图片路径。
    """
    # 1. 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法加载图片，请检查路径是否正确。")
    
    # 2. 全局模糊处理
    blurred_image = cv2.GaussianBlur(image, blur_kernel, 0)

    return blurred_image

rag_item="""Example {0}: Picture {1}: <image>, 标签是\"{2}\"\n"""
rag_template="""你是一个电商领域识图专家,可以理解消费者上传的软件截图或实物拍摄图。现在,请你对消费者上传的图片进行分类。你只需要回答图片分类结果,不需要其他多余的话。以下是可以参考的分类标签,分类标签:[\"实物拍摄(含售后)\",\"商品分类选项\",\"商品头图\",\"商品详情页截图\",\"下单过程中出现异常（显示购买失败浮窗）\",\"订单详情页面\",\"支付页面\",\"消费者与客服聊天页面\",\"评论区截图页面\",\"物流页面-物流列表页面\",\"物流页面-物流跟踪页面\",\"物流页面-物流异常页面\",\"退款页面\",\"退货页面\",\"换货页面\",\"购物车页面\",\"店铺页面\",\"活动页面\",\"优惠券领取页面\",\"账单/账户页面\",\"个人信息页面\",\"投诉举报页面\",\"平台介入页面\",\"外部APP截图\",\"其他类别图片\"]\n
可以参考的例子是[{example}]\n。需要分类的输入是Picture {count}: <image>\n"""

def split_conv(conv):
    user_dialogues = []
    service_dialogues=[]
    # 使用 re.split 按照关键点切分字符串
    lines = conv.split("\n")
    for line in lines:
        if line.startswith("用户:"):
            # 提取用户的内容
            #content = line[3:].strip()
            user_dialogues.append(line)
        elif line.startswith("客服:"):
            service_dialogues.append(line)
    
    return user_dialogues,service_dialogues

class infer_rag(nn.Module):
    def __init__(self,
                 pt_path="/home/pubw/proj/UserIntent/feat/remain_key.pt",
                 image_Augmented="key",
                 clip_path="/home/pubw/proj/clip-vit-large-patch14",
                 key_words_path='/home/pubw/proj/UserIntent/utils/task2_text_frequency_top20.json',
                 topk=3,
                 rag_image_path="",
                 **kwargs):
        super().__init__()
        
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_path).cuda()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        load_pt=torch.load(pt_path)
        retrival_embeds=load_pt['feat'].cuda()
        retrival_embeds=retrival_embeds/ retrival_embeds.norm(dim=-1, keepdim=True)
        self.retrival_embeds=retrival_embeds.float()
        self.retrival_info=load_pt['info']
        key_words=json.load(open(key_words_path))
        key_words = [item for sublist in list(key_words.values()) for item in sublist]
        self.key_words=key_words
        self.image_Augmented=image_Augmented
        self.use_ocr= (image_Augmented !="not" and image_Augmented!="layout")
        self.topk=topk
        self.rag_image_path=rag_image_path
        if self.use_ocr:
            self.ocr=build_ocr()
            #logging.setLevel(logging.INFO)
            logging.getLogger('ppocr').setLevel(logging.ERROR)
    

    def get_ocr_mask(self,img_path):
        mask_list=[]
        ocr_res=self.ocr.ocr(img_path, cls=False)[0]
        if ocr_res is None:
            image_pil=Image.open(img_path).convert('RGB')
            return image_pil
        for ocr_tiem in ocr_res:
            if any(keyword in jieba.lcut(ocr_tiem[1][0]) for keyword in self.key_words) and "key" in  self.image_Augmented:
                continue
            #print(ocr_tiem[1])
            mask_list.append(ocr_tiem[0])
        image_bgr=mask_and_fill_with_external_values(img_path,mask_list)
        image_rgb =cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # 将NumPy数组转换为PIL图像对象，并指定模式为RGB
        image_pil = Image.fromarray(image_rgb, mode='RGB')
        return image_pil

    def rag_res(self,img_path):
        if self.use_ocr:
            image_pil=self.get_ocr_mask(img_path)
        elif self.image_Augmented=='layout':
            image_bgr=global_blur(img_path)
            image_rgb =cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            # 将NumPy数组转换为PIL图像对象，并指定模式为RGB
            image_pil = Image.fromarray(image_rgb, mode='RGB')
        else:
            image_pil=Image.open(img_path).convert('RGB')
        #print(image_pil)
        inputs =self.clip_processor(images=[image_pil], return_tensors="pt")
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        image_embeds = self.clip_model(**inputs).image_embeds
        
        image_embeds=image_embeds/ image_embeds.norm(dim=-1, keepdim=True)
        
        scores=image_embeds@self.retrival_embeds.t()
        scores[torch.isclose(scores, torch.tensor(1.0))] = 0
        _, topk_indices = torch.topk(scores, self.topk, dim=-1)
        return topk_indices[0][1:]
        
    
    def forward(self,batch):
        messages_7b=[]
        for sub_i,data_item in enumerate(batch):
            item_messages=[]
            instruction=[]
            img_list=data_item['image']
            rag_examples=[]
            image_count=0
            rag_labels=[]
            for i,img_item in enumerate(img_list):
                image_message = {
                    "type": "image",
                    "image": img_item  
                }
                item_messages.append(image_message)
                for count,index in enumerate(self.rag_res(img_item)):
                    image_count+=1
                    info_item=self.retrival_info[index]
                    rag_examples.append(rag_item.format(count+1,image_count,info_item['label']))
                    rag_labels.append(info_item['label'])
                    image_message = {
                    "type": "image",
                    "image": self.rag_image_path+info_item['image']  
                    }
                    item_messages.append(image_message)
            
           
            if 'Picture 1' in data_item['instruction'] :
                text=rag_template.format(example=" ".join(rag_examples),count=image_count+1)  

            
            item_messages.append({
                "type": "text",
                "text": text
            })
        #形成batch

        return rag_labels
        
        

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
    model=infer_rag(**config)
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
                    item['predict']=output
                    del item['instruction']
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