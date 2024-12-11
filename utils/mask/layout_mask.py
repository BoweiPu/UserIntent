import json
import os
from PIL import Image

from paddleocr import PaddleOCR
from tqdm import tqdm

import torch
import logging
import jieba
import cv2
import numpy as np


def global_blur(image_path, blur_kernel=(25, 25), output_path="blurred_output.jpg"):
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
    
    # 3. 保存结果
    cv2.imwrite(output_path, blurred_image)
    return blurred_image

#ocr_model= PaddleOCR(lang='ch') 
# 设置基本配置，level=logging.WARNING表示只记录warning及更高级别的日志
logging.getLogger('ppocr').setLevel(logging.ERROR)
train_json="/home/pubw/datasets/www25/train/train_task2.json"
image_path="/home/pubw/datasets/www25/train/images/"
save_path="/home/pubw/proj/UserIntent/masked/layout/"
#key_words=json.load(open('/home/pubw/proj/UserIntent/utils/task2_text_frequency_top20.json'))
os.makedirs(save_path, exist_ok=True)
json_file=json.load(open(train_json))


image_embed_list=[]
image_info_list=[]
for item in tqdm(json_file):
    image_list=[image_path+item_path for item_path in item['image']]
    input_image_list=[]
    #for image_item in image_list:
        #input_image_list.append(Image.open(image_path+image_item).convert('RGB'))
    
    for img in image_list:
        global_blur(img,(89, 89),output_path=save_path+img.split("/")[-1])
    

# python -m /home/pubw/proj/UserIntent/utils/ocr_mask_test.py