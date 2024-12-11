import json
import os
import shutil
from PIL import Image

from paddleocr import PaddleOCR
from tqdm import tqdm

import torch
import logging
import jieba
import cv2
import numpy as np

def mask_and_fill_with_external_values(image_path, polygons, output_path="output.jpg"):
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
    
    # 4. 保存结果
    cv2.imwrite(output_path, result)
    return result
ocr_model= PaddleOCR(lang='ch') 
# 设置基本配置，level=logging.WARNING表示只记录warning及更高级别的日志
logging.getLogger('ppocr').setLevel(logging.ERROR)
train_json="/home/pubw/datasets/www25/test1/test1_task2.json"
image_path="/home/pubw/datasets/www25/test1/images/"
save_path="/home/pubw/proj/UserIntent/masked/remain_key_test/"
key_words=json.load(open('/home/pubw/proj/UserIntent/utils/task2_text_frequency_top20.json'))
key_words = [item for sublist in list(key_words.values()) for item in sublist]
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
        if os.path.exists(save_path+img.split("/")[-1]):
            continue
        mask_list=[]
        ocr_res=ocr_model.ocr(img, cls=False)[0]
        if ocr_res is None:
            shutil.copy(img,save_path+img.split("/")[-1])
            continue
        for ocr_tiem in ocr_res:
            #print(ocr_tiem)
            #print(f"bbox:{ocr_tiem[0]},res:{ocr_tiem[1]}")
            #print(ocr_tiem[1])
            #print(key_words[item['output']])
            #print(jieba.lcut(ocr_tiem[1][0]))
            if any(keyword in jieba.lcut(ocr_tiem[1][0]) for keyword in key_words):
                continue
            #print(ocr_tiem[1])
            mask_list.append(ocr_tiem[0])
        mask_and_fill_with_external_values(img,mask_list,save_path+img.split("/")[-1])


# python utils/mask/ocr_mask_test.py