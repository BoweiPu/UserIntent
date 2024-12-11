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
rag_template="""你是一个电商领域识图专家,可以理解消费者上传的软件截图或实物拍摄图。现在,请你对消费者上传的图片进行分类。你只需要回答图片分类结果,不需要其他多余的话。以下是可以参考的分类标签,分类标签:[{labels}]\n
可以参考的例子是[{example}]\n。真实输入是Picture {count}: <image>\n"""

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

class infer_rag(InferModel_base):
    def __init__(self,
                 path_2b='/home/pubw/proj/Qwen2-VL-2B-Instruct',
                 path_7b='/home/pubw/proj/Qwen2-VL-7B-Instruct',
                 pt_path="/home/pubw/proj/UserIntent/feat/remain_key.pt",
                 image_Augmented="key",
                 clip_path="/home/pubw/proj/clip-vit-large-patch14",
                 key_words_path='/home/pubw/proj/UserIntent/utils/task2_text_frequency_top20.json',
                 topk=3,
                 rag_image_path="",
                 **kwargs):
        super().__init__(path_2b,path_7b)
        
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
        self.use_ocr= (image_Augmented !="no" and image_Augmented!="layout")
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
        ##only for task2
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
                rag_res=self.rag_res(img_item)
                label_set=[self.retrival_info[index]['label'] for index in rag_res]
                label_set=list(set(label_set))
                for count,index in enumerate(rag_res[:3]):
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
                text=rag_template.format(example=" ".join(rag_examples),
                                         count=image_count+1,
                                         labels="\""+"\",\"".join(label_set)+"\"")  

            
            item_messages.append({
                "type": "text",
                "text": text
            })
        #形成batch
            messages_7b.append([{
                "role": "user",
                "content": item_messages
            }])
        print(text,batch[0]['output'])
        output=self.infer_llm(messages_7b,'7b')
        #print(image_caption_outputs,output)
        return output
        
        
