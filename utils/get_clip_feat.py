import json
from PIL import Image

from tqdm import tqdm
import torch

from transformers import CLIPProcessor, CLIPVisionModelWithProjection
train_json="/home/pubw/datasets/www25/train/train_task2.json"
image_path="/home/pubw/proj/UserIntent/masked/layout/"
pt_path="/home/pubw/proj/UserIntent/feat/layout.pt"
json_file=json.load(open(train_json))

model = CLIPVisionModelWithProjection.from_pretrained("/home/pubw/proj/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("/home/pubw/proj/clip-vit-large-patch14")



image_embed_list=[]
image_info_list=[]
for item in tqdm(json_file):
    image_list=item['image']
    input_image_list=[]
    for image_item in image_list:
        input_image_list.append(Image.open(image_path+image_item).convert('RGB'))

    inputs =processor(images=input_image_list, return_tensors="pt")
    outputs = model(**inputs)
    image_id=item['id']
    del item['instruction']
    image_embed_list.append(outputs.image_embeds)
    image_info_list.append(item)

torch.save({"feat":image_embed_list,"info":image_info_list},pt_path)

#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python utils/get_clip_feat.py