import json
from PIL import Image

from tqdm import tqdm
import torch

from transformers import CLIPProcessor, CLIPVisionModelWithProjection
train_json="/home/pubw/datasets/www25/train/train_task2.json"
image_path="/home/pubw/proj/UserIntent/masked/no_word/"
pt_path="/home/pubw/proj/UserIntent/feat/no_word.pt"
json_file=json.load(open(train_json))

model = CLIPVisionModelWithProjection.from_pretrained("/home/pubw/proj/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("/home/pubw/proj/clip-vit-large-patch14")



image_embed_list=[]
image_info_list=[]
for item in tqdm(json_file):
    image_list=item['image']
    input_image_list=[]
    image_id=item['id']

    for image_item in image_list:
        save_info={}
        save_info['image']=image_item
        save_info['label']=item['output']
        inputs =processor(images=[Image.open(image_path+image_item).convert('RGB')], return_tensors="pt")
        outputs = model(**inputs)
        image_embed_list.append(outputs.image_embeds)
        image_info_list.append(save_info)

torch.save({"feat":torch.cat(image_embed_list).cpu(),"info":image_info_list},pt_path)

#CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 python utils/get_clip_feat.py