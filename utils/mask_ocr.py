

from builder.ocr_builder import build_ocr





import json
from PIL import Image

from tqdm import tqdm
import torch

train_json="/home/pubw/datasets/www25/train/train_task2.json"
image_path="/home/pubw/datasets/www25/train/images/"
json_file=json.load(open(train_json))

ocr_model=build_ocr()

image_embed_list=[]
image_info_list=[]
for item in tqdm(json_file):
    image_list=item['image']
    input_image_list=[]
    for image_item in image_list:
        input_image_list.append(Image.open(image_path+image_item).convert('RGB'))
    print(ocr_model.ocr(image_list, cls=False))
    

