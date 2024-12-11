#from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re
from collections import Counter

from paddleocr import PaddleOCR

def build_ocr():
    return PaddleOCR(lang='ch') 

def extract_ocr_info(data):
    # 1. 提取所有文本并过滤掉非中文文本
    texts = [entry[1][0] for entry in data]
    
    # 使用正则表达式过滤掉非中文文本
    chinese_texts = [text for text in texts if re.search('[\u4e00-\u9fa5]', text)]

    # 2. 统计出现频率最高的词
    word_counter = Counter(chinese_texts)
    most_common_word, most_common_count = word_counter.most_common(1)[0]

    # 3. 提取最上方和最下方的中文文本
    # 计算每个文本框的y坐标的中心位置
    y_positions = [(entry[0][0][1] + entry[0][2][1]) / 2 for entry in data]
    
    # 找到最上方和最下方的中文文本
    # 排除掉最上方的行（非中文行），并找到最上方和最下方的中文文本
    valid_texts = [text for i, text in enumerate(chinese_texts) if re.search('[\u4e00-\u9fa5]', text)]
    valid_y_positions = [y_positions[i] for i in range(len(chinese_texts)) if re.search('[\u4e00-\u9fa5]', chinese_texts[i])]
    
    top_text = valid_texts[valid_y_positions.index(min(valid_y_positions))]
    bottom_text = valid_texts[valid_y_positions.index(max(valid_y_positions))]

    return most_common_word, most_common_count, top_text, bottom_text


"""
ocr = PaddleOCR(lang='ch') # need to run only once to download and load model into memory
img_path = '/home/pubw/datasets/www25/train/images/feb7a339-4af7-415d-96b3-39299e151e73-1788-0.jpg'
result = ocr.ocr(img_path, cls=False)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

def draw_ocr(image: Image, boxes: list, txts: list, scores: list = None, font_path=None):
    
    Draw OCR results on the image.
    
    :param image: The image object (PIL Image) to draw on.
    :param boxes: List of bounding box coordinates for each OCR text.
    :param txts: List of OCR extracted texts.
    :param scores: List of confidence scores for each detected text (optional).
    :param font_path: Path to a ttf font file for drawing text (optional).
    :retrn: The image with OCR boxes and text drawn on it.
    
    # Convert image to a format that we can draw on (PIL Image)
    draw = ImageDraw.Draw(image)

    # Optional: Load a font for drawing text
    if font_path:
        font = ImageFont.truetype(font_path, 20)
    else:
        font = ImageFont.load_default()

    # Iterate through each detected box, text, and score
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        # Draw the bounding box
        box = np.array(box)
        draw.polygon([(box[0][0], box[0][1]),
                      (box[1][0], box[1][1]),
                      (box[2][0], box[2][1]),
                      (box[3][0], box[3][1])], outline="red", width=2)
        
        # Optionally display the score as text next to the box
        if scores is not None:
            score_text = f'{scores[idx]:.2f}'
            draw.text((box[0][0], box[0][1] - 10), score_text, fill="red", font=font)

        # Draw the OCR text
        text_position = (box[0][0] + 5, box[0][1] + 5)  # Adjust text position for readability
        draw.text(text_position, txt, fill="blue", font=font)

    # Return the image with the results drawn on it
    return image

# draw result

result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores)
im_show.show()

"""