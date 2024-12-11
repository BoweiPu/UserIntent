#from model.infer_ocr import infer_ocr
from model.inferModel_base import InferModel_base
from model.infer_cot import infer_cot
from model.infer_rag import infer_rag

def build(config):
    
    #if config['model']=='ocr':
    #    return infer_ocr(config['path_2b '],config['path_7b'])
    if config['model']=='cot':
        return infer_cot(**config)
    if config['model']=='rag':
        return infer_rag(**config)
    else:
        return InferModel_base(**config)
    