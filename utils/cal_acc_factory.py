
import re
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json


def calculate_metrics(y_true, y_pred):
    """
    计算多标签分类任务的指标: 加权F1, 总体acc, precision, recall。

    参数:
    y_true (List[List[int]]): 真实标签
    y_pred (List[List[int]]): 预测标签

    返回:
    dict: 各指标的分数
    """

    # 计算指标
    metrics = {
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted")),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
    }

    return metrics

def clean_string(s):
    # 去除前后指定符号
    return re.sub(r'^[\"\'\[\]\{\}\s]+|[\"\'\[\]\{\}\s]+$', '', s)

def get_label_pred(jsonl_file,label_json):
    """获取测试集标签以及预测结果

    Args:
        test_file (_type_): 带ground_truth标签的测试集文件
        pred_file (_type_): 对应的预测结果文件
    """
    labels_task1=[]
    preds_task1=[]
    labels_task2=[]
    preds_task2=[]
    label_json=json.load(open(label_json))
    rebuild_label_json={}
    for item in label_json:
        instruction=item['instruction'] if  'instruction' in item else item['prompt']
        label=item['output']
        if  'Picture 1' in instruction :
            labels_task2.append(label)
        else:
            labels_task1.append(label)
    for line in open(jsonl_file, "r"):
        prediction_data = json.loads(line)
        instruction=prediction_data['instruction'] if  'instruction' in prediction_data else prediction_data['prompt']
        pred=clean_string(prediction_data['predict'])
        if  'Picture 1' in instruction :
            preds_task2.append(pred)
        else:
            preds_task1.append(pred)



    return labels_task1, preds_task1,labels_task2,preds_task2


def cal_acc(jsonl_file,label_json):
    labels_task1, preds_task1,labels_task2,preds_task2 = get_label_pred(jsonl_file,label_json)
    metrics_task1 = calculate_metrics(y_true=labels_task1, y_pred=preds_task1)
    metrics_task2= calculate_metrics(y_true=labels_task2, y_pred=preds_task2)
    metrics_all= calculate_metrics(y_true=(labels_task2+labels_task1), y_pred=(preds_task2+preds_task1))
    print(f'metrics_task1:{metrics_task1}\nmetrics_task2:{metrics_task2}\nmetrics_all:{metrics_all}\n')


if __name__ == "__main__":
    jsonl_file = "/home/linlh/LLaMA-Factory/saves/qwen2_vl-7b/full/sft-infer/generated_predictions.jsonl"
    label_json = "/home/pubw/datasets/www25/train/train.json"
    cal_acc(jsonl_file,label_json)

