from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import os

import pandas as pd
from transformers import AutoTokenizer

'''KorQuAD v1.0에 대한 공식 평가 스크립트 '''
'''본 스크립트는 SQuAD v1.1 평가 스크립트 https://rajpurkar.github.io/SQuAD-explorer/ 를 바탕으로 작성됨.'''

def normalize_answer(s):    
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text) 
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)   
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)      
        return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
   
    #F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)
        
    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)   
        
    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def evaluate_result(dataset_file, prediction_file):
    
    with open(dataset_file, encoding="utf-8") as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
    with open(prediction_file, encoding="utf-8") as f:
        predictions = json.load(f)
    return evaluate(dataset, predictions)


def analyze(dataset, predictions, f1_threshold, prediction_file, model_name_or_path):
    error_conts = []
    error_qs = []
    error_gts =[]
    error_conts_lens = []
    error_qs_lens = []
    error_gts_lens =[]    
    error_preds = []
    error_f1s = []

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        do_lower_case=False,
        use_fast=True,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )

    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                
                exact_match = metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 = metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

                if exact_match == 0 and f1 < f1_threshold:
                  error_conts.append(paragraph['context'])
                  error_qs.append(qa['question'])
                  error_gts.append(ground_truths[0])
                  error_conts_lens.append(len(tokenizer.tokenize(paragraph['context'])))
                  error_qs_lens.append(len(tokenizer.tokenize(qa['question'])))
                  error_gts_lens.append(len(tokenizer.tokenize(ground_truths[0])))
                  error_preds.append(prediction)
                  error_f1s.append(f1)

    df = pd.DataFrame({"context": error_conts,
                      "question": error_qs,
                      "answer": error_gts,            
                      "prediction": error_preds,
                      "f1 score": error_f1s,     
                      "context length": error_conts_lens,
                      "question length": error_qs_lens,
                      "answer length": error_gts_lens})
    df_path = os.path.join("/".join(prediction_file.split('/')[:-1]), 'error_analysis.csv')
    df.to_csv(df_path, encoding='utf-8-sig')

    return df 


def analyze_result(dataset_file, prediction_file, f1_threshold, model_name_or_path):
    
    with open(dataset_file, encoding="utf-8") as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
    with open(prediction_file, encoding="utf-8") as f:
        predictions = json.load(f)
    return analyze(dataset, predictions, f1_threshold, prediction_file, model_name_or_path)