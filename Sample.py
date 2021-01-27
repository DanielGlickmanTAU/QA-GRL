import os

import torch
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, BertForQuestionAnswering
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import os
import re
import sys

import requests
import string
import numpy as np
# from colorama import Fore
from tokenizers import BertWordPieceTokenizer
from tqdm import tqdm
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

gpu = torch.device('cuda')
# ============================================= DOWNLOADING DATA =======================================================
max_seq_length = 384
batch_size = 16
epochs = 4

download = False
if download:
    train_data = requests.get("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json")
    if train_data.status_code in (200,):
        with open('train.json', 'wb') as train_file:
            train_file.write(train_data.content)
    eval_data = requests.get("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json")
    if eval_data.status_code in (200,):
        with open('eval.json', 'wb') as eval_file:
            eval_file.write(eval_data.content)
with open('train.json') as f:
    raw_train_data = json.load(f)
with open('eval.json') as f:
    raw_eval_data = json.load(f)
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
if not os.path.exists("bert_base_uncased/"):
    os.makedirs("bert_base_uncased/")
slow_tokenizer.save_pretrained("bert_base_uncased/")
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)


max_seq_length = 384
class Sample:
    def __init__(self, question, context, start_char_idx=None, answer_text=None, all_answers=None):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False
        self.start_token_idx = -1
        self.end_token_idx = -1

    def preprocess(self):
        context = self.join_with_spaces(self.context)
        question = self.join_with_spaces(self.question)
        tokenized_context = tokenizer.encode(context)
        tokenized_question = tokenizer.encode(question)
        if self.answer_text is not None:
            answer = self.join_with_spaces(self.answer_text)
            end_char_idx = self.start_char_idx + len(answer)
            if end_char_idx >= len(context):
                self.skip = True
                return
            is_char_in_ans = [0] * len(context)
            for idx in range(self.start_char_idx, end_char_idx):
                is_char_in_ans[idx] = 1
            ans_token_idx = []
            for idx, (start, end) in enumerate(tokenized_context.offsets):
                if sum(is_char_in_ans[start:end]) > 0:
                    ans_token_idx.append(idx)
            if len(ans_token_idx) == 0:
                self.skip = True
                return
            self.start_token_idx = ans_token_idx[0]
            self.end_token_idx = ans_token_idx[-1]
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
        attention_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:
            self.skip = True
            return
        self.input_word_ids = input_ids
        self.input_type_ids = token_type_ids
        self.input_mask = attention_mask
        self.context_token_to_char = tokenized_context.offsets

    def join_with_spaces(self,to_join):
        context = " ".join(str(to_join).split())
        return context