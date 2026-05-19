# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, AutoTokenizer, set_seed
import argparse
from scipy.special import softmax
import torch
import re
import random
from tqdm import tqdm
import os

RANDOM_SEED = 42
BATCH_SIZE = 8
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def preprocess(text):
  EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
  USER_MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_-]+")
  PHONE_PATTERN = re.compile(r"(\+?\d{1,3})?[\s\*\.-]?\(?\d{1,4}\)?[\s\*\.-]?\d{2,4}[\s\*\.-]?\d{2,6}")
  text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
  text = re.sub(USER_MENTION_PATTERN, "[USER]", text)
  text = re.sub(PHONE_PATTERN, " [PHONE]", text).replace('  [PHONE]', ' [PHONE]')
  return text.lower().strip()


def test(test_df, model_path, id2label, label2id):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Load Model", flush=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, trust_remote_code=True, num_labels=len(label2id), id2label=id2label, label2id=label2id, torch_dtype=torch.bfloat16, quantization_config=bnb_config, device_map="auto"
    )
    model.eval()
    print(model.device, flush=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    try:
        model.config.pad_token_id = tokenizer.get_vocab()[tokenizer.pad_token]
    except Exception:
        print("Warning: Exception occurred while setting pad_token_id")

    texts = test_df['text'].tolist()
    all_logits = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), "predict"):
            batch = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch, truncation=True, max_length=512, padding=True, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            all_logits.append(logits.float().cpu())

    prob_pred = softmax(torch.cat(all_logits, dim=0).numpy(), axis=-1)
    return prob_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", "-i", required=True, help="Path to the test jsonl dataset.", type=str)
    parser.add_argument("--output-directory", "-o", required=True, help="Path to the output directory.", type=str)
    args = parser.parse_args()

    test_path = os.path.abspath(args.input_directory)
    prediction_path = os.path.abspath(args.output_directory)

    id2label = {0: "human", 1: "machine"}
    label2id = {"human": 0, "machine": 1}
    set_seed(RANDOM_SEED)

    test_df = pd.read_json(test_path, lines=True)
    test_df['text'] = [preprocess(x) for x in test_df['text']]

    probs = test(test_df, "Mmatias89/Mdok-Qwen3-14B", id2label, label2id)

    if 'id' not in test_df.columns: test_df['id'] = test_df.index
    predictions_df = pd.DataFrame({'id': test_df['id'], 'label': probs[:,1]})
    predictions_df.to_json(f'{prediction_path}/{test_path.split("/")[-1]}', lines=True, orient='records')


if __name__ == '__main__':
    main()
