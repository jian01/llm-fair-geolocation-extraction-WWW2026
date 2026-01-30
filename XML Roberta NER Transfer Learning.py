#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from src.dataset import Dataset
import json
import random
import os
import gc
import shutil
from tqdm import tqdm
from src.dataset import Dataset as NERDataset
from src.dataset import Tag, Document
from src.llm_json_tagging import coerce_tags
from transformers import pipeline, set_seed, EarlyStoppingCallback
import numpy as np
import torch
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
set_seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_fold_by_lead_id(lead_ids, n_groups=10):
    train = pd.read_json("data/humset/train.jsonl", lines=True)
    validation = pd.read_json("data/humset/validation.jsonl", lines=True)
    test = pd.read_json("data/humset/test.jsonl", lines=True)
    humset = pd.concat([train, validation, test], ignore_index=True)
    humset = humset[humset["lead_id"].isin(lead_ids)].drop_duplicates("lead_id", keep="first")

    vc = humset["project_id"].value_counts()

    rng = np.random.default_rng(42)
    noise = rng.uniform(0, 0.5, len(vc))
    order = (-vc.values + noise).argsort()

    projects = vc.index.to_numpy()[order]

    bucket_sizes = np.zeros(n_groups, dtype=int)
    proj_to_group = {}

    for pid in projects:
        g = bucket_sizes.argmin()
        proj_to_group[pid] = g
        bucket_sizes[g] += vc[pid]

    lead_to_proj = humset.set_index("lead_id")["project_id"].to_dict()
    lead_to_fold = {k: proj_to_group[v] for k,v in lead_to_proj.items()}
    for k in lead_ids:
        if not k in lead_to_fold:
            lead_to_fold[k] = 0
    return lead_to_fold

def weak_contains(tag, doc):
    sorted_doc = sorted(doc.tags, key=lambda x: x.start)
    for t in sorted_doc:
        if t.end < tag.start:
            continue
        if tag.start < t.end and t.start < tag.end:
            return t
    return None

def take_metrics(gt_doc, model_doc):
    false_positives = 0
    true_positives = 0
    incomplete = 0
    false_negative = 0
    partial_positive_tags = set()
    for t in gt_doc.tags:
        if not weak_contains(t, model_doc):
            false_negative += 1
    for t in model_doc.tags:
        if t in gt_doc.tags:
            true_positives += 1
            continue
        weak_match_tag = weak_contains(t, gt_doc)
        if weak_match_tag and weak_match_tag not in partial_positive_tags:
            partial_positive_tags.add(weak_match_tag)
            incomplete+=1
        elif weak_match_tag:
            continue
        else:
            false_positives+=1
    return false_positives, true_positives, incomplete, false_negative

def compute_metrics_for_fold(model, tok, gt):
    false_positives = 0
    true_positives = 0
    incomplete = 0
    model_positives = 0
    false_negative = 0
    gt_tag_count = 0
    nlp = pipeline("ner", model=model, tokenizer=tok)
    test_dataset = {}
    texts = []
    for k in gt.keys():
        texts.append(gt[k].text)
    for k, doc in zip(gt.keys(), tqdm(nlp(texts, batch_size=8, grouped_entities=True), total=len(texts))):
        tags = set()
        for ent in doc:
            if ent['entity_group'] == 'LOC':
                tags.add(Tag(start=ent['start'], end=ent['end'], text=gt[k].text[ent['start']:ent['end']]))
        test_dataset[k] = Document(text=gt[k].text, tags=coerce_tags(gt[k].text, tags))
    test_dataset = NERDataset(test_dataset)

    for k in gt.keys():
        real_doc = gt[k]
        model_doc = test_dataset[k]
        fps, tps, inc, fn = take_metrics(real_doc, model_doc)
        false_positives+=fps
        true_positives+=tps
        incomplete+=inc
        false_negative+=fn
        model_positives += len(model_doc.tags)
        gt_tag_count += len(real_doc.tags)
    return false_positives, true_positives, incomplete, false_negative, model_positives, gt_tag_count, test_dataset


# In[ ]:


with open(f"data/tagged/gt_filtered_v2.json", "r") as file:
    gt = NERDataset.from_dict(json.load(file))


# In[ ]:


len(gt)


# In[ ]:


import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, TrainingArguments, Trainer
)
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

WORD_RE = re.compile(r"\S+")

def word_tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int,int]]]:
    tokens, spans = [], []
    for m in WORD_RE.finditer(text):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    return tokens, spans

def char_spans_to_bio_word_tags(
    text: str,
    entities: List[Dict],
    word_spans: List[Tuple[int,int]],
) -> List[str]:
    # sort entities by start
    ents = sorted([(int(e["start"]), int(e["end"]), str(e["label"])) for e in entities], key=lambda x: (x[0], x[1]))
    W = len(word_spans)
    word_tags = ["O"] * W

    for (es, ee, lab) in ents:
        covered = []
        for i, (ws, we) in enumerate(word_spans):
            if max(ws, es) < min(we, ee):  # positive overlap
                covered.append(i)
        if not covered:
            continue
        word_tags[covered[0]] = f"B-{lab}"
        for i in covered[1:]:
            word_tags[i] = f"I-{lab}"
    return word_tags

def build_word_level_ds(samples: List[Dict]):
    word_examples = []
    label_vocab = set(["O"])
    for ex in samples:
        text = ex["text"]
        ents = ex.get("entities", [])
        tokens, spans = word_tokenize_with_offsets(text)
        tags = char_spans_to_bio_word_tags(text, ents, spans)
        word_examples.append({"id": ex.get("id", None), "tokens": tokens, "ner_tags_str": tags})
        label_vocab.update(tags)

    def sort_key(lbl):
        if lbl == "O": return (0, "")
        # enforce B before I
        prefix_order = {"B": 1, "I": 2}
        p = lbl.split("-")[0]
        t = "-".join(lbl.split("-")[1:]) if "-" in lbl else ""
        return (1, t, prefix_order.get(p, 9))

    label_list = sorted(list(label_vocab), key=sort_key)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    for ex in word_examples:
        ex["ner_tags"] = [label2id[t] for t in ex["ner_tags_str"]]

    ds = Dataset.from_list(word_examples)
    return ds, label_list, label2id, id2label

def make_tokenize_and_align(tok, label_pad_id=-100):
    def _fn(batch):
        enc = tok(batch["tokens"], is_split_into_words=True, truncation=True)
        all_labels = []
        for i in range(len(batch["tokens"])):
            word_ids = enc.word_ids(i)  # maps subword -> word index (or None)
            word_labels = batch["ner_tags"][i]
            labels = []
            prev_wid = None
            for wid in word_ids:
                if wid is None:
                    labels.append(label_pad_id)           # special tokens
                elif wid != prev_wid:
                    labels.append(word_labels[wid])       # first subword of a word
                else:
                    labels.append(label_pad_id)           # continuation subwords
                prev_wid = wid
            all_labels.append(labels)
        enc["labels"] = all_labels
        return enc
    return _fn

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    y_true, y_pred = [], []
    for p_row, l_row in zip(preds, labels):
        t_seq, p_seq = [], []
        for p_i, l_i in zip(p_row, l_row):
            if l_i == -100:           # ignore specials/padding
                continue
            t_seq.append(id2label[l_i])
            p_seq.append(id2label[p_i])
        y_true.append(t_seq)
        y_pred.append(p_seq)

    return {
        "precision": precision_score(y_true, y_pred),  # entity-level
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
    }

def train_eval_split(ds_all: Dataset, seed=42):
    random.seed(seed)
    ids = list(range(len(ds_all)))
    random.shuffle(ids)
    train_ids = ids[:len(ids)//10]
    eval_ids = ids[len(ids)//10:]

    return DatasetDict(
        train=ds_all.select(train_ids),
        validation=ds_all.select(eval_ids)
    )


# In[ ]:


model_ckpt = "Davlan/xlm-roberta-base-wikiann-ner" 
tok = AutoTokenizer.from_pretrained(model_ckpt, use_fast=True)


# In[ ]:


false_positives = 0
true_positives = 0
incomplete = 0
model_positives = 0
false_negative = 0
gt_tag_count = 0
all_tagged_documents = {}


# In[ ]:

FOLDS = 5
lead_to_fold = get_fold_by_lead_id(set(int(k) for k in gt.keys()), FOLDS)

for i in range(FOLDS):
    test_keys = set(str(k) for k, v in lead_to_fold.items() if v == i)
    test_dataset = NERDataset({k: gt[k] for k in test_keys})

    dataset = []
    for k in gt.keys():
        if k in test_keys:
            continue
        doc = gt[k]
        tags = sorted(doc.tags, key=lambda x: x.start)
        doc_hf = {"id": k, "text": doc.text, "entities": [{"start":t.start, "end":t.end, "label":"LOC"} for t in tags]}
        dataset.append(doc_hf)

    ds, label_list, label2id, id2label = build_word_level_ds(dataset)
    ds_mapped = ds.map(make_tokenize_and_align(tok), batched=True, remove_columns=["tokens","ner_tags_str","ner_tags","id"])
    train_eval = train_eval_split(ds_mapped)

    model = AutoModelForTokenClassification.from_pretrained(
        model_ckpt,
        num_labels=len(label_list),
        id2label={i: l for i, l in enumerate(label_list)},
        label2id={l: i for i, l in enumerate(label_list)},
        ignore_mismatched_sizes=True,  # re-init classification layer to your label count
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tok)

    args = TrainingArguments(
        output_dir=f"xlmr-ner_{i}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        num_train_epochs=30,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        label_smoothing_factor=0.1,
        logging_strategy="epoch",
        report_to="none",
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_eval["train"],
        eval_dataset=train_eval["validation"],
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=7,      # number of *evaluation* steps with no improvement
            early_stopping_threshold=0.0    # min improvement to count as “better”
        )],
    )

    trainer.train()
    fps, tps, inc, fn, mp, count, tagged_dataset = compute_metrics_for_fold(model, tok, test_dataset)
    false_positives += fps
    true_positives += tps
    incomplete += inc
    model_positives += mp
    false_negative += fn
    gt_tag_count += count
    # Accumulate tagged documents from this fold
    for k in tagged_dataset.keys():
        all_tagged_documents[k] = tagged_dataset[k]
    print(
        f"Fold {i}: TP={tps} FP={fps} FN={fn} Incomplete={inc} Model={mp} GT={count}")
    del trainer
    del model
    torch.cuda.empty_cache()
    gc.collect()

# In[ ]:


print(f"TP={true_positives} FP={false_positives} FN={false_negative} Incomplete={incomplete} Model={model_positives} GT={gt_tag_count}")

# Save the dataset with all tags
final_dataset = NERDataset(all_tagged_documents)
output_path = "data/xml_roberta_tuned_v2.json"
with open(output_path, "w") as f:
    json.dump(final_dataset.to_dict(), f)
print(f"Saved tagged dataset to {output_path}")

# Delete checkpoint folders
for i in range(10):
    checkpoint_dir = f"xlmr-ner_{i}"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"Deleted checkpoint folder: {checkpoint_dir}")
print("Cleanup complete.")

