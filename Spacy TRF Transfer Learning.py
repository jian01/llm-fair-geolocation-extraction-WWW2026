#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from src.dataset import Dataset
import json
import random
import os
import shutil
from tqdm import tqdm
from src.dataset import Dataset as NERDataset
from src.dataset import Tag, Document
from src.llm_json_tagging import coerce_tags
import spacy
spacy.require_gpu()
from spacy.training import Example
from spacy.util import minibatch, compounding
import numpy as np
import pandas as pd
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0))

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
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

def compute_metrics_for_fold(nlp, gt):
    false_positives = 0
    true_positives = 0
    incomplete = 0
    model_positives = 0
    false_negative = 0
    gt_tag_count = 0
    test_dataset = {}
    texts = []
    for k in gt.keys():
        texts.append(gt[k].text)
    for k, doc in zip(gt.keys(), tqdm(nlp.pipe(texts, batch_size=8), total=len(texts))):
        tags = set()
        for ent in doc.ents:
            if ent.label_ == 'LOC':
                tags.add(Tag(start=ent.start_char, end=ent.end_char, text=ent.text))
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

def convert_to_spacy_examples(dataset, nlp):
    """Convert dataset to Spacy Example format for training"""
    examples = []
    for k in dataset.keys():
        doc = dataset[k]
        # Create predicted doc (tokenized only)
        pred_doc = nlp.make_doc(doc.text)
        # Create reference doc with entities
        ref_doc = nlp.make_doc(doc.text)
        entities = []
        for tag in sorted(doc.tags, key=lambda x: x.start):
            span = ref_doc.char_span(
                tag.start,
                tag.end,
                label="LOC",
                alignment_mode="expand"
            )
            if span:
                entities.append(span)
        # Set entities on reference doc
        ref_doc.set_ents(entities)
        # Create example (predicted, reference)
        example = Example(pred_doc, ref_doc)
        examples.append(example)
    return examples

def evaluate_LOC_only(nlp, examples):
    # filter predictions only
    for ex in examples:
        ex.predicted.ents = [ent for ent in ex.predicted.ents if ent.label_ == "LOC"]
    return nlp.evaluate(examples)

def train_spacy_model(base_model, train_examples, output_dir, n_iter=50, patience=5, min_delta=0.0, early_stopping=True):
    """Train a Spacy NER model"""

    # Load base model
    nlp = spacy.load(base_model, exclude=["parser", "tagger", "attribute_ruler", "lemmatizer", "textcat"])
    random.shuffle(train_examples)
    if early_stopping:
        split = int(len(train_examples) * 0.9)
        dev_examples = train_examples[split:]
        train_examples = train_examples[:split]

    best_score = -1.0
    best_bytes = None
    epochs_no_improve = 0

    trainable = {"ner", "transformer", "curated_transformer"}

    # Disable other pipes during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in trainable]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        optimizer.learn_rate = 5e-5
        
        # Training loop
        for epoch in range(n_iter):
            random.shuffle(train_examples)
            losses = {}
            batches = minibatch(train_examples, size=8)
            for batch in tqdm(batches, desc=f"Epoch {epoch}", total=len(train_examples)//8):
                nlp.update(batch, sgd=optimizer, losses=losses, drop=0.1)

            if early_stopping:
                scores = evaluate_LOC_only(nlp, dev_examples)
                f1 = scores["ents_f"]
            else:
                f1 = -1

            print(
                f"Epoch {epoch:02d} | "
                f"loss(ner)={losses.get('ner', 0):.4f} | "
                f"dev ents_f={f1:.2f}"
            )

            # ---- early stopping logic ----
            if f1 > best_score + min_delta:
                best_score = f1
                best_bytes = nlp.to_bytes()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best dev ents_f={best_score:.2f})"
                    )
                    break
    if best_bytes is not None:
        nlp.from_bytes(best_bytes)
    nlp.to_disk(output_dir)
    return nlp


# In[ ]:


with open(f"data/tagged/gt_filtered_v2.json", "r") as file:
    gt = NERDataset.from_dict(json.load(file))


# In[ ]:


len(gt)


# In[ ]:


false_positives = 0
true_positives = 0
incomplete = 0
model_positives = 0
false_negative = 0
gt_tag_count = 0
all_tagged_documents = {}


# In[ ]:


base_model = "en_core_web_trf"
FOLDS = 5

lead_to_fold = get_fold_by_lead_id(set(int(k) for k in gt.keys()), FOLDS)

for i in range(FOLDS):
    test_keys = set(str(k) for k,v in lead_to_fold.items() if v==i)
    test_dataset = NERDataset({k: gt[k] for k in test_keys})
    
    # Create training dataset
    train_dataset = NERDataset({k: gt[k] for k in gt.keys() if k not in test_keys})

    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Load base model for training
    nlp_base = spacy.load(base_model, exclude=["parser", "tagger", "attribute_ruler", "lemmatizer", "textcat"])
    
    # Convert to Spacy examples
    train_examples = convert_to_spacy_examples(train_dataset, nlp_base)

    # Train the model
    output_dir = f"spacy_trf_ner_{i}"
    print(f"Training fold {i}...")
    nlp_trained = train_spacy_model(base_model, train_examples, output_dir, n_iter=5, early_stopping=False)
    
    # Load the trained model for evaluation (to ensure it's fresh)
    nlp_eval = spacy.load(output_dir, exclude=["parser", "tagger", "attribute_ruler", "lemmatizer", "textcat"])
    
    # Evaluate on test set
    fps, tps, inc, fn, mp, count, tagged_dataset = compute_metrics_for_fold(nlp_eval, test_dataset)
    false_positives += fps
    true_positives += tps
    incomplete += inc
    model_positives += mp
    false_negative += fn
    gt_tag_count += count
    
    # Accumulate tagged documents from this fold
    for k in tagged_dataset.keys():
        all_tagged_documents[k] = tagged_dataset[k]
    
    print(f"Fold {i}: TP={tps} FP={fps} FN={fn} Incomplete={inc} Model={mp} GT={count}")
    
    # Clean up the models to free memory
    del nlp_trained
    del nlp_eval
    import gc
    gc.collect()

# In[ ]:


print(f"TP={true_positives} FP={false_positives} FN={false_negative} Incomplete={incomplete} Model={model_positives} GT={gt_tag_count}")

# Save the dataset with all tags
final_dataset = NERDataset(all_tagged_documents)
output_path = "data/spacy_trf_tuned_v2.json"
with open(output_path, "w") as f:
    json.dump(final_dataset.to_dict(), f)
print(f"Saved tagged dataset to {output_path}")

# Delete checkpoint folders
for i in range(10):
    checkpoint_dir = f"spacy_trf_ner_{i}"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"Deleted checkpoint folder: {checkpoint_dir}")
print("Cleanup complete.")

