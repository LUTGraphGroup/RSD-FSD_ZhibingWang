import json
import os
import pickle

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from rsdfsd_config import (
    DEVICE,
    PRETRAIN_MODEL,
    PROCESSED_ICD_V3_FILE,
    STATIC_ANCHOR_FILE,
    STATIC_ANCHOR_MODE,
)


BATCH_SIZE = 16
MAX_LEN = 256


def normalize_text(text):
    return " ".join(str(text or "").strip().split())


def build_anchor_text(row, mode="full"):
    code = normalize_text(row.get("code", ""))
    title = normalize_text(row.get("title", ""))
    anchor_text = normalize_text(row.get("anchor_text", ""))
    clinical_description = normalize_text(row.get("clinical_description", ""))
    comorbidity_profile = normalize_text(row.get("comorbidity_profile", ""))

    if mode == "title":
        text = f"ICD code {code}. Label title: {title}."
    elif mode in {"desc", "full"}:
        text = anchor_text or normalize_text(
            f"ICD code {code}. Label title: {title}. {clinical_description} {comorbidity_profile}"
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return normalize_text(text)


def load_processed_icd_json(json_path, mode):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        rows = list(data.values())
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Expected processed ICD json to be list/dict, got {type(data)}")

    codes = []
    texts = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        codes.append(normalize_text(row.get("code", "")))
        texts.append(build_anchor_text(row, mode=mode))

    return codes, texts


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counted = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counted


def encode_texts(texts, tokenizer, encoder):
    all_vecs = []
    encoder.eval()

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)

            outputs = encoder(**batch, return_dict=True)
            pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=-1)
            all_vecs.append(pooled.cpu())

    return torch.cat(all_vecs, dim=0)


def main():
    if not os.path.exists(PROCESSED_ICD_V3_FILE):
        raise FileNotFoundError(f"Not found: {PROCESSED_ICD_V3_FILE}")

    os.makedirs(os.path.dirname(STATIC_ANCHOR_FILE), exist_ok=True)

    codes, texts = load_processed_icd_json(PROCESSED_ICD_V3_FILE, STATIC_ANCHOR_MODE)
    print(f"Loaded {len(codes)} ICD codes from processed v3 anchor text")
    if codes:
        print("First code:", codes[0])
        print("First text:", texts[0])

    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAIN_MODEL,
        local_files_only=True,
        do_lower_case=True,
    )
    encoder = AutoModel.from_pretrained(
        PRETRAIN_MODEL,
        local_files_only=True,
    ).to(DEVICE)

    anchors = encode_texts(texts, tokenizer, encoder)

    payload = {
        "codes": codes,
        "texts": texts,
        "mode": STATIC_ANCHOR_MODE,
        "source": os.path.basename(PROCESSED_ICD_V3_FILE),
        "source_type": "processed_v3_anchor_text",
        "encoder": os.path.basename(PRETRAIN_MODEL),
        "anchors": anchors,
    }

    with open(STATIC_ANCHOR_FILE, "wb") as f:
        pickle.dump(payload, f)

    print(f"saved: {STATIC_ANCHOR_FILE}")
    print(f"anchor shape: {tuple(anchors.shape)}")


if __name__ == "__main__":
    main()
