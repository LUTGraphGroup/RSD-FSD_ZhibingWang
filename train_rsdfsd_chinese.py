import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rsdfsd_config as cfg

from rsdfsd.models.model import Model
from rsdfsd.data.dataset import RSDFSDDataset
from rsdfsd.training.trainer import RSDFSDTrainer, compute_metrics, search_best_threshold, sigmoid_np
from rsdfsd.training.metrics import all_metrics, print_metrics
from rsdfsd.utils.common import build_and_save_head_tail_info
from rsdfsd.graph.srd_graph import build_srd_graph_artifacts


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_text(text):
    return " ".join(str(text or "").strip().split())


def _load_label_texts(input_json):
    with open(input_json, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if isinstance(labels, dict):
        iterable = labels.values()
    elif isinstance(labels, list):
        iterable = labels
    else:
        raise TypeError(f"Unsupported labels type: {type(labels)}")

    desc_code = []
    def_code = []
    codes = []
    for row in iterable:
        if not isinstance(row, dict):
            continue
        title = normalize_text(row.get("title", ""))
        desc_text = normalize_text(row.get("anchor_text", ""))
        if not desc_text:
            desc_text = normalize_text(
                f"{row.get('clinical_description', '')} {row.get('comorbidity_profile', '')}"
            )
        codes.append(normalize_text(row.get("code", "")))
        desc_code.append(desc_text)
        def_code.append(title)

    return codes, desc_code, def_code


def _tokenize_texts(tokenizer, texts):
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    token_dict = dict(tokens)
    if "token_type_ids" not in token_dict:
        token_dict["token_type_ids"] = torch.zeros_like(token_dict["input_ids"])
    return token_dict


def _ensure_label_tokens_ready(tokenizer):
    desc_path = Path(cfg.DATA_PATH) / f"icd_{cfg.DATA_TYPE}_desc.pkl"
    def_path = Path(cfg.DATA_PATH) / f"icd_{cfg.DATA_TYPE}_def.pkl"
    meta_path = Path(cfg.DATA_PATH) / f"icd_{cfg.DATA_TYPE}_token_meta.json"
    expected_meta = {
        "pretrain_model": cfg.PRETRAIN_MODEL,
        "code_type": cfg.CODE_TYPE,
        "data_type": cfg.DATA_TYPE,
    }

    if desc_path.exists() and def_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            current_meta = json.load(f)
        if all(current_meta.get(k) == v for k, v in expected_meta.items()):
            return
        print("[label-token] model/tokenizer changed, rebuilding label token files...")
    elif desc_path.exists() and def_path.exists():
        print("[label-token] token metadata missing, rebuilding label token files...")

    if desc_path.exists():
        desc_path.unlink()
    if def_path.exists():
        def_path.unlink()

    if desc_path.exists() and def_path.exists():
        return

    input_json = Path(cfg.PROCESSED_ICD_V3_FILE)
    if not input_json.exists():
        raise FileNotFoundError(f"label description json not found: {input_json}")

    print(f"[label-token] building {desc_path.name} and {def_path.name} with {cfg.PRETRAIN_MODEL}")
    codes, desc_code, def_code = _load_label_texts(input_json)
    if not codes:
        raise ValueError(f"no label entries loaded from {input_json}")

    with desc_path.open("wb") as f:
        pickle.dump(_tokenize_texts(tokenizer, desc_code), f)
    with def_path.open("wb") as f:
        pickle.dump(_tokenize_texts(tokenizer, def_code), f)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({**expected_meta, "num_labels": len(codes)}, f, ensure_ascii=False, indent=2)
    print(f"[label-token] labels: {len(codes)}")


def _load_or_build_structure_partition(train_1hot):
    if not os.path.exists(cfg.HEAD_TAIL_MASK_FILE):
        print("head_tail_mask.pkl not found, building...")
        ht_info = build_and_save_head_tail_info(
            train_1hot=train_1hot,
            save_path=cfg.HEAD_TAIL_MASK_FILE,
            coverage_ratio=cfg.STRUCTURE_COVERAGE_RATIO,
        )
    else:
        with open(cfg.HEAD_TAIL_MASK_FILE, "rb") as f:
            ht_info = pickle.load(f)
    return ht_info


def _ensure_srd_graph_ready(train_1hot, ht_info):
    required_files = [
        cfg.ADJ_MATRIX_FILE,
        cfg.EDGE_FEATURE_INDICES_FILE,
        cfg.HEAD_DEGREE_FILE,
        cfg.GROUND_IND_TAIL_FILE,
        cfg.GROUND_IND_HEAD_FILE,
    ]
    if all(os.path.exists(p) for p in required_files):
        return

    if ht_info is None:
        raise RuntimeError("SRD graph build requires head-tail payload, but ht_info is None.")

    print("SRD graph files not found, building SRD graph artifacts...")
    build_srd_graph_artifacts(train_1hot=train_1hot, head_tail_payload=ht_info)


def _save_threshold(threshold, val_micro_f1, val_macro_f1):
    payload = {
        "threshold": float(threshold),
        "val_f1_micro": float(val_micro_f1),
        "val_f1_macro": float(val_macro_f1),
        "source": "validation_set",
    }
    with open(cfg.THRESHOLD_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[threshold] saved {cfg.THRESHOLD_FILE}: {payload}")


def _manual_predict(model, dataset, num_labels):
    model.to(cfg.DEVICE)
    model.eval()
    probabilities = np.zeros((len(dataset), num_labels), dtype=np.float32)
    logits_all = np.zeros((len(dataset), num_labels), dtype=np.float32)
    labels_all = np.zeros((len(dataset), num_labels), dtype=np.int32)

    with tqdm(total=len(dataset)) as pbar:
        for idx, item in enumerate(dataset):
            labels = item.pop("labels").numpy()
            item = {k: v.to(cfg.DEVICE) for k, v in item.items()}

            with torch.no_grad():
                outputs = model(**item)
                logits = outputs["logits"].detach().cpu().numpy().squeeze(0)

            logits_all[idx] = logits
            probabilities[idx] = sigmoid_np(logits)
            labels_all[idx] = labels
            pbar.update(1)

    return logits_all, probabilities, labels_all


def _calibrate_threshold(trainer, val_dataset):
    if not cfg.USE_THRESHOLD_CALIBRATION:
        return float(cfg.DEFAULT_THRESHOLD)

    print("[threshold] calibrating on validation set...")
    val_logits, _, val_labels = _manual_predict(
        trainer.model,
        val_dataset,
        num_labels=trainer.model.config.num_labels,
    )
    threshold, micro_f1, macro_f1 = search_best_threshold(val_labels, val_logits)
    _save_threshold(threshold, micro_f1, macro_f1)
    return threshold


def _load_or_calibrate_threshold(trainer, val_dataset):
    if cfg.USE_THRESHOLD_CALIBRATION and os.path.exists(cfg.THRESHOLD_FILE):
        with open(cfg.THRESHOLD_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        threshold = float(payload["threshold"])
        print(f"[threshold] loaded from {cfg.THRESHOLD_FILE}: {threshold}")
        return threshold

    if not cfg.USE_THRESHOLD_CALIBRATION:
        return float(cfg.DEFAULT_THRESHOLD)

    return _calibrate_threshold(trainer, val_dataset)


def _evaluate_manual(model, dataset, y_gt, threshold, split_name, num_labels):
    logits, probabilities, labels_from_dataset = _manual_predict(model, dataset, num_labels)
    y_true = y_gt if y_gt is not None else labels_from_dataset
    y_pred = (probabilities >= threshold).astype(np.int32)

    metrics = all_metrics(y_pred, y_true, k=[5, 8, 15], yhat_raw=probabilities)
    metrics["threshold"] = float(threshold)

    np.save(os.path.join(cfg.SAVE_PREDICTION_PATH, f"y_{split_name}_prob.npy"), probabilities)
    np.save(os.path.join(cfg.SAVE_PREDICTION_PATH, f"y_{split_name}_pred.npy"), y_pred)
    np.save(os.path.join(cfg.SAVE_PREDICTION_PATH, f"y_{split_name}_true.npy"), y_true)
    np.save(os.path.join(cfg.SAVE_PREDICTION_PATH, f"y_{split_name}_logits.npy"), logits)

    with open(os.path.join(cfg.SAVE_PREDICTION_PATH, f"{split_name}-metrics.txt"), "w", encoding="utf-8") as f:
        f.write(str(metrics))
    with open(os.path.join(cfg.SAVE_METRICS_PATH, f"{split_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[{split_name.upper()} 集合评估结果]:")
    print_metrics(metrics)
    return metrics


def _evaluate_val_and_test_manual(model, val_dataset, test_dataset, val_1hot, test_1hot, threshold, num_labels):
    print("======= LETS TEST ======")
    for split_name, dataset, y_gt in zip(
        ["val", "test"],
        [val_dataset, test_dataset],
        [val_1hot, test_1hot],
    ):
        print(f"Predict {split_name} set ...")
        _evaluate_manual(model, dataset, y_gt, threshold, split_name, num_labels)


def _load_split(split):
    data_file = os.path.join(cfg.DATA_PATH, f"{cfg.DATA_TYPE}_{split}.pkl")
    label_file = os.path.join(cfg.DATA_PATH, f"{cfg.DATA_TYPE}_{split}_1hot.npz")

    with open(data_file, "rb") as file:
        data = pickle.load(file)
    texts = data["text"].tolist()
    labels = np.load(label_file)["arr_0"]
    return texts, labels


def _build_training_arguments():
    kwargs = dict(
        output_dir=cfg.OUTPUT_DIR,
        group_by_length=cfg.GROUP_BY_LENGTH,
        learning_rate=cfg.LEARNING_RATE,
        lr_scheduler_type=cfg.LR_SCHEDULER_TYPE,
        logging_strategy=cfg.LOGGING_STRATEGY,
        num_train_epochs=cfg.EPOCHS,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        save_strategy=cfg.SAVE_STRATEGY,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        save_total_limit=cfg.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=cfg.LOAD_BEST_MODEL_AT_END,
        greater_is_better=cfg.GREATER_IS_BETTER,
        metric_for_best_model=cfg.METRIC_FOR_BEST_MODEL,
        optim=cfg.OPTIM,
        bf16=cfg.USE_BF16,
        fp16=cfg.USE_FP16,
        report_to="none",
    )
    try:
        return TrainingArguments(evaluation_strategy=cfg.EVALUATION_STRATEGY, **kwargs)
    except TypeError:
        return TrainingArguments(eval_strategy=cfg.EVALUATION_STRATEGY, **kwargs)


def main():
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.SEED)
        torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ensure_dir(cfg.SAVE_METRICS_PATH)
    ensure_dir(cfg.SAVE_PREDICTION_PATH)
    ensure_dir(cfg.OUTPUT_DIR)

    print(f"Load {cfg.DATA_TYPE} dataset")
    train_texts, train_1hot = _load_split("train")
    val_texts, val_1hot = _load_split("val")
    test_texts, test_1hot = _load_split("test")

    if cfg.MAX_TRAIN_SAMPLES and cfg.MAX_TRAIN_SAMPLES > 0:
        original_train_size = len(train_texts)
        rng = np.random.default_rng(cfg.SEED)
        sample_size = min(int(cfg.MAX_TRAIN_SAMPLES), original_train_size)
        sampled_indices = np.sort(rng.choice(original_train_size, size=sample_size, replace=False))
        train_texts = [train_texts[i] for i in sampled_indices]
        train_1hot = train_1hot[sampled_indices]
        print(f"[data] train subset enabled: {sample_size}/{original_train_size} samples")

    num_labels = len(train_1hot[0])
    print(f"[data] train={len(train_texts)} val={len(val_texts)} test={len(test_texts)} labels={num_labels}")

    ht_info = _load_or_build_structure_partition(train_1hot)
    _ensure_srd_graph_ready(train_1hot, ht_info)

    print("Load Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.PRETRAIN_MODEL,
        local_files_only=cfg.LOCAL_FILES_ONLY,
        do_lower_case=cfg.DO_LOWER_CASE,
    )
    _ensure_label_tokens_ready(tokenizer)

    print("Load Model...")
    config = AutoConfig.from_pretrained(
        cfg.PRETRAIN_MODEL,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        local_files_only=cfg.LOCAL_FILES_ONLY,
    )
    config.model_name = cfg.PRETRAIN_MODEL
    config.local_files_only = cfg.LOCAL_FILES_ONLY
    config.attention_hidden_size = cfg.HIDDEN_SIZE
    config.transform_size = cfg.TRANSFORM_SIZE
    config.n_heads = cfg.NUM_ATT_HEAD
    config.code_token_file = os.path.join(cfg.DATA_PATH, f"icd_{cfg.DATA_TYPE}_{cfg.CODE_TYPE}.pkl")

    config.adj_matrix_file = cfg.ADJ_MATRIX_FILE
    config.edge_feature_indices_file = cfg.EDGE_FEATURE_INDICES_FILE
    config.head_degree_file = cfg.HEAD_DEGREE_FILE
    config.ground_ind_tail_file = cfg.GROUND_IND_TAIL_FILE
    config.ground_ind_head_file = cfg.GROUND_IND_HEAD_FILE
    config.graph_att_hidd = cfg.GRAPH_ATT
    config.graph_ffn_hidd = cfg.GRAPH_FFN
    config.srd_num_bins = cfg.SRD_NUM_BINS
    config.srd_num_features = len(cfg.SRD_FEATURE_NAMES)
    config.srd_hub_penalty_init = cfg.SRD_HUB_PENALTY_INIT
    config.head_num = int(np.sum(ht_info["head_tail_mask"]))
    config.tail_num = int(num_labels - config.head_num)

    model = Model(config=config)
    if cfg.START_MODEL_FROM_CHECKPOINT:
        model = model.from_pretrained(cfg.START_MODEL_FROM_CHECKPOINT, config=config)

    train_dataset = RSDFSDDataset(train_texts, train_1hot, tokenizer=tokenizer)
    val_dataset = RSDFSDDataset(val_texts, val_1hot, tokenizer=tokenizer)
    test_dataset = RSDFSDDataset(test_texts, test_1hot, tokenizer=tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    trainer = RSDFSDTrainer(
        model=model,
        args=_build_training_arguments(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.EARLY_STOPPING_PATIENCE)],
    )

    json.dumps(model.config.to_dict())
    print("config is serializable!")

    if cfg.MODE == "train":
        print("======= LETS TRAIN ======")
        print("... Starting to Evaluate model (val set) ...")
        trainer.evaluate(eval_dataset=val_dataset)
        print("... Training model ...")
        trainer.train()
        threshold = _calibrate_threshold(trainer, val_dataset)
        _evaluate_val_and_test_manual(trainer.model, val_dataset, test_dataset, val_1hot, test_1hot, threshold, num_labels)
    elif cfg.MODE == "test":
        print("... Loading best model for test ...")
        trainer.model = trainer.model.from_pretrained(cfg.OUTPUT_DIR)
        threshold = _load_or_calibrate_threshold(trainer, val_dataset)
        _evaluate_val_and_test_manual(trainer.model, val_dataset, test_dataset, val_1hot, test_1hot, threshold, num_labels)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
