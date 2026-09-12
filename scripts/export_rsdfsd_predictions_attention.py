import argparse
import csv
import json
import math
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHINESE_ROOT = PROJECT_ROOT
DEFAULT_RUN_DIR = (
    CHINESE_ROOT
    / "save"
    / "fkh_icd10"
    / "seed888_datafkh_icd10_hflchineserobertawwmextlarge_biaY_coddesc_graph1_gratt512"
)
DEFAULT_CHECKPOINTS = [
    DEFAULT_RUN_DIR / "model" / "checkpoint-1735",
    DEFAULT_RUN_DIR / "model" / "checkpoint-2160",
]
DEFAULT_TEST_FILE = CHINESE_ROOT / "model_data" / "fkh_icd10" / "fkh_icd10_test.pkl"
DEFAULT_LABEL_MAP = CHINESE_ROOT / "model_data" / "fkh_icd10" / "fkh_icd10_label_map.json"
DEFAULT_LABEL_NAMES = CHINESE_ROOT / "model_data" / "fkh_icd10" / "icd_fkh_icd10.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "chinese_icd10"


PUNCT_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)
DATE_RE = re.compile(r"^\d{2,4}[-/.年]\d{1,2}([-/.\u6708]\d{1,2}\u65e5?)?$")
NUMBER_RE = re.compile(r"^[\d.]+(%|mg|ml|mmol|g|u|l|ng|sec|cm|mm|次|分|岁)?$", re.IGNORECASE)
ADMIN_TERMS = {
    "患者", "入院", "出院", "情况", "诊断", "诊疗", "经过", "给予", "完善", "相关", "检查",
    "治疗", "后", "无", "未", "见", "明显", "正常", "同意", "定期", "复查", "本院",
    "提示", "考虑", "建议", "目前", "既往", "查体", "辅助", "主任", "好转", "继续",
}
CLINICAL_HINTS = (
    "炎", "癌", "瘤", "结核", "肺", "胸", "支气管", "咳", "痰", "热", "痛", "血", "感染",
    "积液", "气短", "胸闷", "呼吸", "啰音", "肝", "肾", "心", "脑", "腹", "贫血", "低钠",
    "低钾", "白蛋白", "抗痨", "抗感染", "CT", "核磁", "超声", "病理", "强化", "结节",
)
CLINICAL_SINGLE_CHARS = {
    "肺", "胸", "咳", "痰", "热", "痛", "血", "肝", "肾", "心", "脑", "腹", "炎",
    "癌", "瘤", "结", "核", "钠", "钾", "氯", "糖", "尿", "喘", "肿", "液", "气",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export RSD-FSD predictions and real label-aware attention for Chinese ICD-10 test set."
    )
    parser.add_argument("--test_file", default=str(DEFAULT_TEST_FILE))
    parser.add_argument("--label_map", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument("--label_names", default=str(DEFAULT_LABEL_NAMES))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out_pred", default=str(DEFAULT_OUT_DIR / "rsdfsd_predictions.csv"))
    parser.add_argument("--out_top8", default=str(DEFAULT_OUT_DIR / "rsdfsd_top8_predictions.csv"))
    parser.add_argument("--out_attention", default=str(DEFAULT_OUT_DIR / "rsdfsd_attention.csv"))
    parser.add_argument("--out_attention_tokens", default=str(DEFAULT_OUT_DIR / "rsdfsd_attention_tokens.csv"))
    parser.add_argument(
        "--inspection_report",
        default=str(DEFAULT_OUT_DIR / "rsdfsd_checkpoint_inspection.txt"),
    )
    parser.add_argument("--return_attention", action="store_true")
    parser.add_argument("--top_token_per_label", type=int, default=20)
    parser.add_argument("--top_terms_per_label", type=int, default=20)
    parser.add_argument("--attention_pool", choices=["max", "mean"], default="max")
    parser.add_argument("--max_samples", type=int, default=0, help="Debug only; 0 exports the full test set.")
    return parser.parse_args()


def require_file(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_test_split(path):
    path = require_file(path, "test_file")
    if path.suffix.lower() != ".pkl":
        raise ValueError(
            "This exporter currently supports the real project test split pkl. "
            f"Got: {path}"
        )
    with path.open("rb") as f:
        payload = pickle.load(f)
    required = {"text", "hadm_id", "codes"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"test split is missing required keys: {sorted(missing)}")
    return payload


def load_label_map(path):
    path = require_file(path, "label_map")
    with path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    id_to_code = label_map.get("id_to_code")
    if not isinstance(id_to_code, list) or not id_to_code:
        raise ValueError(f"label_map must contain non-empty id_to_code list: {path}")
    return label_map, id_to_code


def load_label_names(path):
    path = require_file(path, "label_names")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    names = {}
    if isinstance(raw, dict):
        for code, value in raw.items():
            if isinstance(value, list):
                names[normalize_code(code)] = str(value[0]) if value else ""
            else:
                names[normalize_code(code)] = str(value)
    elif isinstance(raw, list):
        for row in raw:
            if isinstance(row, dict) and row.get("code"):
                names[normalize_code(row["code"])] = str(row.get("title", ""))
    return names


def normalize_code(code):
    return str(code).strip().upper()


def normalize_truth_codes(value):
    return {normalize_code(x) for x in str(value).replace(",", ";").split(";") if str(x).strip()}


def patch_chinese_config():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import rsdfsd_config as chinese_cfg
    return chinese_cfg


def build_model_config(cfg, num_labels, checkpoint):
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
    config.export_source_checkpoint = str(checkpoint)
    return config


def read_trainer_state(checkpoint):
    state_path = Path(checkpoint) / "trainer_state.json"
    if not state_path.exists():
        return None
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def inspect_checkpoints(explicit_checkpoint, report_path):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    candidates = [Path(explicit_checkpoint)] if explicit_checkpoint else DEFAULT_CHECKPOINTS
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError("No RSD-FSD checkpoint found among: " + ", ".join(str(p) for p in candidates))

    selected = None
    reason = ""
    best_refs = []
    for ckpt in existing:
        state = read_trainer_state(ckpt)
        if not state:
            lines.append(f"{ckpt}: trainer_state.json missing")
            continue
        best_model_checkpoint = state.get("best_model_checkpoint")
        best_metric = state.get("best_metric")
        lines.append(f"{ckpt}: best_model_checkpoint={best_model_checkpoint}; best_metric={best_metric}")
        if best_model_checkpoint:
            best_refs.append((ckpt, Path(best_model_checkpoint), best_metric))

    if explicit_checkpoint:
        selected = Path(explicit_checkpoint)
        reason = "explicit --checkpoint argument"
    elif best_refs:
        for ckpt, best_ref, metric in best_refs:
            if best_ref.exists() and best_ref in existing:
                selected = best_ref
                reason = f"trainer_state best_model_checkpoint (best_metric={metric})"
                break
            if str(best_ref).replace("/", "\\") == str(ckpt).replace("/", "\\"):
                selected = ckpt
                reason = f"trainer_state points to this checkpoint (best_metric={metric})"
                break

    if selected is None:
        selected = DEFAULT_CHECKPOINTS[-1]
        reason = "fallback choice: no usable best checkpoint information was available"

    lines.append("")
    lines.append(f"selected_checkpoint={selected}")
    lines.append(f"selection_reason={reason}")
    lines.append("attention_chunk_pool=max over chunks for term evidence unless overridden")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected


def write_prediction_rows(path, sample_ids, id_to_code, probs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "code", "prob"])
        for sample_id, row in zip(sample_ids, probs):
            for label_idx, prob in enumerate(row):
                writer.writerow([sample_id, id_to_code[label_idx], f"{float(prob):.8f}"])


def write_top8_rows(path, sample_ids, id_to_code, code_names, probs, truth_sets):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "rank", "code", "name", "prob", "is_ground_truth"])
        for sample_id, row, truth in zip(sample_ids, probs, truth_sets):
            top = np.argsort(-row)[:8]
            for rank, label_idx in enumerate(top, start=1):
                code = id_to_code[int(label_idx)]
                writer.writerow([
                    sample_id,
                    rank,
                    code,
                    code_names.get(normalize_code(code), ""),
                    f"{float(row[label_idx]):.8f}",
                    int(normalize_code(code) in truth),
                ])


def is_noise_token(token):
    token = str(token).strip()
    if not token:
        return True
    if token in {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "##"}:
        return True
    cleaned = token.replace("##", "").strip()
    if not cleaned:
        return True
    if PUNCT_RE.match(cleaned) or DATE_RE.match(cleaned) or NUMBER_RE.match(cleaned):
        return True
    if cleaned.lower() in {"mg", "ml", "mmol", "ng", "sec", "cm", "ct"}:
        return True
    if len(cleaned) == 1 and re.match(r"[\u4e00-\u9fff]", cleaned) and cleaned not in CLINICAL_SINGLE_CHARS:
        return True
    return False


def clean_token(token):
    return str(token).replace("##", "").strip()


def is_clinical_term(term):
    term = str(term).strip()
    if len(term) < 2:
        return False
    if term in ADMIN_TERMS:
        return False
    if PUNCT_RE.match(term) or DATE_RE.match(term) or NUMBER_RE.match(term):
        return False
    if any(hint in term for hint in CLINICAL_HINTS):
        return True
    return len(term) >= 3 and bool(re.search(r"[\u4e00-\u9fff]", term))


def aggregate_terms(token_rows, text, max_terms):
    candidates = defaultdict(list)
    ordered_rows = sorted(token_rows, key=lambda row: (row[0], row[3]))
    filtered = [
        (int(chunk_id), clean_token(tok), float(score), int(token_pos))
        for chunk_id, tok, score, token_pos in ordered_rows
        if not is_noise_token(tok)
    ]
    for _, token, score, _ in filtered:
        if token and token in text and is_clinical_term(token):
            candidates[token].append(score)

    # Merge short adjacent Chinese tokens into 2-6 character terms when the merged term occurs in text.
    for n in range(2, 7):
        for i in range(0, max(0, len(filtered) - n + 1)):
            if len({filtered[i + j][0] for j in range(n)}) != 1:
                continue
            positions = [filtered[i + j][3] for j in range(n)]
            if positions != list(range(positions[0], positions[0] + n)):
                continue
            toks = [filtered[i + j][1] for j in range(n)]
            if any(not re.search(r"[\u4e00-\u9fff]", t) for t in toks):
                continue
            term = "".join(toks)
            if term in text and is_clinical_term(term):
                candidates[term].append(max(filtered[i + j][2] for j in range(n)))

    ranked = []
    for term, scores in candidates.items():
        ranked.append((term, max(scores)))
    ranked.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))
    return ranked[:max_terms]


def infer_one(model, item, device):
    labels = item.pop("labels", None)
    item = {k: v.to(device) for k, v in item.items()}
    if item["input_ids"].dim() == 3:
        item["input_ids"] = item["input_ids"].squeeze(0)
    if "token_type_ids" in item:
        if item["token_type_ids"].dim() == 3:
            item["token_type_ids"] = item["token_type_ids"].squeeze(0)
    if item["attention_mask"].dim() == 3:
        item["attention_mask"] = item["attention_mask"].squeeze(0)
    with torch.no_grad():
        outputs = model(**item)
    if labels is not None:
        item["labels"] = labels
    return outputs, item


def top_token_attention_all_labels(outputs, item, tokenizer, num_labels, max_tokens, pool):
    token_attn = outputs.get("token_attn")
    if token_attn is None:
        raise RuntimeError("Model output did not include token_attn; cannot export real RSD-FSD attention.")
    # Expected shape: [num_chunks, num_labels, num_heads, chunk_len]
    attn = token_attn.detach().float().cpu()
    if attn.ndim != 4:
        raise RuntimeError(f"Unsupported token_attn shape: {tuple(attn.shape)}")
    if pool == "mean":
        pooled = attn.mean(dim=2)
    else:
        pooled = attn.max(dim=2).values
    # pooled: [num_chunks, num_labels, chunk_len]

    input_ids = item["input_ids"].detach().cpu()
    attention_mask = item["attention_mask"].detach().cpu()

    token_lookup = []
    for chunk_id in range(pooled.shape[0]):
        valid_len = min(pooled.shape[2], input_ids.shape[1])
        ids = input_ids[chunk_id, :valid_len].tolist()
        mask = attention_mask[chunk_id, :valid_len].tolist()
        chunk_tokens = []
        for pos, (token_id, is_valid) in enumerate(zip(ids, mask)):
            if int(is_valid) != 1:
                chunk_tokens.append(None)
                continue
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            chunk_tokens.append(None if is_noise_token(token) else clean_token(token))
        token_lookup.append(chunk_tokens)

    rows_by_label = {}
    pooled_np = pooled.numpy()
    for label_idx in range(num_labels):
        rows = []
        for chunk_id in range(pooled_np.shape[0]):
            valid_len = min(pooled_np.shape[2], len(token_lookup[chunk_id]))
            scores = pooled_np[chunk_id, label_idx, :valid_len]
            if max_tokens < valid_len:
                candidate_indices = np.argpartition(-scores, max_tokens - 1)[:max_tokens]
                candidate_indices = candidate_indices[np.argsort(-scores[candidate_indices])]
            else:
                candidate_indices = np.argsort(-scores)
            kept = 0
            for pos in candidate_indices:
                token = token_lookup[chunk_id][int(pos)]
                if token is None:
                    continue
                rows.append((chunk_id, token, float(scores[int(pos)]), int(pos)))
                kept += 1
                if kept >= max_tokens:
                    break
        rows.sort(key=lambda x: -x[2])
        rows_by_label[label_idx] = rows[:max_tokens]
    return rows_by_label


def main():
    args = parse_args()
    cfg = patch_chinese_config()
    from rsdfsd.data.dataset import RSDFSDDataset
    from rsdfsd.models.model import Model

    selected_checkpoint = inspect_checkpoints(args.checkpoint, args.inspection_report)

    test_payload = load_test_split(args.test_file)
    label_map, id_to_code = load_label_map(args.label_map)
    code_names = load_label_names(args.label_names)
    num_labels = len(id_to_code)

    texts = list(test_payload["text"])
    sample_ids = [str(x) for x in test_payload["hadm_id"]]
    truth_sets = [normalize_truth_codes(x) for x in test_payload["codes"]]
    if args.max_samples and args.max_samples > 0:
        texts = texts[: args.max_samples]
        sample_ids = sample_ids[: args.max_samples]
        truth_sets = truth_sets[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.PRETRAIN_MODEL,
        local_files_only=cfg.LOCAL_FILES_ONLY,
        do_lower_case=cfg.DO_LOWER_CASE,
    )

    true_matrix = np.zeros((len(texts), num_labels), dtype=np.float32)
    code_to_id = {normalize_code(k): int(v) for k, v in label_map["code_to_id"].items()}
    for row_idx, truth in enumerate(truth_sets):
        for code in truth:
            if code in code_to_id:
                true_matrix[row_idx, code_to_id[code]] = 1.0

    config = build_model_config(cfg, num_labels, selected_checkpoint)
    model = Model.from_pretrained(str(selected_checkpoint), config=config)
    model.to(cfg.DEVICE)
    model.eval()

    dataset = RSDFSDDataset(texts, true_matrix, tokenizer=tokenizer)
    probs = np.zeros((len(dataset), num_labels), dtype=np.float32)

    pred_path = Path(args.out_pred)
    top8_path = Path(args.out_top8)
    attn_token_path = Path(args.out_attention_tokens)
    attn_term_path = Path(args.out_attention)
    for p in [pred_path, top8_path, attn_token_path, attn_term_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    with attn_token_path.open("w", encoding="utf-8-sig", newline="") as tok_f, attn_term_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as term_f:
        tok_writer = csv.writer(tok_f)
        term_writer = csv.writer(term_f)
        tok_writer.writerow(["sample_id", "code", "chunk_id", "token", "attention_score"])
        term_writer.writerow(["sample_id", "code", "term", "attention_score"])

        for sample_idx in range(len(dataset)):
            outputs, model_inputs = infer_one(model, dataset[sample_idx], cfg.DEVICE)
            row_probs = torch.sigmoid(outputs["logits"]).detach().float().cpu().numpy().reshape(-1)
            probs[sample_idx] = row_probs

            label_order = list(range(num_labels))
            # Export all labels, prioritizing top probabilities first for easier partial inspection.
            label_order.sort(key=lambda i: float(row_probs[i]), reverse=True)
            text = str(texts[sample_idx])
            sample_id = sample_ids[sample_idx]
            token_rows_by_label = top_token_attention_all_labels(
                outputs,
                model_inputs,
                tokenizer,
                num_labels,
                args.top_token_per_label,
                args.attention_pool,
            )
            for label_idx in label_order:
                code = id_to_code[label_idx]
                token_rows = token_rows_by_label[label_idx]
                for chunk_id, token, score, _token_pos in token_rows:
                    tok_writer.writerow([sample_id, code, chunk_id, token, f"{score:.8f}"])
                term_rows = aggregate_terms(
                    token_rows,
                    text,
                    args.top_terms_per_label,
                )
                for term, score in term_rows:
                    term_writer.writerow([sample_id, code, term, f"{score:.8f}"])

            if (sample_idx + 1) % 10 == 0 or sample_idx + 1 == len(dataset):
                print(f"[export] {sample_idx + 1}/{len(dataset)} samples")

    write_prediction_rows(pred_path, sample_ids, id_to_code, probs)
    write_top8_rows(top8_path, sample_ids, id_to_code, code_names, probs, truth_sets)

    meta_path = DEFAULT_OUT_DIR / "rsdfsd_export_manifest.json"
    meta = {
        "checkpoint": str(selected_checkpoint),
        "test_file": str(Path(args.test_file)),
        "label_map": str(Path(args.label_map)),
        "num_samples": len(dataset),
        "num_labels": num_labels,
        "prediction_file": str(pred_path),
        "top8_file": str(top8_path),
        "attention_tokens_file": str(attn_token_path),
        "attention_terms_file": str(attn_term_path),
        "attention_pool": args.attention_pool,
        "top_token_per_label": args.top_token_per_label,
        "top_terms_per_label": args.top_terms_per_label,
        "note": "Attention scores are exported from RSD-FSD token_attn; term scores are max-pooled from retained token attention.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {pred_path}")
    print(f"[done] wrote {top8_path}")
    print(f"[done] wrote {attn_token_path}")
    print(f"[done] wrote {attn_term_path}")
    print(f"[done] wrote {meta_path}")


if __name__ == "__main__":
    main()
