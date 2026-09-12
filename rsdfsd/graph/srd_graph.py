import os
import pickle
import numpy as np

from rsdfsd_config import (
    DATA_TYPE,
    DATA_PATH,
    HEAD_TAIL_MASK_FILE,
    STRUCTURE_COVERAGE_RATIO,
    ADJ_MATRIX_FILE,
    EDGE_FEATURE_INDICES_FILE,
    HEAD_DEGREE_FILE,
    GROUND_IND_TAIL_FILE,
    GROUND_IND_HEAD_FILE,
    SRD_FEATURE_NAMES,
    SRD_NUM_BINS,
    SRD_TOPK_PER_TAIL,
    SRD_SCORE_ALPHA,
    SRD_SCORE_BETA,
    SRD_SCORE_GAMMA,
    SRD_SCORE_DELTA,
)
from rsdfsd.utils.common import build_and_save_head_tail_info


EPS = 1e-8


def _safe_minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if abs(x_max - x_min) < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min + 1e-8)


def _bucketize_feature(x: np.ndarray, num_bins: int) -> np.ndarray:
    x_norm = _safe_minmax(x)
    buckets = np.floor(x_norm * (num_bins - 1)).astype(np.int64)
    buckets = np.clip(buckets, 0, num_bins - 1)
    return buckets.astype(np.int16)


def _compute_relation_statistics(train_1hot: np.ndarray, head_labels, tail_labels):
    train = train_1hot.astype(np.float32)
    num_docs = float(train.shape[0])

    tail_mat = train[:, tail_labels]  # [N, Nt]
    head_mat = train[:, head_labels]  # [N, Nh]

    cooc = tail_mat.T @ head_mat  # [Nt, Nh]
    tail_freq = np.sum(tail_mat, axis=0)  # [Nt]
    head_freq = np.sum(head_mat, axis=0)  # [Nh]

    p_r_given_c = cooc / np.maximum(head_freq[None, :], EPS)

    p_rc = cooc / max(num_docs, EPS)
    p_r = tail_freq[:, None] / max(num_docs, EPS)
    p_c = head_freq[None, :] / max(num_docs, EPS)

    pmi = np.log((p_rc + EPS) / (p_r * p_c + EPS))
    pmi = np.clip(pmi, a_min=-10.0, a_max=10.0)

    support = p_rc

    raw_head_degree_proxy = (cooc > 0).astype(np.float32).sum(axis=0)  # [Nh]

    return {
        "cooc": cooc.astype(np.float32),
        "tail_freq": tail_freq.astype(np.float32),
        "head_freq": head_freq.astype(np.float32),
        "p_r_given_c": p_r_given_c.astype(np.float32),
        "pmi": pmi.astype(np.float32),
        "support": support.astype(np.float32),
        "head_degree_proxy": raw_head_degree_proxy.astype(np.float32),
        "num_docs": int(num_docs),
    }


def build_srd_graph_artifacts(
    train_1hot: np.ndarray,
    head_tail_payload: dict,
    adj_matrix_path: str = ADJ_MATRIX_FILE,
    edge_feature_indices_path: str = EDGE_FEATURE_INDICES_FILE,
    head_degree_path: str = HEAD_DEGREE_FILE,
    ground_ind_tail_path: str = GROUND_IND_TAIL_FILE,
    ground_ind_head_path: str = GROUND_IND_HEAD_FILE,
):
    head_tail_mask = np.asarray(head_tail_payload["head_tail_mask"])
    tail_labels = list(head_tail_payload["tail_labels"])
    head_labels = np.where(head_tail_mask == 1)[0].astype(np.int64).tolist()

    stats = _compute_relation_statistics(train_1hot, head_labels=head_labels, tail_labels=tail_labels)

    p_r_given_c = stats["p_r_given_c"]
    pmi = stats["pmi"]
    support = stats["support"]
    cooc = stats["cooc"]
    head_degree_proxy = stats["head_degree_proxy"]

    score = (
        SRD_SCORE_ALPHA * _safe_minmax(p_r_given_c)
        + SRD_SCORE_BETA * _safe_minmax(pmi)
        + SRD_SCORE_GAMMA * _safe_minmax(support)
        - SRD_SCORE_DELTA * _safe_minmax(head_degree_proxy)[None, :]
    )


    score = np.where(cooc > 0, score, -1e9)

    num_tail, num_head = score.shape
    adj_matrix = np.zeros((num_tail, num_head), dtype=np.uint8)

    for r in range(num_tail):
        valid_heads = np.where(cooc[r] > 0)[0]
        if len(valid_heads) == 0:
            continue
        keep_k = min(SRD_TOPK_PER_TAIL, len(valid_heads))
        sorted_idx = np.argsort(score[r, valid_heads])[::-1][:keep_k]
        selected_heads = valid_heads[sorted_idx]
        adj_matrix[r, selected_heads] = 1

    feature_stack = np.stack([
        _bucketize_feature(p_r_given_c, SRD_NUM_BINS),
        _bucketize_feature(pmi, SRD_NUM_BINS),
        _bucketize_feature(support, SRD_NUM_BINS),
    ], axis=-1).astype(np.int16)  # [Nt, Nh, F]


    feature_stack[adj_matrix == 0] = 0

    head_degree = adj_matrix.sum(axis=0).astype(np.float32)

    ground_ind_tail = {i: int(label_idx) for i, label_idx in enumerate(tail_labels)}
    ground_ind_head = {i: int(label_idx) for i, label_idx in enumerate(head_labels)}

    os.makedirs(os.path.dirname(adj_matrix_path), exist_ok=True)
    with open(adj_matrix_path, "wb") as f:
        pickle.dump(adj_matrix, f)
    with open(edge_feature_indices_path, "wb") as f:
        pickle.dump(feature_stack, f)
    with open(head_degree_path, "wb") as f:
        pickle.dump(head_degree, f)
    with open(ground_ind_tail_path, "wb") as f:
        pickle.dump(ground_ind_tail, f)
    with open(ground_ind_head_path, "wb") as f:
        pickle.dump(ground_ind_head, f)

    return {
        "head_labels": head_labels,
        "tail_labels": tail_labels,
        "adj_matrix": adj_matrix,
        "edge_feature_indices": feature_stack,
        "head_degree": head_degree,
        "feature_names": list(SRD_FEATURE_NAMES),
        "ground_ind_tail": ground_ind_tail,
        "ground_ind_head": ground_ind_head,
        "meta": {
            "num_tail": int(num_tail),
            "num_head": int(num_head),
            "topk_per_tail": int(SRD_TOPK_PER_TAIL),
            "num_bins": int(SRD_NUM_BINS),
            "num_docs": int(stats["num_docs"]),
        },
    }


def main():
    train_1hot_file = os.path.join(DATA_PATH, f"{DATA_TYPE}_train_1hot.npz")
    train_1hot = np.load(train_1hot_file)["arr_0"]

    payload = build_and_save_head_tail_info(
        train_1hot=train_1hot,
        save_path=HEAD_TAIL_MASK_FILE,
        coverage_ratio=STRUCTURE_COVERAGE_RATIO
    )

    graph_payload = build_srd_graph_artifacts(train_1hot=train_1hot, head_tail_payload=payload)

    print(f"saved: {HEAD_TAIL_MASK_FILE}")
    print(f"saved: {ADJ_MATRIX_FILE}")
    print(f"saved: {EDGE_FEATURE_INDICES_FILE}")
    print(f"saved: {HEAD_DEGREE_FILE}")
    print(f"saved: {GROUND_IND_TAIL_FILE}")
    print(f"saved: {GROUND_IND_HEAD_FILE}")
    print(f"num_labels: {train_1hot.shape[1]}")
    print(f"head num: {len(graph_payload['head_labels'])}")
    print(f"tail num: {len(graph_payload['tail_labels'])}")
    print(f"adj shape: {graph_payload['adj_matrix'].shape}")
    print(f"edge feature shape: {graph_payload['edge_feature_indices'].shape}")


if __name__ == "__main__":
    main()
