import pickle
import numpy as np
import torch
import torch.nn.functional as F


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none")
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none")

    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.0)
        q_loss.masked_fill_(pad_mask, 0.0)

    return (p_loss.mean() + q_loss.mean()) / 2.0


def compute_class_freq_from_1hot(train_1hot):
    return np.sum(train_1hot, axis=0).astype(np.int64).tolist()


def head_tail_mask_bycoverage(class_freq, num_labels, coverage_ratio=0.7):
    class_freq = np.array(class_freq)
    sorted_labels = np.argsort(class_freq)[::-1]
    sorted_freqs = class_freq[sorted_labels]

    total_samples = np.sum(sorted_freqs) + 1e-12
    cumulative_coverage = np.cumsum(sorted_freqs) / total_samples
    split_idx = np.searchsorted(cumulative_coverage, coverage_ratio) + 1

    head_labels = set(sorted_labels[:split_idx].tolist())
    tail_labels = set(sorted_labels[split_idx:].tolist())

    head_tail_mask = np.zeros(num_labels, dtype=np.int64)
    for idx in head_labels:
        head_tail_mask[idx] = 1

    return head_tail_mask.tolist(), list(tail_labels)


def build_and_save_head_tail_info(train_1hot, save_path, coverage_ratio=0.7):
    num_labels = train_1hot.shape[1]
    class_freq = compute_class_freq_from_1hot(train_1hot)
    head_tail_mask, tail_labels = head_tail_mask_bycoverage(
        class_freq=class_freq,
        num_labels=num_labels,
        coverage_ratio=coverage_ratio,
    )

    payload = {
        "class_freq": class_freq,
        "head_tail_mask": head_tail_mask,
        "tail_labels": tail_labels,
        "coverage_ratio": coverage_ratio,
    }

    with open(save_path, "wb") as f:
        pickle.dump(payload, f)

    return payload


def compute_cosine_similarity_loss(att, lab_desc_vec, reduction="mean"):
    if lab_desc_vec.dim() == 2:
        lab_desc_vec = lab_desc_vec.unsqueeze(0).expand(att.size(0), -1, -1)

    lab_desc_vec = lab_desc_vec.to(att.device)
    cosine_sim = F.cosine_similarity(att, lab_desc_vec, dim=-1)
    loss_per_label = 1.0 - cosine_sim

    if reduction == "mean":
        return loss_per_label.mean()
    if reduction == "none":
        return loss_per_label
    raise ValueError(f"Unsupported reduction: {reduction}")


def get_three_stage_gate(epoch, total_epochs):
    if epoch <= 2:
        return 0.0
    if epoch <= 9:
        return min(max(float(epoch - 2) / 6.0, 0.0), 1.0)
    return 1.0
def get_ssdc_warmup_weight(epoch, warmup_epochs=2, target_weight=1.0):
    if epoch < warmup_epochs:
        return target_weight * float(epoch + 1) / float(warmup_epochs)
    return target_weight
