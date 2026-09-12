import os
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch import nn
from typing import Union, Dict, List, Optional, Tuple, Any
from sklearn.metrics import f1_score, recall_score, precision_score
from transformers import Trainer

from rsdfsd_config import *
from rsdfsd.utils.common import (
    compute_cosine_similarity_loss,
    get_three_stage_gate,
    get_ssdc_warmup_weight,
)
class RSDFSDTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hard_negative_bank = None
        self.hard_negative_codes = None
        if USE_FG_HNCL:
            self._load_hard_negative_bank()

    def _load_hard_negative_bank(self):
        if not os.path.exists(HARD_NEG_BANK_FILE):
            raise FileNotFoundError(f"[FG-HNCL] hard negative bank file not found: {HARD_NEG_BANK_FILE}")

        with open(HARD_NEG_BANK_FILE, "rb") as f:
            bank_payload = pickle.load(f)

        if "topk_neg_indices" not in bank_payload:
            raise KeyError("[FG-HNCL] bank payload must contain key 'topk_neg_indices'.")

        topk_neg_indices = bank_payload["topk_neg_indices"]
        if not isinstance(topk_neg_indices, torch.Tensor):
            topk_neg_indices = torch.tensor(topk_neg_indices, dtype=torch.long)

        if topk_neg_indices.dim() != 2:
            raise ValueError(f"[FG-HNCL] topk_neg_indices must be 2D, but got shape={tuple(topk_neg_indices.shape)}")

        self.hard_negative_bank = topk_neg_indices.long().cpu()
        self.hard_negative_codes = bank_payload.get("codes", None)
        print(f"[FG-HNCL] loaded hard negative bank: shape={tuple(self.hard_negative_bank.shape)}")

    def _compute_per_label_align_loss(self, dynamic_label_repr, static_anchors):
        cosine_sim = F.cosine_similarity(dynamic_label_repr, static_anchors, dim=-1)
        per_label_align_loss = 1.0 - cosine_sim
        return per_label_align_loss

    def _compute_usfc_weights(self, logits):
        prob = torch.sigmoid(logits)
        if USFC_WEIGHT_TYPE == "distance":
            weights = 2.0 * torch.abs(prob - 0.5)
        else:
            prob = prob.clamp(min=1e-8, max=1.0 - 1e-8)
            entropy = -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob))
            weights = 1.0 - entropy / np.log(2.0)
        weights = torch.clamp(weights, min=USFC_MIN_WEIGHT, max=1.0)
        return weights

    def _compute_usfc_align_loss(self, logits, labels, dynamic_label_repr, static_anchors):
        per_label_align_loss = self._compute_per_label_align_loss(dynamic_label_repr, static_anchors)
        pos_mask = (labels > 0.5).float()
        weights = self._compute_usfc_weights(logits)
        weighted_loss = per_label_align_loss * weights * pos_mask
        denom = pos_mask.sum().clamp(min=1.0)
        usfc_align_loss = weighted_loss.sum() / denom
        return usfc_align_loss, per_label_align_loss.detach(), weights.detach()

    def _compute_hncl_loss(self, labels, dynamic_label_repr, static_anchors):
        if self.hard_negative_bank is None:
            return torch.tensor(0.0, device=dynamic_label_repr.device)

        bank = self.hard_negative_bank.to(dynamic_label_repr.device)
        pos_pairs = (labels > 0.5).nonzero(as_tuple=False)
        if pos_pairs.numel() == 0:
            return torch.tensor(0.0, device=dynamic_label_repr.device)

        loss_terms = []
        for pair in pos_pairs:
            batch_idx = int(pair[0].item())
            label_idx = int(pair[1].item())

            z_pos = dynamic_label_repr[batch_idx, label_idx]
            a_pos = static_anchors[batch_idx, label_idx]

            neg_candidates = bank[label_idx][:HNCL_TOPK]

            if neg_candidates.numel() == 0:
                continue

            valid_neg_indices = []
            for cand in neg_candidates:
                cand = int(cand.item())
                if cand == label_idx:
                    continue
                if labels[batch_idx, cand] <= 0.5:
                    valid_neg_indices.append(cand)

            if len(valid_neg_indices) == 0:
                continue

            valid_neg_indices = torch.tensor(valid_neg_indices, device=dynamic_label_repr.device, dtype=torch.long)

            if HNCL_STRATEGY == "top1":
                neg_idx = int(valid_neg_indices[0].item())
            else:
                neg_anchor_candidates = static_anchors[batch_idx, valid_neg_indices]
                neg_sims = F.cosine_similarity(
                    z_pos.unsqueeze(0).expand_as(neg_anchor_candidates),
                    neg_anchor_candidates,
                    dim=-1
                )
                hardest_pos = torch.argmax(neg_sims)
                neg_idx = int(valid_neg_indices[hardest_pos].item())

            a_neg = static_anchors[batch_idx, neg_idx]
            pos_sim = F.cosine_similarity(z_pos.unsqueeze(0), a_pos.unsqueeze(0), dim=-1).squeeze(0)
            neg_sim = F.cosine_similarity(z_pos.unsqueeze(0), a_neg.unsqueeze(0), dim=-1).squeeze(0)
            loss_term = F.relu(HNCL_MARGIN - pos_sim + neg_sim)
            loss_terms.append(loss_term)

        if len(loss_terms) == 0:
            return torch.tensor(0.0, device=dynamic_label_repr.device)

        return torch.stack(loss_terms).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        inputs["input_ids"] = inputs["input_ids"].squeeze(0)

        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(0)

        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)

        epoch_float = self.state.epoch if self.state.epoch is not None else 0.0
        current_epoch = int(epoch_float)

        gate_factor = get_three_stage_gate(current_epoch, EPOCHS) if USE_STAGE_GATE else STRUCTURE_GATE_VALUE

        outputs = model(
            **inputs,
            gate_factor=gate_factor,
            topk_common=TOPK_COMMON
        )

        logits = outputs["logits"]
        cls_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(logits, labels)

        total_loss = cls_loss

        if USE_SSDC and outputs.get("dynamic_label_repr") is not None and outputs.get("static_anchors") is not None:
            ssdc_weight = get_ssdc_warmup_weight(
                current_epoch,
                warmup_epochs=SSDC_WARMUP_EPOCHS,
                target_weight=LAMBDA_SSDC
            )
            ssdc_loss = compute_cosine_similarity_loss(
                outputs["dynamic_label_repr"],
                outputs["static_anchors"]
            )

            if USE_U_SFC:
                usfc_align_loss, per_label_align_loss, usfc_weights = self._compute_usfc_align_loss(
                    logits=logits,
                    labels=labels,
                    dynamic_label_repr=outputs["dynamic_label_repr"],
                    static_anchors=outputs["static_anchors"]
                )
                align_loss = USFC_MIX_ALPHA * ssdc_loss + (1.0 - USFC_MIX_ALPHA) * usfc_align_loss
                total_loss = total_loss + ssdc_weight * align_loss
            else:
                usfc_align_loss = ssdc_loss
                per_label_align_loss = None
                usfc_weights = None
                align_loss = ssdc_loss
                total_loss = total_loss + ssdc_weight * ssdc_loss

            if USE_FG_HNCL and model.training:
                hncl_weight = min(max(epoch_float, 0.0) / max(float(HNCL_WARMUP_EPOCHS), 1.0), 1.0) * LAMBDA_HNCL
                global_step = max(int(self.state.global_step), 0)

                if global_step % HNCL_COMPUTE_INTERVAL == 0:
                    hncl_loss = self._compute_hncl_loss(
                        labels=labels,
                        dynamic_label_repr=outputs["dynamic_label_repr"],
                        static_anchors=outputs["static_anchors"]
                    )
                    total_loss = total_loss + hncl_weight * hncl_loss
                else:
                    hncl_loss = torch.tensor(0.0, device=logits.device)
            else:
                hncl_loss = torch.tensor(0.0, device=logits.device)
        else:
            ssdc_loss = torch.tensor(0.0, device=logits.device)
            usfc_align_loss = torch.tensor(0.0, device=logits.device)
            hncl_loss = torch.tensor(0.0, device=logits.device)
            per_label_align_loss = None
            usfc_weights = None
            align_loss = torch.tensor(0.0, device=logits.device)

        if return_outputs:
            aux_outputs = {
                "logits": logits,
                "labels": labels,
                "cls_loss": cls_loss.detach(),
                "ssdc_loss": ssdc_loss.detach(),
                "usfc_align_loss": usfc_align_loss.detach(),
                "align_loss": align_loss.detach(),
                "hncl_loss": hncl_loss.detach(),
            }
            if per_label_align_loss is not None:
                aux_outputs["per_label_align_loss"] = per_label_align_loss
            if usfc_weights is not None:
                aux_outputs["usfc_weights"] = usfc_weights
            return total_loss, aux_outputs

        return total_loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.autocast_smart_context_manager():
                loss, outputs = self.compute_loss(model=model, inputs=inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, outputs["logits"], outputs["labels"])


def precision_at_k(y_true, y_pred, k: int = 3) -> float:
    head_k_args = np.argsort(np.sum(y_true, axis=0))[-k:]
    sliced_y_true = np.squeeze(y_true[:, [head_k_args]], axis=1)
    sliced_y_pred = np.squeeze(y_pred[:, [head_k_args]], axis=1)
    return precision_score(sliced_y_true, sliced_y_pred, average="micro", zero_division=0)


def recall_at_k(y_true, y_pred, k: int = 3) -> float:
    head_k_args = np.argsort(np.sum(y_true, axis=0))[-k:]
    sliced_y_true = np.squeeze(y_true[:, [head_k_args]], axis=1)
    sliced_y_pred = np.squeeze(y_pred[:, [head_k_args]], axis=1)
    return recall_score(sliced_y_true, sliced_y_pred, average="micro", zero_division=0)


def sigmoid_np(logits):
    return 1.0 / (1.0 + np.exp(-logits))


def threshold_grid():
    if not USE_THRESHOLD_CALIBRATION:
        return np.array([DEFAULT_THRESHOLD], dtype=np.float32)
    return np.arange(
        THRESHOLD_SEARCH_MIN,
        THRESHOLD_SEARCH_MAX + THRESHOLD_SEARCH_STEP * 0.5,
        THRESHOLD_SEARCH_STEP,
        dtype=np.float32,
    )


def search_best_threshold(labels, logits):
    probs = sigmoid_np(logits)
    best_threshold = float(DEFAULT_THRESHOLD)
    best_micro = -1.0
    best_macro = 0.0

    for threshold in threshold_grid():
        preds = (probs >= float(threshold)).astype(np.int32)
        micro = f1_score(labels, preds, average="micro", zero_division=0)
        macro = f1_score(labels, preds, average="macro", zero_division=0)
        if micro > best_micro:
            best_micro = float(micro)
            best_macro = float(macro)
            best_threshold = float(threshold)

    return best_threshold, best_micro, best_macro


def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    best_threshold, _, _ = search_best_threshold(labels, logits)
    preds = (sigmoid_np(logits) >= best_threshold).astype(np.int32)

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)

    precision_at_5 = precision_at_k(y_true=labels, y_pred=preds, k=5)
    precision_at_8 = precision_at_k(y_true=labels, y_pred=preds, k=8)
    precision_at_15 = precision_at_k(y_true=labels, y_pred=preds, k=15)

    recall_at_5 = recall_at_k(y_true=labels, y_pred=preds, k=5)
    recall_at_8 = recall_at_k(y_true=labels, y_pred=preds, k=8)
    recall_at_15 = recall_at_k(y_true=labels, y_pred=preds, k=15)

    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_at_5": precision_at_5,
        "precision_at_8": precision_at_8,
        "precision_at_15": precision_at_15,
        "recall_at_5": recall_at_5,
        "recall_at_8": recall_at_8,
        "recall_at_15": recall_at_15,
        "threshold": best_threshold,
    }
