import os
import pickle
import torch
import torch.nn.functional as F
from rsdfsd_config import *

def load_static_anchor_payload(anchor_file):
    with open(anchor_file, "rb") as f:
        payload = pickle.load(f)
    return payload


def build_hard_negative_bank(anchor_payload, topk=5):
    if "anchors" not in anchor_payload:
        raise KeyError("static_anchors.pkl must contain key 'anchors'.")

    anchors = anchor_payload["anchors"]
    codes = anchor_payload.get("codes", None)
    texts = anchor_payload.get("texts", None)

    if codes is None:
        raise KeyError("static_anchors.pkl must contain key 'codes' for FG-HNCL alignment checking.")

    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors, dtype=torch.float32)
    anchors = F.normalize(anchors.float(), p=2, dim=-1)

    num_labels = anchors.size(0)
    similarity_matrix = torch.matmul(anchors, anchors.T)
    similarity_matrix.fill_diagonal_(-1e9)

    topk = min(topk, max(num_labels - 1, 1))
    topk_sims, topk_indices = torch.topk(similarity_matrix, k=topk, dim=-1)

    bank_payload = {
        "codes": codes,
        "texts": texts,
        "topk_neg_indices": topk_indices.cpu().numpy(),
        "topk_neg_sims": topk_sims.cpu().numpy(),
        "meta": {
            "source": "static_anchor_similarity",
            "anchor_source": anchor_payload.get("source", None),
            "anchor_source_type": anchor_payload.get("source_type", None),
            "anchor_mode": anchor_payload.get("mode", None),
            "anchor_encoder": anchor_payload.get("encoder", None),
            "metric": "cosine",
            "topk": topk,
            "num_labels": num_labels,
        }
    }
    return bank_payload


def main():
    if not os.path.exists(STATIC_ANCHOR_FILE):
        raise FileNotFoundError(f"static anchor file not found: {STATIC_ANCHOR_FILE}")

    anchor_payload = load_static_anchor_payload(STATIC_ANCHOR_FILE)

    bank_payload = build_hard_negative_bank(
        anchor_payload=anchor_payload,
        topk=HNCL_TOPK
    )

    with open(HARD_NEG_BANK_FILE, "wb") as f:
        pickle.dump(bank_payload, f)

    print(f"[FG-HNCL] hard negative bank saved to: {HARD_NEG_BANK_FILE}")
    print(f"[FG-HNCL] shape(topk_neg_indices)={bank_payload['topk_neg_indices'].shape}")
    print(f"[FG-HNCL] topk={bank_payload['meta']['topk']}")
if __name__ == "__main__":
    main()
