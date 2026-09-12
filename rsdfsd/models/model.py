import os
import pickle
import torch
import torch.nn.functional as F
from torch import nn
try:
    from opt_einsum import contract
except ModuleNotFoundError:
    def contract(equation, *operands, **kwargs):
        return torch.einsum(equation, *operands)
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, AutoConfig

from rsdfsd_config import *
from rsdfsd.models.graph import Directed_Bipartite_Graph_Encoder


class DotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(DotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, WQ, WK, WV, a):
        WK = torch.tanh(WK)
        WV = torch.tanh(WV)

        scores = contract("bzth,czh->bczt", WK, WQ)
        num_chunks, num_codes, n_heads, token_length = scores.size()

        scores = scores.to(DEVICE)
        a = a.unsqueeze(1).unsqueeze(2)
        attn_mask = a.expand(num_chunks, num_codes, n_heads, token_length)
        attn = nn.Softmax(dim=-1)(scores + attn_mask)

        context = contract("bzth,bczt->bczh", WV, attn)
        return context, attn


class SRD_Guided_Multi_Label_Attention(nn.Module):
    def __init__(self, d_model, transform_size, graph_att_hidd, graph_ffn_hidd, n_heads,
                 ground_ind_tail_file, ground_ind_head_file,
                 adj_matrix_file, edge_feature_indices_file, head_degree_file,
                 srd_num_bins: int):
        super(SRD_Guided_Multi_Label_Attention, self).__init__()

        self.d_k = int(transform_size / n_heads)
        self.d_v = self.d_k
        self.n_heads = n_heads
        self.transform_size = transform_size

        self.trans = nn.Linear(d_model, self.d_k * n_heads)
        self.Q_proj = nn.Linear(self.d_k * n_heads, self.d_k * n_heads)
        self.K_proj = nn.Linear(d_model, self.d_k * n_heads)
        self.V_proj = nn.Linear(d_model, self.d_k * n_heads)
        self.layernorm = nn.LayerNorm(self.d_k * n_heads)
        self.W = nn.Linear(self.d_k * n_heads, self.d_k * n_heads, bias=False)

        with open(ground_ind_tail_file, "rb") as f:
            self.ground_ind_tail = list(pickle.load(f).values())
        with open(ground_ind_head_file, "rb") as f:
            self.ground_ind_head = list(pickle.load(f).values())

        # 读取新版图本体：稀疏 adj + 多统计量关系分解边 + head_degree。
        with open(adj_matrix_file, "rb") as f:
            adj_raw = pickle.load(f)
        if isinstance(adj_raw, torch.Tensor):
            self.adj_matrix = adj_raw.clone().detach().to(dtype=torch.int64, device=DEVICE)
        else:
            self.adj_matrix = torch.tensor(adj_raw, dtype=torch.int64, device=DEVICE)

        with open(edge_feature_indices_file, "rb") as f:
            edge_feat_raw = pickle.load(f)
        if isinstance(edge_feat_raw, torch.Tensor):
            self.edge_feature_indices = edge_feat_raw.clone().detach().to(dtype=torch.long, device=DEVICE)
        else:
            self.edge_feature_indices = torch.tensor(edge_feat_raw, dtype=torch.long, device=DEVICE)

        with open(head_degree_file, "rb") as f:
            head_degree_raw = pickle.load(f)
        if isinstance(head_degree_raw, torch.Tensor):
            self.head_degree = head_degree_raw.clone().detach().to(dtype=torch.float32, device=DEVICE)
        else:
            self.head_degree = torch.tensor(head_degree_raw, dtype=torch.float32, device=DEVICE)

        # graph encoder 内部从旧版 c_indices attention 切换为 SRD attention，
        # 但对外仍然保持同一条图传播主链，课程式 gate 不变。
        self.graph_encoder = Directed_Bipartite_Graph_Encoder(
            num_encoder_layers=GRAPH_NUM,
            att_hidd_dim=graph_att_hidd,
            ffn_hidd_dim=graph_ffn_hidd,
            num_attention_heads=NUM_ATT_HEAD,
            dropout=0.1,
            edge_feature_indices=self.edge_feature_indices,
            head_degree=self.head_degree,
            ground_ind_tail=self.ground_ind_tail,
            ground_ind_head=self.ground_ind_head,
            num_bins=srd_num_bins,
        ).to(DEVICE)

    def forward(self, Q, H, a, gate_factor=1.0, topk_common=0):
        Qg = self.trans(Q)
        Qg = torch.tanh(Qg)
        Qg = self.graph_encoder(Qg, self.adj_matrix, gate_factor=gate_factor, topk_common=topk_common)

        n_classes = Qg.size(0)
        chunk_num = H.size(0)

        q = self.Q_proj(Qg).view(n_classes, self.n_heads, self.d_k)
        k = self.K_proj(H).view(chunk_num, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.V_proj(H).view(chunk_num, -1, self.n_heads, self.d_k).transpose(1, 2)

        context, token_attn = DotProductAttention(self.d_k)(q, k, v, a)
        context = context.reshape(chunk_num, n_classes, self.n_heads * self.d_v)

        Qg_proj = self.W(Qg)
        chunk_logits = contract("bch,ch->bc", context, Qg_proj)
        dynamic_label_repr = context.mean(dim=0)

        return {
            "chunk_logits": chunk_logits,
            "graph_label_repr": Qg,
            "dynamic_label_repr": dynamic_label_repr,
            "token_attn": token_attn,
        }


class RSDFSD(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super(RSDFSD, self).__init__(config)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            local_files_only=getattr(config, "local_files_only", True),
        ).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=getattr(config, "local_files_only", True),
        )

        embed_cache_file = OUTPUT_DIR + "/{}_embeddings.pkl".format(CODE_TYPE)
        if not os.path.exists(embed_cache_file):
            with open(config.code_token_file, "rb") as f:
                code_token_raw = pickle.load(f)

            if hasattr(code_token_raw, "data"):
                code_token_raw = code_token_raw.data
            if not isinstance(code_token_raw, dict):
                raise TypeError(
                    f"code_token_file 内容必须是 dict 或 BatchEncoding，当前是: {type(code_token_raw)}"
                )

            self.code_token = {}
            for k, v in code_token_raw.items():
                if not torch.is_tensor(v):
                    v = torch.tensor(v)
                self.code_token[k] = v.long()

            dataset = TensorDataset(
                self.code_token["input_ids"],
                self.code_token["token_type_ids"],
                self.code_token["attention_mask"],
            )

            self.out_code = []
            self.model.eval()
            with torch.no_grad():
                label_token_dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
                for input_ids_batch, token_type_ids_batch, attention_mask_batch in label_token_dataloader:
                    input_ids_batch = input_ids_batch.to(DEVICE)
                    token_type_ids_batch = token_type_ids_batch.to(DEVICE)
                    attention_mask_batch = attention_mask_batch.to(DEVICE)

                    out_code = self.model.bert(
                        input_ids=input_ids_batch,
                        token_type_ids=token_type_ids_batch,
                        attention_mask=attention_mask_batch,
                        return_dict=False,
                    )[0]
                    self.out_code.append(self._mean_pool_label_embeddings(out_code, attention_mask_batch))

            self.out_code = torch.cat(self.out_code, dim=0).cpu()
            with open(embed_cache_file, "wb") as f:
                pickle.dump(self.out_code, f)
        else:
            with open(embed_cache_file, "rb") as f:
                self.out_code = pickle.load(f)

        self.label_delta = nn.Parameter(torch.zeros_like(self.out_code, dtype=torch.float32))
        self.label_delta_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        # 模型对外仍叫 attention，但内部已从 Co-occurrence 图切换为 SRD 图本体。
        self.attention = SRD_Guided_Multi_Label_Attention(
            config.attention_hidden_size,
            config.transform_size,
            config.graph_att_hidd,
            config.graph_ffn_hidd,
            config.n_heads,
            config.ground_ind_tail_file,
            config.ground_ind_head_file,
            config.adj_matrix_file,
            config.edge_feature_indices_file,
            config.head_degree_file,
            getattr(config, 'srd_num_bins', SRD_NUM_BINS),
        )

        self.static_anchors = None
        self.static_anchor_codes = None
        self.anchor_proj = None
        if USE_SSDC and os.path.exists(STATIC_ANCHOR_FILE):
            with open(STATIC_ANCHOR_FILE, "rb") as f:
                anchor_payload = pickle.load(f)

            self.static_anchor_codes = anchor_payload.get("codes", None)
            static_anchors = anchor_payload["anchors"]
            if not isinstance(static_anchors, torch.Tensor):
                static_anchors = torch.tensor(static_anchors, dtype=torch.float32)
            static_anchors = static_anchors.float()

            self.static_anchors = static_anchors
            self.anchor_proj = nn.Linear(self.static_anchors.size(1), config.transform_size, bias=False)

            print(f"[SSDC] loaded static anchors: shape={tuple(self.static_anchors.shape)}")
            if self.static_anchor_codes is not None:
                print(f"[SSDC] num anchor codes: {len(self.static_anchor_codes)}")

    @staticmethod
    def _mean_pool_label_embeddings(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        pooled = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return pooled / denom

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        gate_factor: float = 1.0,
        topk_common: int = 0,
    ):
        encoder_outputs = self.model.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        sentences_encodings = encoder_outputs[0]

        base_codes = self.out_code.to(sentences_encodings.device).float()
        codes = base_codes + self.label_delta.to(sentences_encodings.device) * self.label_delta_scale.to(sentences_encodings.device)
        documents = sentences_encodings

        attn_out = self.attention(
            codes,
            documents,
            attention_mask,
            gate_factor=gate_factor,
            topk_common=topk_common,
        )

        classification_outputs = attn_out["chunk_logits"]
        logits = nn.MaxPool1d(
            classification_outputs.shape[0],
            stride=classification_outputs.shape[0],
        )(classification_outputs.permute(1, 0)).T

        dynamic_label_repr = attn_out["dynamic_label_repr"].unsqueeze(0)
        dynamic_label_repr = F.normalize(dynamic_label_repr.float(), p=2, dim=-1)

        if self.static_anchors is not None and self.anchor_proj is not None:
            static_anchors = self.static_anchors.to(sentences_encodings.device)
            static_anchors = self.anchor_proj(static_anchors)
            static_anchors = F.normalize(static_anchors.float(), p=2, dim=-1)
            static_anchors = static_anchors.unsqueeze(0)
        else:
            static_anchors = None

        return {
            "logits": logits,
            "graph_label_repr": attn_out["graph_label_repr"],
            "dynamic_label_repr": dynamic_label_repr,
            "static_anchors": static_anchors,
            "token_attn": attn_out["token_attn"],
        }


class Model(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super(Model, self).__init__(config)
        self.model = RSDFSD(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        gate_factor: float = 1.0,
        topk_common: int = 0,
    ):
        return self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            gate_factor=gate_factor,
            topk_common=topk_common,
        )
