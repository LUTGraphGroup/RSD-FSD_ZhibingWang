import torch
import torch.nn as nn
import torch.nn.functional as F
from rsdfsd_config import *


class RelationDecomposedEdgeEncoder(nn.Module):
    def __init__(self, num_bins: int, num_heads: int, num_features: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_bins, num_heads) for _ in range(num_features)
        ])

    def forward(self, edge_feature_indices: torch.Tensor) -> torch.Tensor:
        # edge_feature_indices: [Nt, Nh, F]
        bias_terms = []
        for feat_id, emb in enumerate(self.embeddings):
            feat_idx = edge_feature_indices[..., feat_id].long()
            bias_terms.append(emb(feat_idx))  # [Nt, Nh, H]
        edge_bias = torch.stack(bias_terms, dim=0).sum(dim=0)  # [Nt, Nh, H]
        return edge_bias


class Directed_Bipartite_MH_Attention(nn.Module):
    def __init__(self, att_hidd_dim=None, num_heads=None, dropout=None,
                 edge_feature_indices=None, head_degree=None,
                 ground_ind_tail=None, ground_ind_head=None,
                 num_bins: int = 16):
        super().__init__()
        self.att_hidd_dim = att_hidd_dim
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)
        self.head_dim = att_hidd_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        if not torch.is_tensor(edge_feature_indices):
            edge_feature_indices = torch.tensor(edge_feature_indices, dtype=torch.long)
        if not torch.is_tensor(head_degree):
            head_degree = torch.tensor(head_degree, dtype=torch.float32)

        self.register_buffer("edge_feature_indices", edge_feature_indices.long())
        self.register_buffer("head_degree", head_degree.float())

        num_features = int(self.edge_feature_indices.size(-1))
        self.edge_encoder = RelationDecomposedEdgeEncoder(
            num_bins=num_bins,
            num_heads=num_heads,
            num_features=num_features,
        )


        self.hub_lambda = nn.Parameter(torch.tensor(float(SRD_HUB_PENALTY_INIT)))

        self.ground_ind_tail = torch.tensor(ground_ind_tail, dtype=torch.long, device=DEVICE)
        self.ground_ind_head = torch.tensor(ground_ind_head, dtype=torch.long, device=DEVICE)

        self.q_proj = nn.Linear(att_hidd_dim, att_hidd_dim)
        self.k_proj = nn.Linear(att_hidd_dim, att_hidd_dim)
        self.v_proj = nn.Linear(att_hidd_dim, att_hidd_dim)
        self.out_proj = nn.Linear(att_hidd_dim, att_hidd_dim)

    def forward(self, query=None, key=None, value=None, adj_matrix=None,
                gate_factor=1.0, topk_common=0):
        q_tail = (self.q_proj(query[self.ground_ind_tail]) * self.scaling).view(
            -1, self.num_heads, self.head_dim
        ).transpose(0, 1)

        k_head = self.k_proj(key[self.ground_ind_head]).view(
            -1, self.num_heads, self.head_dim
        ).transpose(0, 1)

        v_head = self.v_proj(value[self.ground_ind_head]).view(
            -1, self.num_heads, self.head_dim
        ).transpose(0, 1)

        tail_mask = torch.where(adj_matrix == 0, -1e9, 0.0).unsqueeze(0).to(q_tail.dtype)

        edge_bias = self.edge_encoder(self.edge_feature_indices).permute(2, 0, 1).to(q_tail.dtype)

        hub_penalty = self.hub_lambda * torch.log(self.head_degree + 1.0)
        hub_penalty = hub_penalty.view(1, 1, -1).to(q_tail.dtype)

        attn_scores = torch.bmm(q_tail, k_head.transpose(1, 2)) + tail_mask

        if BIAS_TYPE == "Y":
            structural_term = edge_bias - hub_penalty
            attn_scores = attn_scores + gate_factor * structural_term

        _ = topk_common

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout_module(attn_probs)

        attn_values_tail = attn_probs @ v_head

        attn_values_all = torch.zeros(
            self.num_heads,
            len(query),
            self.head_dim,
            dtype=query.dtype,
            device=query.device,
        )
        attn_values_all[:, self.ground_ind_tail, :] = attn_values_tail
        attn_values_all[:, self.ground_ind_head, :] = query[self.ground_ind_head].view(
            -1, self.num_heads, self.head_dim
        ).transpose(0, 1)

        attn_values_all = attn_values_all.transpose(0, 1).contiguous().view(-1, self.att_hidd_dim)
        return self.out_proj(attn_values_all)


class Directed_Bipartite_Graph_EncoderLayer(nn.Module):
    def __init__(self, att_hidd_dim, ffn_hidd_dim, num_attention_heads,
                 dropout, edge_feature_indices, head_degree,
                 ground_ind_tail, ground_ind_head, num_bins):
        super().__init__()
        self.self_attn = Directed_Bipartite_MH_Attention(
            att_hidd_dim=att_hidd_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            edge_feature_indices=edge_feature_indices,
            head_degree=head_degree,
            ground_ind_tail=ground_ind_tail,
            ground_ind_head=ground_ind_head,
            num_bins=num_bins,
        )

        self.self_attn_layer_norm = nn.LayerNorm(att_hidd_dim)
        self.fc1 = nn.Linear(att_hidd_dim, ffn_hidd_dim)
        self.fc2 = nn.Linear(ffn_hidd_dim, att_hidd_dim)
        self.final_layer_norm = nn.LayerNorm(att_hidd_dim)
        self.dropout_module = nn.Dropout(dropout)
        self.activation_fn = nn.GELU()

    def forward(self, x, adj_matrix=None, gate_factor=1.0, topk_common=0):
        x = x + self.dropout_module(
            self.self_attn(x, x, x, adj_matrix, gate_factor=gate_factor, topk_common=topk_common)
        )
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class Directed_Bipartite_Graph_Encoder(nn.Module):
    def __init__(self, num_encoder_layers=None, att_hidd_dim=None,
                 ffn_hidd_dim=None, num_attention_heads=None, dropout=None,
                 edge_feature_indices=None, head_degree=None,
                 ground_ind_tail=None, ground_ind_head=None,
                 num_bins: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([
            Directed_Bipartite_Graph_EncoderLayer(
                att_hidd_dim=att_hidd_dim,
                ffn_hidd_dim=ffn_hidd_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                edge_feature_indices=edge_feature_indices,
                head_degree=head_degree,
                ground_ind_tail=ground_ind_tail,
                ground_ind_head=ground_ind_head,
                num_bins=num_bins,
            )
            for _ in range(num_encoder_layers)
        ])

    def forward(self, x, adj_matrix, gate_factor=1.0, topk_common=0):
        for layer in self.layers:
            x = layer(x, adj_matrix, gate_factor=gate_factor, topk_common=topk_common)
        return x
