import torch.nn.functional as F

def attention_pool(features, attention_layer, mask=None):
        # features: [batch, seq_len, hidden]

        # Calculate attention scores
        """Pool sequence features with learned attention weights and an optional mask."""
        attn_weights = attention_layer(features)  # [batch, seq_len, 1]

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_weights, dim=1)

        # Apply attention
        weighted_features = features * attn_weights
        pooled = weighted_features.sum(dim=1)  # [batch, hidden]

        return pooled