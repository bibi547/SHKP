import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QueryDecoder(nn.Module):
    def __init__(self, d_model=256, num_queries=50, num_layers=5,
                 dropout=0.15, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        self.query_embed = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.kaiming_uniform_(self.query_embed, a=math.sqrt(5))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout
        )

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, image_embed):
        """
        Args:
            image_embed: [B, H*W, C] - fused image embeddings
        Returns:
            prompt_embed: [B, num_queries, d_model] - generated prompt embeddings
            labels: [B, num_queries] - confidence for each prompt embedding, range [0, 1]
        """
        B, _, _ = image_embed.shape
        device = image_embed.device

        query_embed = self.query_embed.to(device)
        query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)

        prompt_embed = self.transformer_decoder(query_embed, image_embed)

        return prompt_embed