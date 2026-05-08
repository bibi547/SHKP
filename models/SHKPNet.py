import torch
import torch.nn as nn
import torch.nn.functional as F
from models.query_deocoder import QueryDecoder
from models.mask_decoder import MaskDecoder
from models.transformer import TwoWayTransformer
from models.common import SharedMLP1d


class SHKP_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        self.query_num = args.query_num
        self.dim = 256

        # image feature → 64×64×256
        self.img_decoder = nn.Sequential(
            nn.Conv2d(1536, self.dim, 1), # dino 1536 clip 768
            nn.GELU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        )

        # queries: [B,Q,256]
        self.query_decoder = QueryDecoder(
            d_model=self.dim,
            num_queries=self.query_num,
        )

        self.heat_decoder = MaskDecoder(transformer_dim=self.dim,num_query=self.query_num, transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.dim,
                mlp_dim=2048,
                num_heads=8,
            ),)

        self.prob_head = nn.Sequential(SharedMLP1d([256, 64], args.norm),
                                      nn.Dropout(0.),
                                      nn.Conv1d(64, 2, kernel_size=1), )
        self.vis_head = nn.Sequential(SharedMLP1d([256, 64], args.norm),
                                       nn.Dropout(0.),
                                       nn.Conv1d(64, 1, kernel_size=1), )

    def forward(self, img_emb, img_feat):
        B, C, H, W = img_feat.shape

        img_feat = self.img_decoder(img_feat)  # B,256,64,64
        img_emb = img_feat.flatten(2).transpose(1, 2)  # B,4096,256

        # queries
        q_emb = self.query_decoder(img_emb)   # B,Q,256

        heats, query_emb = self.heat_decoder(img_feat, q_emb)

        probs = self.prob_head(query_emb.permute(0, 2, 1)).permute(0, 2, 1)
        vis = self.vis_head(query_emb.permute(0, 2, 1)).permute(0, 2, 1)

        return probs, heats, vis
