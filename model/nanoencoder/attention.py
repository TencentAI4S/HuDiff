import torch
import torch.nn as nn
from inspect import isfunction

import torch
from ..encoder.cross_attention import AttLayer


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, att_model, dim_feedforward):
        super(TransformerBlock, self).__init__()
        self.attention = AttLayer(d_model, att_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = x + attn_output
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        return x



class SelfAttBlock(nn.Module):

    def __init__(self, d_model, att_model, dim_feedforward, nhead):
        super().__init__()
        self.attn_h = AttLayer(d_model, att_model, nhead)
        self.attn_h_c = AttLayer(d_model, att_model, nhead)

        self.norm_h1 = nn.LayerNorm(d_model)
        self.norm_h2 = nn.LayerNorm(d_model)

        self.ff_h = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, h, mask=None):
        """

        :param h:
        :param l:
        :param mask:
        :return:
        """

        at_h = h + self.attn_hl(h)

        at_h = at_h + self.attn_h_c(self.norm_h1(at_h))

        h = self.ff_h(self.norm_h2(at_h)) + h
        return h



class SelfAttNet(nn.Module):

    def __init__(self, d_model, att_model, dim_feedforward, nhead, num_cross_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [SelfAttBlock(d_model, att_model, dim_feedforward, nhead) for _ in range(num_cross_layers)]
        )

    def forward(self, h, mask=None):
        """

        :param h:
        :param l:
        :param mask:
        :return:
        """
        # h_l = h_l + pos_emb
        for layer in self.layers:
            h = layer(h)
        return h