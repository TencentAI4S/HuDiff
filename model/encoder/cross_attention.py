import torch
import torch.nn as nn
from inspect import isfunction

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class AttLayer(nn.Module):

    def __init__(self, d_model, att_model, nhead, length=152):
        super().__init__()
        self.nhead = nhead
        self.dk = (att_model / nhead) ** 0.5

        self.query = nn.Linear(d_model, att_model)
        self.key = nn.Linear(d_model, att_model)
        self.value = nn.Linear(d_model, att_model)
        self.softmax = nn.Softmax(dim=-1)

        self.out_put = nn.Linear(att_model, d_model)

        rope = precompute_freqs_cis(att_model//nhead, length)
        self.register_buffer('rope', rope)


    def forward(self, x, context=None, mask=None):
        Q = self.query(x)
        if context is None:
            K = self.key(x)
            V = self.value(x)
        else:
            K = self.key(context)
            V = self.key(context)

        Q = Q.view(Q.shape[0], Q.shape[1], self.nhead, -1)  #.permute(0, 2, 1, 3)
        K = K.view(K.shape[0], K.shape[1], self.nhead, -1)  #.permute(0, 2, 1, 3)
        V = V.view(V.shape[0], V.shape[1], self.nhead, -1)  #.permute(0, 2, 1, 3)

        Q, K = apply_rotary_emb(Q, K, self.rope)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.dk
        attn_weights = self.softmax(attn_weights)
        output = torch.matmul(attn_weights, V)
        output = output.permute(0, 2, 1, 3).contiguous().view(Q.shape[0], -1, Q.shape[1]*Q.shape[3])
        output = self.out_put(output)
        return output


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


class TransformerNet(nn.Module):
    def __init__(self, d_model, att_model, nhead, num_layers, dim_feedforward):
        super(TransformerNet, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, att_model, dim_feedforward) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class CrossAttBlock(nn.Module):

    def __init__(self, d_model, att_model, dim_feedforward, nhead):
        super().__init__()
        self.attnh = AttLayer(d_model, att_model, nhead)
        self.attn_hc = AttLayer(d_model, att_model, nhead)
        self.attnl = AttLayer(d_model, att_model, nhead)
        self.attn_lc = AttLayer(d_model, att_model, nhead)

        self.normh1 = nn.LayerNorm(d_model)
        self.normh2 = nn.LayerNorm(d_model)

        self.norml1 = nn.LayerNorm(d_model)
        self.norml2 = nn.LayerNorm(d_model)

        self.ffh = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.ffl = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, h, l, mask=None):
        """

        :param h:
        :param l:
        :param mask:
        :return:
        """
        at_h = h + self.attnh(h)
        at_l = l + self.attnl(l)

        at_h = at_h + self.attn_hc(self.normh1(h), l)
        at_l = at_l + self.attn_lc(self.norml1(l), h)

        h = self.ffh(self.normh2(at_h)) + at_h
        l = self.ffl(self.norml2(at_l)) + at_l
        return h, l


class SelfAttBlock(nn.Module):

    def __init__(self, d_model, att_model, dim_feedforward, nhead, rolength):
        super().__init__()
        self.attn_hl = AttLayer(d_model, att_model, nhead, rolength)
        self.attn_hl_c = AttLayer(d_model, att_model, nhead, rolength)

        self.norm_hl1 = nn.LayerNorm(d_model)
        self.norm_hl2 = nn.LayerNorm(d_model)

        self.ff_hl = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, h_l, mask=None):
        """

        :param h:
        :param l:
        :param mask:
        :return:
        """

        at_hl = h_l + self.attn_hl(h_l)

        at_hl = at_hl + self.attn_hl_c(self.norm_hl1(at_hl))

        h_l = self.ff_hl(self.norm_hl2(at_hl)) + h_l
        return h_l



class SelfAttNet(nn.Module):

    def __init__(self, d_model, att_model, dim_feedforward, nhead, rolength, num_cross_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [SelfAttBlock(d_model, att_model, dim_feedforward, nhead, rolength) for _ in range(num_cross_layers)]
        )

    def forward(self, h_l, mask=None):
        """

        :param h:
        :param l:
        :param mask:
        :return:
        """
        # h_l = h_l + pos_emb
        for layer in self.layers:
            h_l = layer(h_l)
        return h_l


class CrossAttNet(nn.Module):

    def __init__(self, d_model, att_model, dim_feedforward, nhead, num_cross_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttBlock(d_model, att_model, dim_feedforward, nhead) for _ in range(num_cross_layers)]
        )

    def forward(self, h, l, pos_emb, mask=None):
        """

        :param h:
        :param l:
        :param mask:
        :return:
        """
        h = h + pos_emb[:h.size(0)]
        l = l + pos_emb[h.size(0):]
        for layer in self.layers:
            h, l = layer(h, l)
        return h, l
