import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sequence_models.layers import PositionFeedForward, DoubleEmbedding
from sequence_models.convolutional import ByteNetBlock, MaskedConv1d
from .cross_attention import CrossAttNet, TransformerNet, SelfAttNet, precompute_freqs_cis
from collections import OrderedDict
import tempfile
import os
from pymol import cmd
from abnumber import Chain

# Abnativ
from ..nanoencoder.abnativ_scoring import get_abnativ_nativeness_scores

alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

class MLP(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.ln1 = nn.Linear(n_embd, 2 * n_embd)
        self.gelu = nn.GELU()
        self.ln2 = nn.Linear(2 * n_embd, n_embd)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.ln2(x)
        x = self.dropout(x)
        return x

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model=8, length=500):
        super().__init__()
        self.d_model = d_model
        self.length = length

    def forward(self, x):
        """
        Used for encoding timestep in diffusion models

        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if self.d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(self.d_model))
        pe = torch.zeros(self.length, self.d_model)
        position = torch.arange(0, self.length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) * -(np.log(10000.0) / self.d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        device = x.device
        pe = pe.to(device)
        return pe[x] # .to(x.device)


class PositionalEncoding(nn.Module):

    """
    2D Positional encoding for transformer
    :param d_model: dimension of the model
    :param max_len: max number of positions
    """

    def __init__(self, d_model, max_len=152):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        y = self.pe[:x.size(1)]
        x = x + y.reshape(y.shape[1], y.shape[0], y.shape[2])
        return x


class ByteNetTime(nn.Module):
    """Stacked residual blocks from ByteNet paper defined by n_layers

         Shape:
            Input: (N, L,)
            input_mask: (N, L, 1), optional
            Output: (N, L, d)
    """

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu', down_embed=False,
                 timesteps=None, aa_h_length=152, aa_l_length=139):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :padding_idx: location of padding token in ordered alphabet
        :param causal: if True, chooses MaskedCausalConv1d() over MaskedConv1d()
        :param rank: rank of compressed weight matrices
        :param n_frozen_embs: number of frozen embeddings
        :param slim: if True, use half as many dimensions in the NLP as in the CNN
        :param activation: 'relu' or 'gelu'
        :param down_embed: if True, have lower dimension for initial embedding than in CNN layers
        :param timesteps: None or int providing max timesteps in DM model
        """
        super().__init__()
        self.timesteps = timesteps
        self.time_encoding = PositionalEncoding1D(d_embedding, timesteps) # Timestep encoding
        if n_tokens is not None:
            if n_frozen_embs is None:
                self.embedder = nn.Embedding(n_tokens, d_embedding, padding_idx=padding_idx)
            else:
                self.embedder = DoubleEmbedding(n_tokens - n_frozen_embs, n_frozen_embs,
                                                d_embedding, padding_idx=padding_idx)
        else:
            self.embedder = nn.Identity()
        if down_embed:
            self.up_embedder = PositionFeedForward(d_embedding, d_model)
        else:
            self.up_embedder = nn.Identity()
            assert d_model == d_embedding
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        h_layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        l_layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        self.h_layers = nn.ModuleList(modules=h_layers)
        self.l_layers = nn.ModuleList(modules=l_layers)
        self.dropout = dropout
        self.aa_h_length = aa_h_length
        self.aa_l_length = aa_l_length

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length)
        :param y: (batch)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        e = self._embed(x, timesteps=self.timesteps)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x, timesteps=None):
        e = self.embedder(x)
        e = self.up_embedder(e)
        return e

    def _convolve(self, e, input_mask=None):
        h_e = e[:, :self.aa_h_length, :]
        l_e = e[:, self.aa_h_length:, :]
        for h_layer, l_layer in zip(self.h_layers, self.l_layers):
            h_e = h_layer(h_e, input_mask=input_mask)
            l_e = l_layer(l_e, input_mask=input_mask)
            if self.dropout > 0.0:
                h_e = F.dropout(h_e, self.dropout)
                l_e = F.dropout(l_e, self.dropout)
        e = torch.cat((h_e, l_e), dim=1)
        return e


class SideEmbedder(nn.Module):

    def __init__(self, n_side, s_embedding, d_side, aa_h_length=152, aa_l_length=139):
        super().__init__()
        self.side_embeddinng = nn.Embedding(n_side, s_embedding)
        self.side_mlp = nn.Sequential(
            nn.Linear(s_embedding, d_side),
            nn.LayerNorm(d_side),
            nn.ReLU(),
            nn.Linear(d_side, d_side),
        )
        self.aa_h_length = aa_h_length
        self.aa_l_length = aa_l_length

    def forward(self, side, mask=None):
        emb_side = self.side_embeddinng(side.view(-1, 1))
        emb_side = self.side_mlp(emb_side)
        h_emb_side = emb_side[side == 0]
        h_emb_side = h_emb_side.repeat(1, self.aa_h_length, 1)
        l_emb_side = emb_side[side != 0]
        l_emb_side = l_emb_side.repeat(1, self.aa_l_length, 1)
        emb_side = torch.cat((h_emb_side, l_emb_side), dim=1)
        return emb_side


class RegionEmbedder(nn.Module):

    def __init__(self, r_pos, r_embedding, r_model, rank=None):
        super().__init__()
        self.region_embedding = nn.Embedding(r_pos, r_embedding)
        self.region_layer1 = nn.Sequential(
            nn.LayerNorm(r_embedding),
            nn.ReLU(),
            PositionFeedForward(r_embedding, r_model, rank=rank),
            nn.LayerNorm(r_model),
            nn.ReLU()
        )


    def forward(self, pos_seq, mask=None):
        """
        :param pos_seq:
        :param mask:
        :return:
        """
        x = self.region_embedding(pos_seq)
        x = self.region_layer1(x)
        return x


class PosEmbedder(nn.Module):
    """
    This position embedding method is PE encoding.
    """
    def __init__(self, p_emb, max_len):
        super().__init__()
        self.pos_embedding = PositionalEncoding(p_emb, max_len)
        self.pos_lin = MLP(n_embd=p_emb)

    def forward(self, H_L_region_emb):
        x = self.pos_embedding(H_L_region_emb)
        pos_emb = self.pos_lin(x)
        x = x + pos_emb
        return x


class DualConv(nn.Module):

    def __init__(self, d_model, n_layers, kernel_size, r, rank=None,
                causal=False, dropout=0.5, slim=True, activation='relu', timesteps=None,
                 aa_h_length=152, aa_l_length=139):
        super().__init__()

        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        h_layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        l_layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        self.h_layers = nn.ModuleList(modules=h_layers)
        self.l_layers = nn.ModuleList(modules=l_layers)
        self.aa_h_length = aa_h_length
        self.aa_l_length = aa_l_length
        self.dropout = dropout

    def forward(self, s, mask=None):
        """

        :param s:
        :param batch:
        :param mask:
        :return:
        """
        h_s = s[:, :self.aa_h_length, :]
        l_s = s[:, self.aa_h_length:, :]

        h_s = self._hconv(h_s)
        l_s = self._lconv(l_s)
        return h_s, l_s

    def _hconv(self, h_s):
        for layer in self.h_layers:
            h_s = layer(h_s)
            if self.dropout > 0.0:
                h_s = F.dropout(h_s)
        return h_s

    def _lconv(self, l_s):
        for layer in self.l_layers:
            l_s = layer(l_s)
            if self.dropout > 0.0:
                l_s = F.dropout(l_s)
        return l_s

class ByteNetLMTime(nn.Module):
    pass


class TransformerEncoder(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, att_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        self.embed = nn.Embedding(n_tokens, d_embedding)
        self.up_embedder = PositionFeedForward(d_embedding, d_model)
        self.att_net = TransformerNet(d_model, att_model, nhead, num_layers=num_layers, dim_feedforward=dim_feedforward)

    def forward(self, x):
        emb_x = self.embed(x)
        up_emb_x = self.up_embedder(emb_x)
        x = self.att_net(up_emb_x)
        return x


class AntiTFNet(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, n_encoder_layers, aa_kernel_size, r,
                 n_side, s_embedding, s_model,
                 n_region, r_embedding, r_model,
                 n_pos_model, max_len,
                 sum_d_model, dual_layers,
                 att_model, dim_feedforward, nhead, cs_layers,
                 rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu',
                 down_embed=False, timesteps=None):
        super().__init__()

        self.aa_encoder = ByteNetTime(n_tokens, d_embedding, d_model, n_encoder_layers, aa_kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs,
                                timesteps=timesteps)
        self.side_encoder = SideEmbedder(n_side, s_embedding, s_model)
        self.region_encoder = RegionEmbedder(n_region, r_embedding, r_model)
        self.pos_encoder = PosEmbedder(n_pos_model, max_len)
        self.dual_conv_block = DualConv(sum_d_model, dual_layers, aa_kernel_size, r, dropout=dropout)
        self.self_at = SelfAttNet(sum_d_model, att_model, dim_feedforward, nhead, rolength=max_len, num_cross_layers=cs_layers)
        self.last_norm = nn.LayerNorm(sum_d_model)
        self.decoder = nn.Linear(sum_d_model, n_tokens)


    def _encoder(self, h_l_aa_seq, h_l_chn_type, h_l_region_type):
        H_L_emb = self.aa_encoder(h_l_aa_seq)
        aa_seq_length = H_L_emb.size(1)
        H_L_chn_emb = self.side_encoder(h_l_chn_type, aa_seq_length)
        H_L_region_emb = self.region_encoder(h_l_region_type)
        H_L_pos_emb = self.pos_encoder(H_L_region_emb)
        H_L_emb = H_L_emb + H_L_pos_emb + H_L_chn_emb
        h_l_feature = torch.cat((H_L_emb, H_L_pos_emb, H_L_chn_emb), dim=-1)
        return h_l_feature

    def _att(self, h, l):
        h_l = torch.cat((h, l), dim=1)
        h_l = self.self_at(h_l)
        return h_l

    def forward(self, H_L_seq, H_L_region_type, H_L_chn_type):
        """

        :param H_L_seq: (Batch, length);
        :param H_L_pos_type: (Batch, length); distinguish the different region of Chain.
        :param H_L_chn_type: (Batch); gene ?
        :param H_L_batch: (Batch); distinguish the type of Chain.
        :param H_L_mask: None
        :return: (Batch, length, feature)
        """
        h_l_feature = self._encoder(
            h_l_aa_seq=H_L_seq.int(),
            h_l_chn_type=H_L_chn_type,
            h_l_region_type=H_L_region_type.int(),
        )
        h, l = self.dual_conv_block(h_l_feature)   # ablation study.
        h_l = self._att(h, l)
        h_l = self.decoder(self.last_norm(h_l))
        return h_l


class AntiFrameWork(nn.Module):
    def __init__(self, config, pretrained_model_list, tokenizer, aa_heavy_length=152, aa_light_length=139) -> None:
        super().__init__()

        self.anti_infilling = pretrained_model_list['antibody_pretrained']
        self.ab_vh_model = pretrained_model_list['ab_vh_model']
        self.ab_vlk_model = pretrained_model_list['ab_vlk_model']
        self.ab_vll_model = pretrained_model_list['ab_vll_model']
        self.Tokenizer = tokenizer

        self.all_seq = config.all_seq
        self.loss_type = config.loss_type
        self.human_threshold = config.human_threshold
        self.mutation = config.mutation
        self.aa_heavy_length = aa_heavy_length
        self.aa_light_length = aa_light_length
        
        # Here is the fixed parameters:
        self.imgt_true_max_index = 20
        self.temperature = 1.
        # self.only_cdr3_h = None
        self.aho_pad_idx = 20
        self.imgt_heavy_length = 152
        self.imgt_heavy_tail = 150

        self.imgt_light_tail = 290
        self.abnativ_class = 21

        self.aho_heavy_length = 149
        self.aho_light_tail = 296
        self.aho_heavy_tail = 147
        self.light_lambda_idx = 1
        self.light_kappa_idx = 2

        self.heavy_mutation_threshold = 17
        self.light_mutation_threshold = 15
        self.norm_mutation = 10

    def forward(self, H_L_seq, H_L_region_type, H_L_chn_type, masked_idx, H_L_true, aho_h_seq, aho_l_seq, device):
        pred_seq, pred_logits = self.infilling_seq(H_L_seq, H_L_region_type, H_L_chn_type, masked_idx)
        aho_h_l_seq = torch.cat((aho_h_seq, aho_l_seq), dim=1)
        imgt_mask, aho_mask = self.get_corresponding_mask(H_L_true, aho_h_l_seq)
        infilling_aho_h_l_seq, infilling_aho_h_l_mask = self.trans_align_scheme(
            H_L_true,
            pred_seq,
            imgt_mask,
            aho_h_l_seq,
            aho_mask,
            masked_idx
        )
        infilling_aho_h_seq = infilling_aho_h_l_seq[:, :self.aho_heavy_length, :]
        infilling_aho_l_seq = infilling_aho_h_l_seq[:, self.aho_heavy_length:, :]
        infilling_aho_h_mask = infilling_aho_h_l_mask[:, :self.aho_heavy_length]
        infilling_aho_l_mask = infilling_aho_h_l_mask[:, self.aho_heavy_length:]

        output_aho_heavy_abnativ = self.ab_vh_model(infilling_aho_h_seq)
        heavy_chain_score = self.batch_get_each_seq_score(
                                            output_aho_heavy_abnativ,
                                            infilling_aho_h_mask,
                                            all_seq=self.all_seq,
                                            model_type='VH'
                                        )

        # Need consider the light chain type.
        L_chn_type = H_L_chn_type[H_L_chn_type != 0]

        kappa_light_mask = L_chn_type == self.light_kappa_idx
        lambda_light_mask = L_chn_type == self.light_lambda_idx

        infilling_aho_lk_seq = infilling_aho_l_seq[kappa_light_mask]
        infilling_aho_ll_seq = infilling_aho_l_seq[lambda_light_mask]

        infilling_aho_lk_mask = infilling_aho_l_mask[kappa_light_mask]
        infilling_aho_ll_mask = infilling_aho_l_mask[lambda_light_mask]

        if infilling_aho_lk_seq.size(0) != 0:
            output_aho_light_kappa_abnativ = self.ab_vlk_model(infilling_aho_lk_seq)
            light_kappa_chain_score = self.batch_get_each_seq_score(
                output_aho_light_kappa_abnativ,
                infilling_aho_lk_mask,
                all_seq=self.all_seq,
                model_type='VKappa'
            )
        else:
            light_kappa_chain_score = torch.empty((0))
        
        if infilling_aho_ll_seq.size(0) != 0:
            output_aho_light_lambda_abnativ = self.ab_vll_model(infilling_aho_ll_seq)
            light_lambda_chain_score = self.batch_get_each_seq_score(
                output_aho_light_lambda_abnativ,
                infilling_aho_ll_mask,
                all_seq=self.all_seq,
                model_type='VLambda'
            )
        else:
            light_lambda_chain_score = torch.empty((0))
        
        if self.loss_type == 'mse_loss':
            vh_loss = F.mse_loss(heavy_chain_score, torch.ones_like(heavy_chain_score) * self.human_threshold)
            if light_kappa_chain_score.size(0) != 0.:
                vlk_loss = F.mse_loss(light_kappa_chain_score, torch.ones_like(light_kappa_chain_score) * self.human_threshold)
            else:
                vlk_loss = 0.

            if light_lambda_chain_score.size(0) != 0.:
                vll_loss = F.mse_loss(light_lambda_chain_score, torch.ones_like(light_lambda_chain_score) * self.human_threshold)
            else:
                vll_loss = 0.
            ab_score_loss = vh_loss + vlk_loss + vll_loss
        elif self.loss_type == 'smooth_loss':
            vh_loss = F.smooth_l1_loss(heavy_chain_score, torch.ones_like(heavy_chain_score) * self.human_threshold)
            if light_kappa_chain_score.size(0) != 0.:
                vlk_loss = F.smooth_l1_loss(light_kappa_chain_score, 
                                            torch.ones_like(light_kappa_chain_score) * self.human_threshold,
                                            reduction='none'
                                            )
            else:
                vlk_loss = torch.zeros(0).to(device)

            if light_lambda_chain_score.size(0) != 0.:
                vll_loss = F.smooth_l1_loss(light_lambda_chain_score, 
                                            torch.ones_like(light_lambda_chain_score) * self.human_threshold,
                                            reduction='none'
                                            )
            else:
                vll_loss = torch.zeros(0).to(device)
            vl_loss = (vlk_loss.sum() + vll_loss.sum()) / L_chn_type.size(0)
            ab_score_loss = vh_loss + vl_loss
        
        if self.mutation:
            h_mutation_loss, l_mutation_loss = self.muation_loss(pred_logits, H_L_true, masked_idx)
            return ab_score_loss, pred_logits, h_mutation_loss, l_mutation_loss
        else:
            return ab_score_loss, pred_logits, 0,  0

    
    def muation_loss(self, predict_logits, true_seq, h_l_masked):
        _, predict_seq = torch.max(predict_logits, dim=-1)

        # Need to divide as heavy, and light.
        heavy_predict_seq = predict_seq[:, :self.aa_heavy_length]
        light_predict_seq = predict_seq[:, self.aa_heavy_length:]

        heavy_target_seq = true_seq[:, :self.aa_heavy_length]
        light_target_seq = true_seq[:, self.aa_heavy_length:]

        heavy_masked = h_l_masked.bool()[:, :self.aa_heavy_length]
        light_masked = h_l_masked.bool()[:, self.aa_heavy_length:]

        heavy_mutation_num = torch.sum(
            (heavy_predict_seq != heavy_target_seq) * heavy_masked,
            dim=-1
        )

        light_mutation_num = torch.sum(
            (light_predict_seq != light_target_seq) * light_masked,
            dim=-1
        )

        heavy_mutation_loss = torch.clamp(
            (
                (heavy_mutation_num-self.heavy_mutation_threshold) / self.norm_mutation
            ),
            min=0
        )
        light_mutation_loss = torch.clamp(
            (
                (light_mutation_num-self.light_mutation_threshold) / self.norm_mutation
            ) ** 2,
            min=0
        )
        return torch.mean(heavy_mutation_loss), torch.mean(light_mutation_loss)


    
    def batch_get_each_seq_score(self, abnativ_out, recover_resi_mask, all_seq, model_type='VH'):
        scores = get_abnativ_nativeness_scores(abnativ_out, recover_resi_mask, model_type, all_seq=all_seq)
        return scores
    
    def get_corresponding_aho_idx_of_infilling_seq(self, 
                                                    infilling_seq,  
                                                    imgt_h_l_seq_tensor,
                                                    imgt_h_l_selected_mask,
                                                    limit_imgt_mask,
                                                    aho_mask,
                                                    aho_h_l_seq
                                                ):
        new_infilling_seq = torch.zeros_like(infilling_seq).detach().clone()
        # May need to detach the gradient.
        imgt_h_l_seq_tensor = imgt_h_l_seq_tensor.clone().detach()
        aho_h_l_seq = aho_h_l_seq.clone().detach()

        imgt_h_l_seq_tensor[imgt_h_l_selected_mask.bool()] = new_infilling_seq

        limit_imgt_h_l_seq_tensor = torch.cat(
            (
                imgt_h_l_seq_tensor[:, :self.imgt_heavy_tail, :],
                imgt_h_l_seq_tensor[:, self.imgt_heavy_length:self.imgt_light_tail, :]
            ),
                dim=1
        )
        no_pad_infilling_seq = limit_imgt_h_l_seq_tensor[limit_imgt_mask]

        aho_h_l_seq[aho_mask] = no_pad_infilling_seq.type_as(aho_h_l_seq)
        new_idx_for_aho = aho_h_l_seq.sum(dim=-1) == 0
        return new_idx_for_aho.clone().detach()


    def trans_align_scheme(self, imgt_h_l_seq, infilling_seq, imgt_mask, aho_h_l_seq, aho_mask, imgt_h_l_selected_mask):
        """
        Need to replace the changed residues of the abnativ_seq, and get a new abnativ_seq for eval.
        :param infilling_seq: [num_mask, imgt_length]
        :param imgt_mask: [batch, imgt_length]
        :param abnativ_seq: [batch, aho_length, resi_type]
        :param aho_mask: [batch, aho_length]
        :return: new_abnativ_seq [batch, aho_length]
        """
        # Modify the vhhseq pad index to match the abnativ pad index.
        imgt_h_l_seq = imgt_h_l_seq.clone().detach()
        number_change_mask = imgt_h_l_seq == self.Tokenizer.idx_pad  # 21
        imgt_h_l_seq[number_change_mask] = self.imgt_true_max_index  # 20
        imgt_h_l_seq_tensor = F.one_hot(imgt_h_l_seq.long(), num_classes=self.abnativ_class)
        
        # Need to complete the seq tensor, and make sure the dim is same as the dim of vhh_seq_tensor
        infilling_seq = torch.cat([infilling_seq, torch.zeros(infilling_seq.size(0), 1).to(infilling_seq.device)], dim=-1)
        
        # Limit beacuse the abnativ add the tail which makes us 
        # need to limit the mask, and do not consider the tail.
        limit_imgt_mask = torch.cat(
                            (
                        imgt_mask[:, :self.imgt_heavy_tail],
                        imgt_mask[:, self.imgt_heavy_length:self.imgt_light_tail]
                    ),
                        dim=-1
                    )
        # Here, we load the infilling seq to imgt tensor.
        imgt_h_l_seq_tensor = imgt_h_l_seq_tensor.type_as(infilling_seq)
        imgt_h_l_seq_tensor[imgt_h_l_selected_mask.bool()] = infilling_seq

        # Need to concat the limit heavy and light tensor. (ingore tail)
        limit_imgt_h_l_seq_tensor = torch.cat(
            (
                imgt_h_l_seq_tensor[:, :self.imgt_heavy_tail, :],
                imgt_h_l_seq_tensor[:, self.imgt_heavy_length:self.imgt_light_tail, :]
            ),
                dim=1
        )
        no_pad_infilling_seq = limit_imgt_h_l_seq_tensor[limit_imgt_mask]
        
        # Make sure the tail doen't include.
        aho_mask[:, self.aho_heavy_tail:self.aho_heavy_length] = False
        aho_mask[:, self.aho_light_tail] = False

        assert aho_mask.sum() == no_pad_infilling_seq.size(0), print('During trans seq type, mask has problem.')
        aho_h_l_seq[aho_mask] = no_pad_infilling_seq.type_as(aho_h_l_seq)

        aho_h_l_infilling_resi_idx = self.get_corresponding_aho_idx_of_infilling_seq(
            infilling_seq,
            imgt_h_l_seq_tensor,
            imgt_h_l_selected_mask,
            limit_imgt_mask,
            aho_mask,
            aho_h_l_seq,
        )
        return aho_h_l_seq, aho_h_l_infilling_resi_idx


    def get_corresponding_mask(self, imgt_h_l, aho_h_l):
        imgt_h_l_mask = imgt_h_l < self.Tokenizer.idx_pad
        # Here we need to set specific postion as True,
        # the tail of heavy or light.
        # Specific heavy position. (150, 151)
        # Specific light position. (290)
        imgt_h_l_mask[:, self.imgt_heavy_tail:self.imgt_heavy_length] = True 
        imgt_h_l_mask[:, self.imgt_light_tail] = True

        aho_h_l_mask = torch.argmax(aho_h_l, dim=-1) != self.aho_pad_idx
        assert imgt_h_l_mask.sum() == aho_h_l_mask.sum(), 'Mask has problem, please debug' 
        return imgt_h_l_mask, aho_h_l_mask


    def token_seq(self, seq_tensor_batch, type='aho'):
        if type == 'aho':
            seq_indices = torch.argmax(seq_tensor_batch, dim=-1)
            seq_list = debug_aho_idx_to_seq(seq_indices.int())
            return seq_list
        elif type == 'imgt':
            # seq_indices = torch.argmax(seq_tensor_batch, dim=-1)
            seq_list = self.Tokenizer.idx2seq_pad_batch(seq_tensor_batch.int())
            return seq_list
        else:
            raise KeyError('Seq code type don not know')


    def seq_debug(self, index_list, imgt_tensor, aho_tensor):
        align_imgt = self.token_seq(imgt_tensor, 'imgt')
        align_aho = self.token_seq(aho_tensor, 'aho')
        print('imgt mask and aho mask are not equal.')
        for i in index_list:
            imgt_seq = align_imgt[i]
            aho_seq = align_aho[i]
            print(f'{i} imgt heavy seq: {imgt_seq[:self.aa_heavy_length]}')
            print(f'{i} aho  heavy seq: {aho_seq[:149]}')
            print('----------------')
        for i in index_list:
            imgt_seq = align_imgt[i]
            aho_seq = align_aho[i]
            print(f'{i} imgt light seq: {imgt_seq[self.aa_heavy_length:]}')
            print(f'{i} aho  light seq: {aho_seq[149:]}')
            print('---------------')

    def infilling_seq(self, H_L_seq, H_L_region_type, H_L_chn_type, masked_idx):
        predict_seq = self.anti_infilling(H_L_seq, H_L_region_type, H_L_chn_type)
        true_p = predict_seq[:, :, :self.imgt_true_max_index]  # avoid generating non standard AA.
        select_p = true_p[masked_idx.bool()]
        p_sample = self.gumbel_softmax(select_p)  # [*, self.imgt_true_max_index]
        return p_sample, predict_seq


    def gumbel_softmax(self, logits):
        # Sample from the normal distribution.
        uniform = torch.rand_like(logits)
        # sample gumbel
        gumbel = -torch.log(-torch.log(uniform + 1e-20) + 1e-20)
        noisy_logits = (logits + gumbel) / self.temperature
        probabilities = F.softmax(noisy_logits, dim=-1)
        hard = torch.argmax(probabilities, dim=-1)

        hard = F.one_hot(hard, num_classes=probabilities.size(-1))
        hard = hard.float()
        # forward we use the hard vector.
        # while using the probabilities during backpropagation.
        return (hard - probabilities).detach() + probabilities
    

def debug_aho_idx_to_seq(idx_mat):

    def idx2seq(idx_vec):
        aa_seq_ext = [alphabet[x] for x in idx_vec.tolist()]
        aa_seq = ''.join(aa_seq_ext)
        return aa_seq

    n_seqs = idx_mat.shape[0]
    aa_seq_list = [idx2seq(idx_mat[x]) for x in range(n_seqs)]
    return aa_seq_list