import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sequence_models.layers import PositionFeedForward, DoubleEmbedding
from sequence_models.convolutional import ByteNetBlock, MaskedConv1d
from ..encoder.cross_attention import TransformerNet, SelfAttNet, precompute_freqs_cis
# from .modules import TransformerLayer, ESM1LayerNorm

# Framework load.
import math

from .abnativ_scoring import get_abnativ_nativeness_scores


# Abnativ res list.
alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
own_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

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


class NanoByteNetTime(nn.Module):
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
        layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
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
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout)
        return e


class NanoSideEmbedder(nn.Module):
    def __init__(self, n_side, s_embedding, d_side, aa_h_length=152):
        super().__init__()
        self.side_embeddinng = nn.Embedding(n_side, s_embedding)
        self.side_mlp = nn.Sequential(
            nn.Linear(s_embedding, d_side),
            nn.LayerNorm(d_side),
            nn.ReLU(),
            nn.Linear(d_side, d_side),
        )
        self.aa_h_length = aa_h_length

    def forward(self, side, mask=None):
        emb_side = self.side_embeddinng(side.view(-1, 1))
        emb_side = self.side_mlp(emb_side)
        emb_side = emb_side.repeat(1, self.aa_h_length, 1)
        return emb_side


class NanoRegionEmbedder(nn.Module):

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


class NanoPosEmbedder(nn.Module):
    """
    This position embedding method is PE encoding.
    """
    def __init__(self, p_emb, max_len):
        super().__init__()
        self.pos_embedding = PositionalEncoding(p_emb, max_len)
        self.pos_lin = MLP(n_embd=p_emb)

    def forward(self, region_emb):
        x = self.pos_embedding(region_emb)
        pos_emb = self.pos_lin(x)
        x = x + pos_emb
        return x


class ByteNetLMTime(nn.Module):

    def __init__(self):
        super().__init__()
        pass


class NanoConv(nn.Module):

    def __init__(self, d_model, n_layers, kernel_size, r, rank=None,
                causal=False, dropout=0.0, slim=True, activation='gelu', timesteps=None,
                 aa_h_length=152, aa_l_length=139):
        super().__init__()

        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.aa_h_length = aa_h_length
        self.aa_l_length = aa_l_length
        self.dropout = dropout

    def _conv(self, s):
        for layer in self.layers:
            s = layer(s)
            if self.dropout > 0.0:
                s = F.dropout(s)
        return s

    def forward(self, s, batch=None, mask=None):
        s = self._conv(s)
        return s




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


class NanoAntiTFNet(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_model, n_encoder_layers, aa_kernel_size, r,
                 n_region, r_embedding, r_model,
                 n_pos_model, max_len,
                 sum_d_model, dual_layers,
                 att_model, dim_feedforward, nhead, cs_layers,
                 rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, slim=True, activation='relu',
                 down_embed=False, timesteps=None):
        super().__init__()

        self.aa_encoder = NanoByteNetTime(n_tokens, d_embedding, d_model, n_encoder_layers, aa_kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs,
                                timesteps=timesteps)
        self.region_encoder = NanoRegionEmbedder(n_region, r_embedding, r_model)
        self.pos_encoder = NanoPosEmbedder(n_pos_model, max_len)
        self.nano_conv_block = NanoConv(sum_d_model, dual_layers, aa_kernel_size, r, dropout=dropout)
        self.self_at = SelfAttNet(sum_d_model, att_model, dim_feedforward, nhead, rolength=max_len, num_cross_layers=cs_layers)
        self.last_norm = nn.LayerNorm(sum_d_model)
        self.decoder = nn.Linear(sum_d_model, n_tokens)

    def _encoder(self, aa_seq, region_type, chn_type):
        emb = self.aa_encoder(aa_seq)
        region_emb = self.region_encoder(region_type)
        pos_emb = self.pos_encoder(region_emb)
        emb = emb + pos_emb
        feature = torch.cat((emb, pos_emb), dim=-1)
        return feature  

    def _att(self, h):
        h = self.self_at(h)
        return h

    def forward(self, H_seq, H_region_type, H_chn_type):
        """

        :param H_L_seq: (Batch, length);
        :param H_L_pos_type: (Batch, length); distinguish the different region of Chain.
        :param H_L_chn_type: (Batch); gene
        :param H_L_batch: (Batch); distinguish the type of Chain.
        :param H_L_mask: None
        :return: (Batch, length, feature)
        """
        h_feature = self._encoder(
            aa_seq=H_seq.int(),
            region_type=H_region_type.int(),
            chn_type=H_chn_type
        )
        h = self.nano_conv_block(h_feature)
        h = self._att(h)
        h = self.decoder(self.last_norm(h))
        return h


class NanoInfillingFramework(nn.Module):

    def __init__(self,
                 config,
                 pretrained_model_list,
                 tokenizer,
                 ):
        super().__init__()
        print('Initializing the abnativ score model and infilling model!')
        self.eval_abnativ_model = pretrained_model_list['abnativ']
        self.infilling_pretrain = pretrained_model_list['infilling']
        self.target_infilling_pretrain = pretrained_model_list['target_infilling']
        self.vhh_nativeness = config.vhh_nativeness
        if self.vhh_nativeness:
            self.vhh_eval_abnativ_model = pretrained_model_list['vhh_abnativ']
            self.vhh_all_seq = config.vhh_all_seq
        self.human_threshold = config.human_threshold
        self.human_all_seq = config.human_all_seq                 # bool
        self.equal_weight = config.equal_weight                   # bool
        
        self.tokenizer = tokenizer
        self.loss_type = config.loss_type
        self.temperature = config.temperature

        # Fixed parameters.  
        self.best_vh_threshold = 0.988047
        self.aho_mask_length = 147
        self.abnativ_class = 21
        self.imgt_mask_length = 150
        self.imgt_true_max_index = 20


    def forward(self, vhh_src_seq, vhh_src_mask, vhh_ref_seq, vhh_region, vhh_ab_input):
        assert vhh_src_seq.size(0) == vhh_ab_input.size(0), print('Make sure input size is equal!')
        if self.vhh_nativeness:
            vhh_ab_old_input = vhh_ab_input.clone().detach()
        vhh_infilling_seq, pred_seq = self.infilling_seq(vhh_src_seq, vhh_region, vhh_src_mask)
        batch_imgt_mask, batch_aho_mask = self.get_aho_and_imgt_mask(vhh_ab_input, vhh_ref_seq)
        infilling_vhh_ab_input, infilling_vhh_ab_mask = self.trans_different_scheme(
                                                                vhh_ref_seq,
                                                                vhh_infilling_seq,
                                                                batch_imgt_mask,
                                                                vhh_ab_input,
                                                                batch_aho_mask,
                                                                vhh_src_mask
                                                            )
        new_output_abnativ = self.eval_abnativ_model(infilling_vhh_ab_input)
        humanness_score = self.batch_get_each_residue_score(new_output_abnativ,
                                                            infilling_vhh_ab_mask,
                                                            self.human_all_seq,
                                                            model_type='VH',
                                                            only_seq=True)

        # Here we consider the abnativ vhh nativeness.
        if self.vhh_nativeness:
            old_output_vhh_abnativ = self.vhh_eval_abnativ_model(vhh_ab_old_input)
            old_seq_vhh_score = self.batch_get_each_residue_score(old_output_vhh_abnativ,
                                                                  infilling_vhh_ab_mask,
                                                                  self.vhh_all_seq,
                                                                  model_type='VHH',
                                                                  only_seq=True)
            new_output_vhh_abnativ = self.vhh_eval_abnativ_model(infilling_vhh_ab_input)
            new_seq_vhh_score = self.batch_get_each_residue_score(new_output_vhh_abnativ,
                                                                  infilling_vhh_ab_mask,
                                                                  self.vhh_all_seq,
                                                                  model_type='VHH',
                                                                  only_seq=True)


        if self.loss_type == 'mse_loss':
            vh_loss = F.mse_loss(humanness_score, torch.ones_like(humanness_score) * self.human_threshold)
        elif self.loss_type == 'smooth_loss':
            vh_loss = F.smooth_l1_loss(humanness_score, torch.ones_like(humanness_score) * self.human_threshold)
        elif self.loss_type == 'l1_loss':
            vh_loss = F.l1_loss(humanness_score, torch.ones_like(humanness_score) * self.human_threshold)
        else:
            raise KeyError('Loss type do not know.')

        if self.vhh_nativeness:
            delta_vhh_change = F.mse_loss(new_seq_vhh_score, old_seq_vhh_score)
            if self.equal_weight:
                if delta_vhh_change < vh_loss:
                    # loss equal contribution to the gradient.
                    delta_loss = delta_vhh_change / (delta_vhh_change / vh_loss).detach()
                else:
                    delta_loss = delta_vhh_change
                loss = vh_loss + delta_loss
            else:
                loss = vh_loss + delta_vhh_change
        else:
            loss = vh_loss
        if self.vhh_nativeness:
            return loss, pred_seq, vh_loss, delta_vhh_change
        else:
            return loss, pred_seq, vh_loss, 0

    def get_aho_and_imgt_mask(self, vhh_ab_input, vhh_ref_seq):
        vhh_aho_mask = torch.argmax(vhh_ab_input, dim=-1) != 20
        vhh_aho_mask[:, self.aho_mask_length:] = True

        # fake_residues_score = torch.ones_like(vhh_ref_seq).to(vhh_ref_seq.device)
        vhh_imgt_mask = vhh_ref_seq < self.imgt_true_max_index
        vhh_imgt_mask[:, self.imgt_mask_length: ] = True
        assert vhh_imgt_mask.sum() == vhh_aho_mask.sum(), print('vhh_imgt_mask not equal vhh_aho_mask')
        return vhh_imgt_mask, vhh_aho_mask

    def batch_get_each_residue_score(self, abnativ_out, recover_resi_mask, all_seq, model_type='VH', only_seq=False):
        scores = get_abnativ_nativeness_scores(abnativ_out, recover_resi_mask, model_type, all_seq=all_seq)
        if not only_seq:
            aho_mask = torch.argmax(abnativ_out['inputs'], dim=-1) != 20
            aho_mask[:, self.aho_mask_length:] = True
            res_scores = torch.exp(-abnativ_out['recon_error_pposi'])
            return scores, res_scores, aho_mask
        else:
            return scores

    def token_seq(self, seq_tensor_batch, type='aho'):
        if type == 'aho':
            seq_indices = torch.argmax(seq_tensor_batch, dim=-1)
            seq_list = debug_aho_idx_to_seq(seq_indices.int())
            return seq_list
        elif type == 'imgt':
            # seq_indices = torch.argmax(seq_tensor_batch, dim=-1)
            seq_list = self.tokenizer.idx2seq_pad_batch(seq_tensor_batch.int())
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
            print(f'{i} imgt seq: {imgt_seq}')
            print(f'{i} aho  seq: {aho_seq}')

    def mask_residues(self, vhh_seq, residues_score, vhh_aho_mask, vhh_imgt_cdr_mask, vhh_ab_input):
        fake_residues_score = torch.ones_like(vhh_seq).to(vhh_seq.device)
        vhh_imgt_mask = torch.ones_like(vhh_seq).bool()
        vhh_imgt_mask[:, :self.imgt_mask_length] = vhh_seq[:, :self.imgt_mask_length] < self.imgt_true_max_index
        seq_prob_index_list = torch.where(vhh_imgt_mask.sum(dim=1) != vhh_aho_mask.sum(dim=1))[0].cpu().numpy()
        # self.seq_debug(seq_prob_index_list, vhh_seq, vhh_ab_input)
        assert vhh_imgt_mask.sum() == vhh_aho_mask.sum(), self.seq_debug(seq_prob_index_list, vhh_seq, vhh_ab_input)
        true_seq_residues_score = residues_score[vhh_aho_mask]
        fake_residues_score[vhh_imgt_mask] = true_seq_residues_score
        score_to_mask_idx = fake_residues_score < self.best_vh_threshold

        # Here need to consider the cdr will not change.
        # Because those score contain the cdr residues score.
        # We need to mask it, and make sure it is not change.
        need_to_mask_idx = score_to_mask_idx * ~vhh_imgt_cdr_mask.bool()
        assert score_to_mask_idx.sum() > need_to_mask_idx.sum(), print('CDR mask and score mask has problem.')
        vhh_seq[need_to_mask_idx] = self.tokenizer.idx_msk
        return vhh_seq, need_to_mask_idx, vhh_imgt_mask

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

    def infilling_seq(self, masked_seq, vhh_region, masked_idx):
        pred_seq = self.infilling_pretrain(masked_seq, vhh_region, None)
        true_p = pred_seq[:, :, :self.imgt_true_max_index]  # avoid generating non standard AA.
        select_p = true_p[masked_idx.bool()]
        p_sample = self.gumbel_softmax(select_p)
        return p_sample, pred_seq

    def get_corresponding_aho_idx_of_infilling_seq(self, infilling_seq, vhh_seq_tensor, vhh_select_mask, limit_imgt_mask, aho_mask, abnativ_seq):
        new_infilling_seq = torch.zeros_like(infilling_seq).detach().clone()
        # May need to detach the gradient.
        vhh_seq_tensor = vhh_seq_tensor.clone().detach()
        abnativ_seq = abnativ_seq.clone().detach()

        vhh_seq_tensor[vhh_select_mask.bool()] = new_infilling_seq
        no_pad_infilling_seq = vhh_seq_tensor[:, :self.imgt_mask_length, :][limit_imgt_mask]
        abnativ_seq[aho_mask] = no_pad_infilling_seq.type_as(abnativ_seq)
        new_idx_for_aho = abnativ_seq.sum(dim=-1) == 0
        return new_idx_for_aho

    def trans_different_scheme(self, vhh_seq, infilling_seq, imgt_mask, abnativ_seq, aho_mask, vhh_selected_mask):
        """
        Need to replace the changed residues of the abnativ_seq, and get a new abnativ_seq for eval.
        :param infilling_seq: [batch, imgt_length]
        :param imgt_mask: [batch, imgt_length]
        :param abnativ_seq: [batch, aho_length, resi_type]
        :param aho_mask: [batch, aho_length]
        :return: new_abnativ_seq [batch, aho_length]
        """
        # Modify the vhhseq pad index to match the abnativ pad index.
        vhh_seq = vhh_seq.clone().detach()
        number_change_mask = vhh_seq == self.tokenizer.idx_pad  # 21
        vhh_seq[number_change_mask] = self.imgt_true_max_index  # 20
        vhh_seq_tensor = F.one_hot(vhh_seq.long(), num_classes=self.abnativ_class)

        # Need to complete the seq tensor, and make sure the dim is same as the dim of vhh_seq_tensor
        infilling_seq = torch.cat([infilling_seq, torch.zeros(infilling_seq.size(0), 1).to(infilling_seq.device)], dim=-1)

        limit_imgt_mask = imgt_mask[:, :self.imgt_mask_length]
        vhh_seq_tensor = vhh_seq_tensor.type_as(infilling_seq)
        vhh_seq_tensor[vhh_selected_mask.bool()] = infilling_seq
        no_pad_infilling_seq = vhh_seq_tensor[:, :self.imgt_mask_length, :][limit_imgt_mask]
        aho_mask[:, self.aho_mask_length:] = False
        assert aho_mask.sum() == no_pad_infilling_seq.size(0), print('During trans seq type, mask has problem.')
        abnativ_seq[aho_mask] = no_pad_infilling_seq.type_as(abnativ_seq)
        new_abnativ_seq = abnativ_seq
        aho_infilling_resi_idx = self.get_corresponding_aho_idx_of_infilling_seq(
            infilling_seq,
            vhh_seq_tensor,
            vhh_selected_mask,
            limit_imgt_mask,
            aho_mask,
            abnativ_seq,
        )
        return new_abnativ_seq, aho_infilling_resi_idx

def debug_aho_idx_to_seq(idx_mat):

    def idx2seq(idx_vec):
        aa_seq_ext = [alphabet[x] for x in idx_vec.tolist()]
        aa_seq = ''.join(aa_seq_ext)
        return aa_seq

    n_seqs = idx_mat.shape[0]
    aa_seq_list = [idx2seq(idx_mat[x]) for x in range(n_seqs)]
    return aa_seq_list