# (c) 2023 Sormannilab and Aubin Ramon
#
# AbNatiV model, Pytorch version
#
# ============================================================================

from typing import Tuple
import math

from .abnativ_vq import VectorQuantize
from .abnativ_utils import find_optimal_cnn1d_padding, find_out_padding_cnn1d_transpose

import torch
from torch import nn
from torch.nn import functional as F

# import pytorch_lightning as pl
from einops.layers.torch import Rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, d_embedding, max_len):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embedding, 2) * (-math.log(10000.0) / d_embedding))
        pe = torch.zeros(max_len, d_embedding)

        # apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)

        # apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, input_seq_len, d_embedding]
        """
        x = x + self.pe[:x.size(1)]
        return x


class MHAEncoderBlock(nn.Module):
    def __init__(self, d_embedding, num_heads, d_ff, dropout):
        super(MHAEncoderBlock, self).__init__()

        self.self_MHA = torch.nn.MultiheadAttention(d_embedding, num_heads, batch_first=True)

        self.MLperceptron = nn.Sequential(
            nn.Linear(d_embedding, d_ff),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_embedding))

        self.layernorm1 = nn.LayerNorm(d_embedding, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_embedding, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: Tensor, shape [batch_size, input_seq_len, d_embedding]
        """
        # Attention
        attn_output, attn_output_weights = self.self_MHA(x, x, x)  # (batch_size, input_seq_len, d_embedding)
        x = x + self.dropout(attn_output)
        x = self.layernorm1(x)

        # MLP
        linear_output = self.MLperceptron(x)
        x = x + self.dropout(linear_output)
        x = self.layernorm2(x)  # (batch_size, input_seq_len, d_embedding) + residual

        return x, attn_output_weights


class Encoder(nn.Module):
    def __init__(self, d_embedding, kernel, stride, num_heads, num_mha_layers, d_ff,
                 length_seq, alphabet_size, dropout=0):
        super(Encoder, self).__init__()

        # CNN1d embedding
        self.l_red, self.padding = find_optimal_cnn1d_padding(L_in=length_seq, K=kernel, S=stride)
        self.cnn_embedding = nn.Sequential(Rearrange('b l r -> b r l'),
                                           nn.Conv1d(alphabet_size, d_embedding, kernel_size=kernel, stride=stride,
                                                     padding=self.padding),
                                           Rearrange('b r l -> b l r'))

        # Positional encoding
        self.en_pos_encoding = PositionalEncoding(d_embedding, max_len=self.l_red)
        self.en_dropout = nn.Dropout(dropout)

        # MHA blocks
        self.en_MHA_blocks = nn.ModuleList([MHAEncoderBlock(d_embedding, num_heads, d_ff, dropout)
                                            for _ in range(num_mha_layers)])

    def forward(self, x) -> torch.Tensor:
        """
        Args:
          x: Tensor, shape [batch_size, input_seq_len, alphabet_size]
        """
        # CNN1d Embedding
        h = self.cnn_embedding(x)  # (batch_size, l_red, d_embedding)

        # Positional encoding
        h = self.en_pos_encoding(h)
        h = self.en_dropout(h)

        #  MHA blocks
        for i, l in enumerate(self.en_MHA_blocks):
            h, attn_enc_weights = self.en_MHA_blocks[i](h)  # (batch_size, l_red, d_embedding)

        return h


class Decoder(nn.Module):
    def __init__(self, d_embedding, kernel, stride, num_heads, num_mha_layers, d_ff,
                 length_seq, alphabet_size, dropout=0):
        super(Decoder, self).__init__()

        # Positional encoding
        self.l_red, self.padding = find_optimal_cnn1d_padding(L_in=length_seq, K=kernel, S=stride)
        self.de_pos_encoding = PositionalEncoding(d_embedding, max_len=self.l_red)
        self.de_dropout = nn.Dropout(dropout)

        # MHA blocks
        self.de_MHA_blocks = nn.ModuleList([MHAEncoderBlock(d_embedding, num_heads, d_ff, dropout)
                                            for _ in range(num_mha_layers)])

        # Dense reconstruction
        self.dense_to_alphabet = nn.Linear(d_embedding, alphabet_size)
        self.dense_reconstruction = nn.Linear(alphabet_size * self.l_red, length_seq * alphabet_size)

        # CNN1d reconstruction
        self.out_pad = find_out_padding_cnn1d_transpose(L_obj=length_seq, L_in=self.l_red, K=kernel, S=stride,
                                                        P=self.padding)
        self.cnn_reconstruction = nn.Sequential(Rearrange('b l r -> b r l'),
                                                nn.ConvTranspose1d(d_embedding, alphabet_size, kernel_size=kernel,
                                                                   stride=stride,
                                                                   padding=self.padding, output_padding=self.out_pad),
                                                Rearrange('b r l -> b l r'))

    def forward(self, q) -> torch.Tensor:
        """
        Args:
          q: Tensor, shape [batch_size, l_red, d_embedding]
        """
        # Positional encoding
        z = self.de_pos_encoding(q)
        z = self.de_dropout(z)

        #  MHA blocks
        for i, l in enumerate(self.de_MHA_blocks):
            z, attn_dec_weights = self.de_MHA_blocks[i](z)  # (batch_size, l_red, d_embedding)

        # CNN reconstruction
        z = self.cnn_reconstruction(z)  # (batch_size, input_seq_len, alphabet_size)
        z_recon = F.softmax(z, dim=-1)

        return z_recon


class AbNatiV_Model(nn.Module):
    def __init__(self, hparams: dict):
        super(AbNatiV_Model, self).__init__()

        self.encoder = Encoder(hparams['d_embedding'], hparams['kernel'], hparams['stride'], hparams['num_heads'],
                               hparams['num_mha_layers'], hparams['d_ff'], hparams['length_seq'],
                               hparams['alphabet_size'], dropout=hparams['drop'])

        self.decoder = Decoder(hparams['d_embedding'], hparams['kernel'], hparams['stride'], hparams['num_heads'],
                               hparams['num_mha_layers'], hparams['d_ff'], hparams['length_seq'],
                               hparams['alphabet_size'], dropout=hparams['drop'])

        self.vqvae = VectorQuantize(
            dim=hparams['d_embedding'],
            codebook_size=hparams['num_embeddings'],
            codebook_dim=hparams['embedding_dim_code_book'],
            decay=hparams['decay'],
            kmeans_init=True,
            commitment_weight=hparams['commitment_cost']
        )

        self.learning_rate = hparams['learning_rate']
        # self.save_hyperparameters()

    def forward(self, data) -> dict:
        inputs = data
        m_inputs = data.clone().detach()

        x = self.encoder(m_inputs)
        vq_outputs = self.vqvae(x)
        x_recon = self.decoder(vq_outputs['quantize_projected_out'])

        # Loss computing
        recon_error_pres_pposi = F.mse_loss(x_recon, inputs, reduction='none')
        recon_error_pposi = torch.mean(recon_error_pres_pposi, dim=-1)
        recon_error_pbe = torch.mean(recon_error_pposi, dim=1)

        loss_pbe = torch.add(recon_error_pbe, vq_outputs['loss_vq_commit_pbe'])

        return {
            'inputs': inputs,  # (batch_size, input_seq_len, alphabet_size)
            'x_recon': x_recon,  # (batch_size, input_seq_len, alphabet_size)
            'recon_error_pres_pposi': recon_error_pres_pposi,  # (batch_size, input_seq_len, alphabet_size)
            'recon_error_pposi': recon_error_pposi,  # (batch_size, input_seq_len)
            'recon_error_pbe': recon_error_pbe,  # (batch_size)
            'loss_pbe': loss_pbe,  # (batch_size)
            **vq_outputs
        }

    def configure_optimizers(self):
        optim_groups = list(self.encoder.parameters()) + \
                       list(self.decoder.parameters()) + \
                       list(self.vqvae.parameters())

        return torch.optim.AdamW(optim_groups, lr=self.learning_rate)

    def training_step(self, batch, batch_idx) -> torch.float32:
        vqvae_output = self(batch)

        loss_vqvae = torch.mean(vqvae_output['loss_pbe'])
        self.log("train_loss_vqvae", loss_vqvae, on_step=True, prog_bar=True, logger=True)

        loss_vq_commit = torch.mean(vqvae_output['loss_vq_commit_pbe'])
        self.log("train_loss_vq_commit", loss_vq_commit, on_step=True, prog_bar=True, logger=True)

        nmse_accuracy = torch.mean(vqvae_output['recon_error_pbe'])
        self.log("train_loss_nmse_recons", nmse_accuracy, on_step=True, prog_bar=True, logger=True)

        perplexity = vqvae_output['perplexity']
        self.log("train_perplexity", perplexity, on_step=True, prog_bar=True, logger=True)

        return loss_vqvae

    def validation_step(self, batch, batch_idx) -> dict:
        model_output = self(batch)
        return {'val_loss': torch.mean(model_output['loss_pbe']), 'model_output': model_output}

    def validation_epoch_end(self, outputs) -> dict:
        val_losses = torch.Tensor([out['val_loss'] for out in outputs])
        total_val_loss = torch.mean(val_losses)
        self.log('val_loss', total_val_loss, on_epoch=True, logger=True)

        val_accuracies = torch.Tensor([torch.mean(out['model_output']['recon_error_pbe']) for out in outputs])
        total_val_accuracy = torch.mean(val_accuracies)
        self.log('val_nmse_accuracy', total_val_accuracy, on_epoch=True, logger=True)

        val_perplexities = torch.Tensor([out['model_output']['perplexity'] for out in outputs])
        total_val_perplexity = torch.mean(val_perplexities)
        self.log('val_perplexity', total_val_perplexity, on_epoch=True, logger=True)

        return {'val_loss': total_val_loss, 'val_nmse_accuracy': total_val_accuracy,
                'val_perplexity': total_val_perplexity}


