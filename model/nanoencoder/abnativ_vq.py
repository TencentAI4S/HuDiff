# (c) 2023 Sormannilab and Aubin Ramon
# 
# Vector-Quantization of the latent space in the AbNatiV model.
#
# Modified from (https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py)
# under the copyright (c) 2020 Phil Wang (MIT Licenced)
# ============================================================================


from einops import rearrange
import pandas as pd
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.cuda.amp import autocast

from .abnativ_utils import uniform_init, kmeans, sample_vectors_distributed, noop, batched_embedding, batched_sample_vectors
from .abnativ_utils import l2norm, gumbel_sample, ema_inplace, default
                           

class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 3,
        use_ddp = False,
        learnable_codebook = False,
        sample_codebook_temp = 0.
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim = True,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)

        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    @autocast(enabled = False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        shape, dtype = x.shape, x.dtype

        flatten = rearrange(x, 'h ... d -> h (...) d')
        flatten = l2norm(flatten)

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = l2norm(embed)

        dist = einsum('h n d, h c d -> h n c', flatten, embed)
        embed_ind = gumbel_sample(dist, dim = -1, temperature = self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = batched_embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(dim = 1)
            self.all_reduce_fn(bins)

            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            self.all_reduce_fn(embed_sum)

            embed_normalized = embed_sum / rearrange(bins, '... -> ... 1')
            embed_normalized = l2norm(embed_normalized)

            embed_normalized = torch.where(
                rearrange(zero_mask, '... -> ... 1'),
                embed,
                embed_normalized
            )

            ema_inplace(self.embed, embed_normalized, self.decay)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        return quantize, embed_ind


class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim,
        heads = 1,
        separate_codebook_per_head = False,
        decay = 0.8,
        eps = 1e-5,
        kmeans_init = True,
        kmeans_iters = 10,
        sync_kmeans = True,
        threshold_ema_dead_code = 3,
        commitment_weight = 1.,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_active_codes_only = False,
        orthogonal_reg_max_codes = None,
        sample_codebook_temp = 0.,
        sync_codebook = False
    ):
        super().__init__()
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.commitment_weight = commitment_weight

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        codebook_class = CosineSimCodebook

        self._codebook = codebook_class(
            dim = codebook_dim,
            num_codebooks = heads if separate_codebook_per_head else 1,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            sync_kmeans = sync_kmeans,
            decay = decay,
            eps = eps,
            threshold_ema_dead_code = threshold_ema_dead_code,
            use_ddp = sync_codebook,
            learnable_codebook = has_codebook_orthogonal_loss,
            sample_codebook_temp = sample_codebook_temp
        )

        self.codebook_size = codebook_size

    @property
    def codebook(self):
        codebook = self._codebook.embed
        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, '1 ... -> ...')

    def forward(self, x,):
        shape, device, heads, is_multiheaded, codebook_size = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size

        x = self.project_in(x)

        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h = heads)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        
        detached_inputs = x.detach()
        loss = F.mse_loss(quantize, detached_inputs, reduction='none')
        loss_pbe = torch.mean(loss, dim=(1,2)) # (batch_size)

        if self.commitment_weight > 0:
            detached_quantize = quantize.detach()
            commit_loss = F.mse_loss(detached_quantize, x, reduction='none')

            loss_pbe = loss_pbe + torch.mean(commit_loss * self.commitment_weight, dim=(1,2)) # (batch_size)

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h = heads)
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h = heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h = heads)
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h = heads)

        quantize_latent = quantize.detach().clone()
        quantize = self.project_out(quantize)

        avg_probs = torch.mean(F.one_hot(embed_ind, self.codebook_size).type(torch.float32).view((-1, self.codebook_size)), 0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
        'quantize_projected_in': x, # (batch_size, l_r, codebook_dim)
        'quantize_latent': quantize_latent, # (batch_size, l_r, codebook_dim)
        'quantize_projected_out': quantize, # (batch_size, l_r, dim)
        'loss_vq_commit_pbe': loss_pbe, # (batch_size)
        'perplexity': perplexity, # (batch_size)
        'encoding_indices': embed_ind # (batch_size, l_r)
    } 

