import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot, softmax
from torch.nn.functional import cross_entropy
from sklearn.metrics import roc_auc_score

from .tokenizer import Tokenizer


class MaskedAccuracy(object):
    """Masked accuracy.

    Inputs:
        pred (N, L, C)
        tgt (N, L)
        mask (N, L)
    """

    def __call__(self, pred, tgt, mask):
        _, p = torch.max(pred, -1)
        masked_tgt = torch.masked_select(tgt, mask.bool())
        p = torch.masked_select(p, mask.bool())
        return torch.mean((p == masked_tgt).float()), 0


class OasMaskedSplitCrossEntropyLoss(CrossEntropyLoss):
    """Masked cross-entropy loss for sequences.
    Evaluates the cross-entropy loss at specified locations in a sequence
    When reweight = True, reweights CE according to Hoogeboom et al.;
    reweight term = 1/(D-t+1)
    Shape:
        Inputs:
            - pred: (N, L, n_tokens)
            - tgt: (N, L)
            - mask: (N, L) boolean
            - timestep (N, L) output from OAMaskCollater
            - input mask (N, L)
            - weight: (C, ): class weights for nn.CrossEntropyLoss

    Returns
        ce_losses
        nll_losses
    """
    def __init__(self, weight=None, reduction='none', reweight=True, aa_h_lenghth=152, l_weight=1, tokenizer=Tokenizer()):
        super().__init__(weight=weight, reduction=reduction)
        self.reweight=reweight
        self.tokenizer = tokenizer
        self.aa_h_length = aa_h_lenghth
        self.l_loss_weight = l_weight

    def forward(self, H_L_pred, H_L_tgt, H_L_mask, H_L_cdr_mask, H_L_timesteps):
        # Make sure we have that empty last dimension
        if len(H_L_mask.shape) == len(H_L_pred.shape) - 1:
            H_L_mask = H_L_mask.unsqueeze(-1)

        # Need to split for loss.
        batch_size = H_L_pred.size(0) // 2
        H_pred = H_L_pred[:, :self.aa_h_length, :]
        L_pred = H_L_pred[:, self.aa_h_length:, :]

        H_tgt = H_L_tgt[:, :self.aa_h_length]
        L_tgt = H_L_tgt[:, self.aa_h_length:]

        H_mask = H_L_mask[:, :self.aa_h_length, :]
        L_mask = H_L_mask[:, self.aa_h_length:, :]

        H_cdr_mask = H_L_cdr_mask[:, :self.aa_h_length]
        L_cdr_mask = H_L_cdr_mask[:, self.aa_h_length:]

        H_timesteps = H_L_timesteps[:, 0]
        L_timesteps = H_L_timesteps[:, 1]
        H_L_timesteps = H_timesteps + L_timesteps


        # Make sure mask is boolean
        H_mask = H_mask.bool()

        H_mask_tokens = H_mask.sum()  # masked tokens
        H_cdr_mask_tokens = H_cdr_mask.sum().int()
        # H_nonpad_tokens, L_nonpad_tokens = H_input_mask.sum(dim=1), L_input_mask.sum(dim=1) # nonpad tokens

        # Cal H loss.
        H_p = torch.masked_select(H_pred, H_mask).view(H_mask_tokens, -1)  # [T x K] predictions for each mask char
        H_t = torch.masked_select(H_tgt, H_mask.squeeze())  # [ T ] true mask char
        H_loss = super().forward(H_p, H_t.long()) # [ T ] loss per mask char
        H_nll_losses = H_loss.mean()

        # Cal H cdr loss.
        H_cdr_mask = H_cdr_mask.bool()
        H_cdr_p = torch.masked_select(H_pred, H_cdr_mask.unsqueeze(-1)).view(H_cdr_mask_tokens, -1)
        H_cdr_t = torch.masked_select(H_tgt, H_cdr_mask)
        H_cdr_loss = super().forward(H_cdr_p, H_cdr_t.long())
        H_cdr_losses = H_cdr_loss.mean()


        L_mask = L_mask.bool()

        L_mask_tokens = L_mask.sum()  # masked tokens
        L_cdr_mask_tokens = L_cdr_mask.sum().int()
        # Cal L loss.
        L_p = torch.masked_select(L_pred, L_mask).view(L_mask_tokens, -1)  # [T x K] predictions for each mask char
        L_t = torch.masked_select(L_tgt, L_mask.squeeze())  # [ T ] true mask char
        L_loss = super().forward(L_p, L_t.long()) # [ T ] loss per mask char
        L_nll_losses = L_loss.mean()

        # Cal L cdr loss.
        L_cdr_mask = L_cdr_mask.bool()
        L_cdr_p = torch.masked_select(L_pred, L_cdr_mask.unsqueeze(-1)).view(L_cdr_mask_tokens, -1)
        L_cdr_t = torch.masked_select(L_tgt, L_cdr_mask)
        L_cdr_loss = super().forward(L_cdr_p, L_cdr_t.long())
        L_cdr_losses = L_cdr_loss.mean() * self.l_loss_weight

        if self.reweight: # Uses Hoogeboom OARDM reweighting term
            # For H
            H_rwt_term = 1. / H_L_timesteps

            no_pad_number = torch.tensor([H_pred.size(1)]).repeat(H_pred.size(0)).to(H_rwt_term.device)
            H_rwt_term = H_rwt_term.repeat_interleave(H_timesteps)
            H_n_tokens = no_pad_number.repeat_interleave(H_timesteps)
            H_ce_loss = H_n_tokens * H_rwt_term * H_loss
            H_ce_losses = H_ce_loss.mean()  # reduce mean

            # For L
            L_rwt_term = 1. / H_L_timesteps

            no_pad_number = torch.tensor([L_pred.size(1)]).repeat(L_pred.size(0)).to(L_rwt_term.device)
            L_rwt_term = L_rwt_term.repeat_interleave(L_timesteps)
            L_n_tokens = no_pad_number.repeat_interleave(L_timesteps)
            L_ce_loss = L_n_tokens * L_rwt_term * L_loss
            L_ce_losses = L_ce_loss.mean() * self.l_loss_weight # reduce mean

        else:
            H_ce_losses = H_nll_losses
            L_ce_losses = L_nll_losses
        return H_ce_losses, H_nll_losses.to(torch.float64), H_cdr_losses, L_ce_losses, L_nll_losses.to(torch.float64), L_cdr_losses


class OasMaskedCrossEntropyLoss(CrossEntropyLoss):
    """Masked cross-entropy loss for sequences.
    Evaluates the cross-entropy loss at specified locations in a sequence
    When reweight = True, reweights CE according to Hoogeboom et al.;
    reweight term = 1/(D-t+1)
    Shape:
        Inputs:
            - pred: (N, L, n_tokens)
            - tgt: (N, L)
            - mask: (N, L) boolean
            - timestep (N, L) output from OAMaskCollater
            - input mask (N, L)
            - weight: (C, ): class weights for nn.CrossEntropyLoss

    Returns
        ce_losses
        nll_losses
    """
    def __init__(self, weight=None, reduction='none', reweight=True, tokenizer=Tokenizer()):
        self.reweight=reweight
        self.tokenizer = tokenizer
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, H_L_pred, H_L_tgt, H_L_mask, H_L_cdr_mask, H_L_timesteps):
        # Make sure we have that empty last dimension
        if len(H_L_mask.shape) == len(H_L_pred.shape) - 1:
            H_L_mask = H_L_mask.unsqueeze(-1)



        # Make sure mask is boolean
        H_L_mask = H_L_mask.bool()

        H_L_mask_tokens = H_L_mask.sum()  # masked tokens
        H_L_cdr_mask_tokens = H_L_cdr_mask.sum().int()
        # H_nonpad_tokens, L_nonpad_tokens = H_input_mask.sum(dim=1), L_input_mask.sum(dim=1) # nonpad tokens

        # Cal H & L loss.
        H_L_p = torch.masked_select(H_L_pred, H_L_mask).view(H_L_mask_tokens, -1)  # [T x K] predictions for each mask char
        H_L_t = torch.masked_select(H_L_tgt, H_L_mask.squeeze())  # [ T ] true mask char
        H_L_loss = super().forward(H_L_p, H_L_t.long()) # [ T ] loss per mask char
        H_L_nll_losses = H_L_loss.mean()

        # Cal H & L cdr loss.
        H_L_cdr_mask = H_L_cdr_mask.bool()
        H_L_cdr_p = torch.masked_select(H_L_pred, H_L_cdr_mask.unsqueeze(-1)).view(H_L_cdr_mask_tokens, -1)
        H_L_cdr_t = torch.masked_select(H_L_tgt, H_L_cdr_mask)
        H_L_cdr_loss = super().forward(H_L_cdr_p, H_L_cdr_t.long())
        H_L_cdr_losses = H_L_cdr_loss.mean()

        if self.reweight: # Uses Hoogeboom OARDM reweighting term
            H_L_timesteps = H_L_timesteps.sum(dim=-1)
            H_L_rwt_term = 1. / H_L_timesteps

            no_pad_number = torch.tensor([H_L_pred.size(1)]).repeat(H_L_pred.size(0)).to(H_L_rwt_term.device)
            H_L_rwt_term = H_L_rwt_term.repeat_interleave(H_L_timesteps)
            H_L_n_tokens = no_pad_number.repeat_interleave(H_L_timesteps)
            H_L_ce_loss = H_L_n_tokens * H_L_rwt_term * H_L_loss
            H_L_ce_losses = H_L_ce_loss.mean()  # reduce mean

        else:
            H_L_ce_losses = H_L_nll_losses
        return H_L_ce_losses, H_L_nll_losses.to(torch.float64), H_L_cdr_losses


class OasMaskedHeavyCrossEntropyLoss(CrossEntropyLoss):
    """Masked cross-entropy loss for sequences.
    Evaluates the cross-entropy loss at specified locations in a sequence
    When reweight = True, reweights CE according to Hoogeboom et al.;
    reweight term = 1/(D-t+1)
    Shape:
        Inputs:
            - pred: (N, L, n_tokens)
            - tgt: (N, L)
            - mask: (N, L) boolean
            - timestep (N, L) output from OAMaskCollater
            - input mask (N, L)
            - weight: (C, ): class weights for nn.CrossEntropyLoss

    Returns
        ce_losses
        nll_losses
    """
    def __init__(self, weight=None, reduction='none', reweight=True, tokenizer=Tokenizer()):
        self.reweight=reweight
        self.tokenizer = tokenizer
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, H_pred, H_tgt, H_mask, H_cdr_mask, H_timesteps):
        # Make sure we have that empty last dimension
        if len(H_mask.shape) == len(H_pred.shape) - 1:
            H_mask = H_mask.unsqueeze(-1)



        # Make sure mask is boolean
        H_mask = H_mask.bool()

        H_mask_tokens = H_mask.sum()  # masked tokens
        H_cdr_mask_tokens = H_cdr_mask.sum().int()
        # H_nonpad_tokens, L_nonpad_tokens = H_input_mask.sum(dim=1), L_input_mask.sum(dim=1) # nonpad tokens

        # Cal H loss.
        H_p = torch.masked_select(H_pred, H_mask).view(H_mask_tokens, -1)  # [T x K] predictions for each mask char
        H_t = torch.masked_select(H_tgt, H_mask.squeeze())  # [ T ] true mask char
        H_loss = super().forward(H_p, H_t.long()) # [ T ] loss per mask char
        H_nll_losses = H_loss.mean()

        # Cal H cdr loss.
        H_cdr_mask = H_cdr_mask.bool()
        H_cdr_p = torch.masked_select(H_pred, H_cdr_mask.unsqueeze(-1)).view(H_cdr_mask_tokens, -1)
        H_cdr_t = torch.masked_select(H_tgt, H_cdr_mask)
        H_cdr_loss = super().forward(H_cdr_p, H_cdr_t.long())
        H_cdr_losses = H_cdr_loss.mean()

        if self.reweight: # Uses Hoogeboom OARDM reweighting term
            H_rwt_term = 1. / H_timesteps

            no_pad_number = torch.tensor([H_pred.size(1)]).repeat(H_pred.size(0)).to(H_rwt_term.device)
            H_rwt_term = H_rwt_term.repeat_interleave(H_timesteps)
            H_n_tokens = no_pad_number.repeat_interleave(H_timesteps)
            H_ce_loss = H_n_tokens * H_rwt_term * H_loss
            H_ce_losses = H_ce_loss.mean()  # reduce mean

        else:
            H_ce_losses = H_nll_losses
        return H_ce_losses, H_nll_losses.to(torch.float64), H_cdr_losses


class OasMaskedNanoCrossEntropyLoss(CrossEntropyLoss):

    def __init__(self, weight=None, reduction='none'):
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, H_pred, H_tgt, H_cdr_mask, H_mask, H_timesteps, reconstruct=False):
        H_cdr_mask_tokens = H_cdr_mask.sum().int()
        H_cdr_mask = H_cdr_mask.bool()
        H_cdr_p = torch.masked_select(H_pred, H_cdr_mask.unsqueeze(-1)).view(H_cdr_mask_tokens, -1)
        H_cdr_t = torch.masked_select(H_tgt, H_cdr_mask)
        H_cdr_loss = super().forward(H_cdr_p, H_cdr_t.long())
        H_cdr_losses = H_cdr_loss.mean()

        if not reconstruct:
            return H_cdr_losses
        else:
            # Here we consider the loss of reconstruct the Nanobody method.
            H_mask = H_mask.bool()
            H_mask_tokens = H_mask.sum()

            H_p = torch.masked_select(H_pred, H_mask.unsqueeze(-1)).view(H_mask_tokens, -1)  # [T x K] predictions for each mask char
            H_t = torch.masked_select(H_tgt, H_mask.squeeze())  # [ T ] true mask char
            H_loss = super().forward(H_p, H_t.long())  # [ T ] loss per mask char
            # H_nll_losses = H_loss.mean()

            H_rwt_term = 1. / H_timesteps
            no_pad_number = torch.tensor([H_pred.size(1)]).repeat(H_pred.size(0)).to(H_rwt_term.device)
            H_rwt_term = H_rwt_term.repeat_interleave(H_timesteps)
            H_n_tokens = no_pad_number.repeat_interleave(H_timesteps)
            H_ce_loss = H_n_tokens * H_rwt_term * H_loss
            H_ce_losses = H_ce_loss.mean()

            return H_cdr_losses, H_ce_losses

