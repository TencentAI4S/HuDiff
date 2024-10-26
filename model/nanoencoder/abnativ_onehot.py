# (c) 2023 Sormannilab and Aubin Ramon
#
# AbNAtiV BERT-style masking OneHotEncoder Iterator. 
#
# ============================================================================

from typing import Tuple
import numpy as np
import math
import random
import pandas as pd
from pandas.api.types import CategoricalDtype

from Bio import SeqIO
import torch

alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

def data_loader_masking_bert_onehot_fasta(fp_data: str, batch_size: int, perc_masked_residues: float, 
                                          is_masking: bool) -> torch.utils.data.DataLoader:
    ''' 
    Generate a Torch dataloader iterator from fp_data.

    Parameters
    ----------
    fp_data: str
        The align seq list.
    batch_size: int
    perc_masked_residues: float
        Ratio of residues to apply the BERT masking on (between 0 and 1).
    is_masking: bool

    '''
    iterator = IterableMaskingBertOnehotDatasetFasta(fp_data, perc_masked_residues=perc_masked_residues, is_masking=is_masking)
    loader = torch.utils.data.DataLoader(iterator, batch_size=batch_size, num_workers=0, shuffle=is_masking)
    return loader 


class IterableMaskingBertOnehotDatasetFasta(torch.utils.data.IterableDataset):
    '''
    BERT-style masking onehot generator for all sequences given a fasta file.
    '''
    def __init__(self, fp_seq_list, perc_masked_residues=0.0, is_masking=False):
        self.fp_seq = fp_seq_list
        self.perc_masked_residues = perc_masked_residues
        self.is_masking = is_masking

    def __iter__(self) -> torch.utils.data.IterableDataset:
        for record in SeqIO.parse(self.fp_seq, 'fasta'):
            if len(str(record.seq)) != 149:
                raise Exception(
                    f'Sequence {record.id} is shorter than 149 characters. All sequences must be aligned with the AHo scheme.')
            yield torch_masking_BERT_onehot(str(record.seq), perc_masked_residues=self.perc_masked_residues,
                                            is_masking=self.is_masking)

def torch_masking_BERT_onehot(seq: str, perc_masked_residues: float=0.0, 
                              is_masking: bool=False, alphabet: list=alphabet) -> Tuple[torch.Tensor, torch.Tensor] or torch.Tensor:
    '''
    BERT-style masking on a one-hot encoding input. When a residue is masked, it is replaced 
    by the dummie vector [1/21,...,1/21] of size 21. 80% of perc_masked_residues are masked, 
    10% are replaced by another residue, 10% are left as they are.

    Parameters
    ----------
    seq: str
    perc_masked_residues: float
        Ratio of residues to apply the BERT masking on (between 0 and 1).
    is_masking: bool
        False for evaluation.
    alphabet: list 
        List of string of the alphabet of residues used in the one hot encoder

    Returns
    -------
    onehot_seq: tensor
        One hot encoded input.
    m_tf_onehot_seq: tensor 
        BERT masked one hot encoded input.

    '''

    alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']

    # One Hot Encoding
    onehot_seq = np.array((pd.get_dummies(pd.Series(list(seq)).astype(CategoricalDtype(categories=alphabet))))).astype(float)
    onehot_seq = torch.tensor(onehot_seq, dtype=torch.float32)
    ln_seq = len(onehot_seq)

    m_tf_onehot_seq = onehot_seq.clone().detach()

    if is_masking:
        if perc_masked_residues > 1:
            raise NotImplementedError('Masking percentage should be between 0 and 1.')

        # the onehot vector of the masked residue
        len_alphabet = len(alphabet)
        masked_letter = [1/len_alphabet]*len_alphabet

        # MASKING
        nb_masking = math.floor(ln_seq * perc_masked_residues)
        nb_to_mask = math.floor(nb_masking*0.8) #80% replace with mask token
        nb_to_replace = math.floor(nb_masking*0.1) #10% replace with random residue

        if nb_to_mask != 0:

            rd_ids = torch.Tensor(random.sample(range(ln_seq),ln_seq)[:nb_to_mask+nb_to_replace]).type(torch.int64)

            rd_alphabet_selection_to_replace = random.choices(alphabet, k=nb_to_replace)
            dummies_to_replace =  np.array((pd.get_dummies(pd.Series(rd_alphabet_selection_to_replace).astype(CategoricalDtype(categories=alphabet)))))

            updates = np.array([masked_letter]*nb_to_mask)
            updates = torch.Tensor(np.concatenate((updates,dummies_to_replace)))

            m_tf_onehot_seq[rd_ids] = updates

    if is_masking:
        return onehot_seq, m_tf_onehot_seq
    else:
        return onehot_seq