import os
import pickle
import lmdb
import random
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# from tfold_utils.prot_constants import RESD_NAMES_1C
from utils.tokenizer import Tokenizer
from dataset.preprocess import (
    parse_cgz_file, 
    HEAVY_CDR_INDEX, LIGHT_CDR_INDEX,
    HEAVY_CDR_KABAT_NO_VERNIER, LIGHT_CDR_KABAT_NO_VERNIER
)

# Abnativ function.
from model.nanoencoder.abnativ_onehot import torch_masking_BERT_onehot

Chn_seqs = set()

HEAVY_REGION_INDEX = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                      3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                      6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]


LIGHT_REGION_INDEX = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                      3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                      6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
# May can pad the light cdr region

def _pad(tokenized, value, max_len=None, dim=2):
    """
    Utility function that pads batches to the same length.

    tokenized: list of tokenized sequences
    value: pad index
    """
    batch_size = len(tokenized)
    if max_len is None:
        max_len = max(len(t) for t in tokenized)

    if dim == 3: # dim = 3 (one hot)
        categories = tokenized[0].shape[-1]
        output = torch.zeros((batch_size, max_len, categories)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t), :] = t
    elif dim == 2: # dim = 2 (tokenized)
        output = torch.zeros((batch_size, max_len)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t)] = t
    else:
        print("padding not supported for dim > 3")
    return output


def zero_batch(seq):
    """
    Get batch zero
    :param seq: str
    :return: len(str) * 0
    """
    str_num = len(seq)
    return str_num * [0]


def cdr_batch(idx, seq):
    '''
    :param idx: int
    :param seq: str
    :return: [int] * len(str)
    '''
    if idx == 1:
        return len(seq) * [idx]
    elif idx == 3:
        return len(seq) * [idx-1]
    elif idx == 5:
        return len(seq) * [idx-2]


def operate_line(seg_seqs_hc, seg_seqs_lc, pad_seg_seqs_hc, pad_seg_seqs_lc, type_hc, type_lc):
    H_seq = ''.join(seg_seqs_hc)
    L_seq = ''.join(seg_seqs_lc)

    H_pad_seq = ''.join(pad_seg_seqs_hc)
    L_pad_seq = ''.join(pad_seg_seqs_lc)

    hc_cdr = sum([zero_batch(seq)
                  if idx % 2 == 0 else cdr_batch(idx, seq) for idx, seq in enumerate(seg_seqs_hc)], [])
    lc_cdr = sum([zero_batch(seq)
                  if idx % 2 == 0 else cdr_batch(idx, seq) for idx, seq in enumerate(seg_seqs_lc)], [])
    H_seq_cdr = torch.tensor(hc_cdr)
    L_seq_cdr = torch.tensor(lc_cdr)

    pad_hc_cdr = sum([zero_batch(seq)
                      if idx % 2 == 0 else cdr_batch(idx, seq) for idx, seq in enumerate(pad_seg_seqs_hc)], [])
    pad_lc_cdr = sum([zero_batch(seq)
                      if idx % 2 == 0 else cdr_batch(idx, seq) for idx, seq in enumerate(pad_seg_seqs_lc)], [])
    pad_H_seq_cdr = torch.tensor(pad_hc_cdr)
    pad_L_seq_cdr = torch.tensor(pad_lc_cdr)

    operate_data = {
        'h_seq': H_seq,
        'l_seq': L_seq,
        'h_pad_seq': H_pad_seq,
        'l_pad_seq': L_pad_seq,
        'h_seq_cdr': H_seq_cdr,
        'l_seq_cdr': L_seq_cdr,
        'pad_h_seq_cdr': pad_H_seq_cdr,
        'pad_l_seq_cdr': pad_L_seq_cdr,
        'H_type': type_hc,
        'L_type': type_lc
    }
    return operate_data



class OasPairDataset(Dataset):

    def __init__(self, raw_path, mouse=False, transform=None, version='tmp', split_ratio=0.95):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.split_ratio = split_ratio
        self.cgz_path = os.path.join(self.raw_path, 'new_cgz_data')
        self.processed_path = os.path.join(self.raw_path, 'newprocessed/' +
                                           f'train_processed_pad_{version}.lmdb')
        self.index_path = os.path.join(self.raw_path, 'newprocessed/' +
                                       f'oas_pair_index_pad_{version}.pt')
        self.mouse = mouse
        self.transform = transform
        self.db = None

        self.keys = None
        
        # Filter set.
        self.Chn_seqs = set()

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        
        if not os.path.exists(self.index_path):
            print(f'{self.index_path} does not exist, begin generating...')
            self._idx_pt()

    def _idx_pt(self):
        idx_db =  lmdb.open(
            self.processed_path,
            # map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with idx_db.begin() as txn:
            # cursor = txn.cursor()
            # for key, value in cursor:
            keys = list(txn.cursor().iternext(values=False))
            max_idx = max([int(key.decode()) for key in keys])
        idx_db.close()
        idx_list = list(range(max_idx))

        random.shuffle(idx_list)
        split = int(len(idx_list) * self.split_ratio)
        train_idx = idx_list[:split]
        val_idx = idx_list[split:]
        idx_dict = {'train': train_idx, 'val': val_idx}
        torch.save(idx_dict, self.index_path)
        print('Sum number pairs:', len(idx_list))


    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None


    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=100 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )

        tokenizer = Tokenizer(has_bos=False, has_eos=False)
        idx_list = []
        seq_list = []
        cgz_names = sorted(os.listdir(self.cgz_path))
        for cgz_fname in tqdm(cgz_names, desc='parsing GZ-compressed CSV files'):
            cgz_fpath = os.path.join(self.cgz_path, cgz_fname)
            if not os.path.isdir(cgz_fpath):
                seq_data, self.Chn_seqs = parse_cgz_file(cgz_fpath, self.Chn_seqs, mouse=self.mouse)
                seq_list.extend(seq_data)
        with db.begin(write=True, buffers=True) as txn:
            for idx, seq_l_data in enumerate(tqdm(seq_list)):
                if self.mouse:
                    name, seq_hc, seq_lc, pad_seq_hc, pad_seq_lc, aho_pad_seq_hc, aho_pad_seq_lc, type_hc, type_lc = seq_l_data
                else:
                    name, seq_hc, seq_lc, pad_seq_hc, pad_seq_lc, type_hc, type_lc = seq_l_data
                    aho_pad_seq_hc, aho_pad_seq_lc = None, None
                line_data = {
                    'name': name,
                    'h_seq': seq_hc,
                    'l_seq': seq_lc,
                    'h_pad_seq': pad_seq_hc,
                    'l_pad_seq': pad_seq_lc,
                    'aho_h_pad_seq': aho_pad_seq_hc,
                    'aho_l_pad_seq': aho_pad_seq_lc,
                    'h_type': type_hc,
                    'l_type': type_lc
                }
                txn.put(
                    key=str(idx).encode(),
                    value=pickle.dumps(line_data)
                )
                idx_list.append(idx)
        db.close()

        # idx_divide.
        random.shuffle(idx_list)
        split = int(len(idx_list) * self.split_ratio)
        train_idx = idx_list[:split]
        val_idx = idx_list[split:]
        idx_dict = {'train': train_idx, 'val': val_idx}
        torch.save(idx_dict, self.index_path)
        print('Sum number pairs:', len(idx_list))


    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        return data

def merge_stack(H_d, L_d):
    """
    Cat tensor for training.
    :param H_d: list of tensor,
    :param L_d: list of tensor
    :return: cat tensor
    """
    return torch.cat(
        (torch.stack(H_d), torch.stack(L_d)), dim=0
    )

def light_pad_token(l_aa, pad_token=None):
    """

    :param l_aa: AA_seq
    :return: AA_seq
    """
    assert pad_token is not None, print("Need pad token.")
    pad_length = len(HEAVY_CDR_INDEX) - len(LIGHT_CDR_INDEX)
    pad_seq = torch.tensor([pad_token]*pad_length)
    l_aa = torch.cat((l_aa, pad_seq), dim=0)
    return l_aa

def light_pad_cdr(l_cdr_idx, pad_v=0):
    """
    :param l_cdr_idx: idx list
    :return: idx list
    """
    copy_l_cdr_idx = deepcopy(l_cdr_idx)
    pad_length = len(HEAVY_CDR_INDEX) - len(LIGHT_CDR_INDEX)
    pad_seq = [pad_v] * pad_length
    copy_l_cdr_idx.extend(pad_seq)
    return copy_l_cdr_idx


class OasPairMaskCollater(object):
    """
    OrderAgnosic Mask Collater for masking batch data according to Hoogeboom et al. OA ARDMS
    inputs:
        list_sequences_dict : dict of H/L chain torch tensor, including the CDR regions.
        inputs_padded: if inputs are padded (due to truncation in Simple_Collater) set True (default False)

    OA-ARM variables:
        D : possible permutations from 0.. max length
        t : randomly selected timestep

    outputs:
        src : source  masked sequences (model input)
        timesteps: (D-t+1) term
        tokenized: tokenized sequences (target seq)
        masks: masks used to generate src
    """
    def __init__(self, n_region, consider_mouse=False, tokenizer=Tokenizer()):
        self.tokenizer = tokenizer
        self.consider_mouse = consider_mouse
        if n_region == 7:
            self.region_off = 0
        elif n_region == 14:
            self.region_off = 7
        else:
            print('Region has problem, please check.')


    def __call__(self, list_sequences_dict):
        # insert_tokenized = torch.tensor([self.tokenizer.idx_eos])
        # Need to consider the aho.
        if self.consider_mouse:
            align_aho_h_seq_list = [torch_masking_BERT_onehot(s_dict['aho_h_pad_seq']) for s_dict in list_sequences_dict]
            align_aho_l_seq_list = [torch_masking_BERT_onehot(s_dict['aho_l_pad_seq']) for s_dict in list_sequences_dict]
            align_aho_h_seq_token = torch.stack(align_aho_h_seq_list, dim=0)
            align_aho_l_seq_token = torch.stack(align_aho_l_seq_list, dim=0)

        H_tokenized = [self.tokenizer.seq2idx(s_dict['h_pad_seq']) for s_dict in list_sequences_dict]
        L_tokenized = [self.tokenizer.seq2idx(s_dict['l_pad_seq']) for s_dict in list_sequences_dict]
        H_L_tokenized = [torch.cat((H_tok, L_tok), dim=0)for H_tok, L_tok in zip(H_tokenized, L_tokenized)]


        if self.consider_mouse:
            H_cdr_index = [torch.tensor(HEAVY_CDR_KABAT_NO_VERNIER) for _ in list_sequences_dict]
            L_cdr_index = [torch.tensor(LIGHT_CDR_KABAT_NO_VERNIER) for _ in list_sequences_dict]
            H_L_cdr_index = [torch.cat((H_cdr, L_cdr), dim=0) for H_cdr, L_cdr in zip(H_cdr_index, L_cdr_index)]

        else:
            H_cdr_index = [torch.tensor(HEAVY_CDR_INDEX) for _ in list_sequences_dict]
            L_cdr_index = [torch.tensor(LIGHT_CDR_INDEX) for _ in list_sequences_dict]
            H_L_cdr_index = [torch.cat((H_cdr, L_cdr), dim=0) for H_cdr, L_cdr in zip(H_cdr_index, L_cdr_index)]


        H_type = [s_dict['h_type'] for s_dict in list_sequences_dict]
        L_type = [s_dict['l_type'] for s_dict in list_sequences_dict]
        chain_type = torch.cat(
            (
                torch.tensor([self.tokenizer.chain_type_idx(h_c) for h_c in H_type]),
                torch.tensor([self.tokenizer.chain_type_idx(l_c) for l_c in L_type])
            ),
            dim=0
        )
        batch_split_index = torch.cat(
            (
                torch.tensor(0).repeat(len(H_cdr_index)),
                torch.tensor(1).repeat(len(L_cdr_index))
            ),
            dim=0
        )

        H_L_max_len = max(len(t) for t in H_L_tokenized)
        H_max_len = max(len(t) for t in H_tokenized)
        L_max_len = max(len(t) for t in L_tokenized)
        H_L_src = []
        # L_src = []
        H_L_timesteps = []
        # L_timesteps = []
        H_L_masks = []
        # L_masks = []
        H_L_cdr_mask = []
        # L_cdr_mask = []
        # D: 291, H: 152, L: 139.
        mask_id = torch.tensor(self.tokenizer.idx_msk, dtype=torch.int64)
        for i, h_l_x in enumerate(H_L_tokenized):
            # Randomly generate timestep and indices to mask
            D = len(h_l_x)   # D should have the same dimensions as each sequence length
            # l_D = len(l_x)
            if D <= 1:  # for sequence length = 1 in dataset
                t = 1
            else:
                t = np.random.randint(1, D) # randomly sample timestep

            num_mask = (D-t+1) # from OA-ARMS
            # Generate H mask.
            mask_arr = np.random.choice(D, num_mask, replace=False) # Generates array of len num_mask

            h_l_index_arr = np.arange(0, H_L_max_len) #index array [1...seq_len]
            h_l_mask = np.isin(h_l_index_arr, mask_arr, invert=False).reshape(h_l_index_arr.shape) # True represents mask, vice versa
            h_l_cdr_mask = H_L_cdr_index[i] != 0
            if self.consider_mouse:
                h_l_x_pad_mask = h_l_x == self.tokenizer.idx_pad
                h_l_x_pad_mask = h_l_x_pad_mask * ~h_l_cdr_mask
                h_l_cdr_mask = h_l_cdr_mask.to(torch.int) + h_l_x_pad_mask.to(torch.int)
                h_l_cdr_mask = h_l_cdr_mask.bool()
            h_l_mask = torch.tensor(h_l_mask, dtype=torch.bool)
            h_before_fix_true_number = h_l_mask[:D].sum()
            h_l_mask[:D] = h_l_mask[:D] * ~h_l_cdr_mask
            h_after_fix_true_number = h_l_mask[:D].sum()
            assert h_before_fix_true_number >= h_after_fix_true_number, 'H chain Mask has problem'
            h_num_mask = h_l_mask[:H_max_len].sum()
            l_num_mask = h_l_mask[H_max_len:].sum()
            true_num_mask = h_after_fix_true_number
            assert true_num_mask == h_num_mask + l_num_mask
            H_L_masks.append(h_l_mask)
            H_L_cdr_mask.append(h_l_cdr_mask)

            # Generate timestep H.
            h_l_x_t = ~h_l_mask[0:D] * h_l_x + h_l_mask[0:D] * mask_id
            H_L_src.append(h_l_x_t)
            H_L_timesteps.append([h_num_mask, l_num_mask])

        # PAD src out
        H_L_src = _pad(H_L_src, self.tokenizer.idx_pad)

        # Pad mask out
        H_L_masks = _pad(H_L_masks*1, 0) #, self.seq_length, 0)


        # Pad token out
        H_L_tokenized = _pad(H_L_tokenized, self.tokenizer.idx_pad)

        # Pad CDR mask for loss.
        H_L_cdr_mask = _pad(H_L_cdr_mask*1, 0)

        # Pad H and L region index.
        H_region = torch.tensor([HEAVY_REGION_INDEX for _ in list_sequences_dict])
        L_region = torch.tensor([LIGHT_REGION_INDEX for _ in list_sequences_dict]) + self.region_off
        H_L_region = torch.cat((H_region, L_region), dim=-1)
        H_L_timesteps = torch.tensor(H_L_timesteps)
        if self.consider_mouse:
            return (H_L_src, H_L_tokenized, H_L_region, chain_type, batch_split_index,
                    H_L_masks,
                    H_L_cdr_mask,
                    H_L_timesteps,
                    align_aho_h_seq_token,
                    align_aho_l_seq_token)
        else:
            return (H_L_src, H_L_tokenized, H_L_region, chain_type, batch_split_index,
                    H_L_masks,
                    H_L_cdr_mask,
                    H_L_timesteps)


class OasPairMouseCollector(object):
    """

    """
    def __init__(self, n_region, tokenizer=Tokenizer()):
        self.tokenizer = tokenizer
        if n_region == 7:
            self.region_off = 0
        elif n_region == 14:
            self.region_off = 7
        else:
            print('Region has problem, please check.')

    def __call__(self, list_sequences_dict, light_pad_v=0):
        H_tokenized = [self.tokenizer.seq2idx(s_dict['h_pad_seq']) for s_dict in list_sequences_dict]
        L_tokenized = [self.tokenizer.seq2idx(s_dict['l_pad_seq']) for s_dict in list_sequences_dict]
        H_L_tokenized = [torch.cat((H_tok, L_tok), dim=0) for H_tok, L_tok in zip(H_tokenized, L_tokenized)]

        H_cdr_index = [torch.tensor(HEAVY_CDR_INDEX) for _ in list_sequences_dict]
        L_cdr_index = [torch.tensor(LIGHT_CDR_INDEX) for _ in list_sequences_dict]
        H_L_cdr_index = [torch.cat((H_cdr, L_cdr), dim=0) for H_cdr, L_cdr in zip(H_cdr_index, L_cdr_index)]

        H_type = [s_dict['h_type'] for s_dict in list_sequences_dict]
        L_type = [s_dict['l_type'] for s_dict in list_sequences_dict]
        chain_type = torch.cat(
            (
                torch.tensor([self.tokenizer.chain_type_idx(h_c) for h_c in H_type]),
                torch.tensor([self.tokenizer.chain_type_idx(l_c) for l_c in L_type])
            ),
            dim=0
        )
        batch_split_index = torch.cat(
            (
                torch.tensor(0).repeat(len(H_cdr_index)),
                torch.tensor(1).repeat(len(L_cdr_index))
            ),
            dim=0
        )
        H_region = torch.tensor([HEAVY_REGION_INDEX for _ in list_sequences_dict])
        L_region = torch.tensor([LIGHT_REGION_INDEX for _ in list_sequences_dict]) + self.region_off
        H_L_region = torch.cat((H_region, L_region), dim=-1)
        return (H_L_tokenized, H_L_cdr_index, chain_type, batch_split_index, H_L_region)