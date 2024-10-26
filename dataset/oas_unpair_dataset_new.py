import os
import pickle
import lmdb
import numpy as np
import random
import sys
import tempfile
current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)

import torch
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.tokenizer import Tokenizer
from utils.anti_numbering import get_seq_list_from_SeqRecords
from dataset.oas_pair_dataset_new import HEAVY_REGION_INDEX

# Abnativ fuction
from model.nanoencoder.abnativ_onehot import torch_masking_BERT_onehot


HEAVY_CDR_INDEX = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


LIGHT_CDR_INDEX = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3, 3, 3, 3,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Chn_seqs = set()

def _pad(tokenized, value, dim=2):
    """
    Utility function that pads batches to the same length.

    tokenized: list of tokenized sequences
    value: pad index
    """
    batch_size = len(tokenized)
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


class OasUnPairDataset(Dataset):

    def __init__(self,
                 data_dpath=None,
                 chaintype=None,
                 transform=None,
                 split_ratio=0.95
                 ):
        super().__init__()
        self.raw_path = os.path.dirname(data_dpath)
        self.data_path = data_dpath
        self.processed_path = os.path.join(self.raw_path,
                                           f'{chaintype}_test_nano.lmdb')
        self.index_path = os.path.join(self.raw_path,
                                       f'{chaintype}_nano_idx.pt')
        self.transform = transform
        self.split_ratio = split_ratio
        self.db = None

        self.keys = None

        # Filter set.
        self.Chn_seqs = set()

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

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


    def _process(self, is_VHH=True, verbose=True):
        db = lmdb.open(
            self.processed_path,
            map_size=15 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )

        # Deal Chain data.
        line_data_list = None
        with open(self.data_path, 'rb') as f:
            line_data_list = pickle.load(f)
            f.close()

        idx_list = []
        with db.begin(write=True, buffers=True) as txn:
            for line_idx, line in tqdm(enumerate(line_data_list), total=len(line_data_list)):
                name, chn_seq, pad_seq, chain_type, aho_seq, _ = line

                # Need to align the seq by aho, which is acceleration for the abnativ prediction.
                line_data = {
                    'name': name,
                    'seq': chn_seq,
                    'pad_seq': pad_seq,
                    'chain': chain_type,
                    'aho_seq': aho_seq
                }
                txn.put(
                    key=str(line_idx).encode(),
                    value=pickle.dumps(line_data)
                )
                idx_list.append(line_idx)
        db.close()

        # Idx divide. Only run at the first time.
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
    

class OasHeavyMaskCollater(object):
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
    def __init__(self, tokenizer=Tokenizer()):
        """ Add noise to model the vhh cdr3 residue. """

        self.tokenizer = tokenizer

    def __call__(self, h_sequences_dict):

        H_tokenized = [self.tokenizer.seq2idx(s_dict['pad_seq']) for s_dict in h_sequences_dict]
        H_cdr_index = [torch.tensor(HEAVY_CDR_INDEX) for _ in h_sequences_dict]

        H_type = [s_dict['chain'] for s_dict in h_sequences_dict]
        chain_type = torch.tensor([self.tokenizer.chain_type_idx(h_c) for h_c in H_type])


        H_max_len = max(len(t) for t in H_tokenized)

        H_src = []
        H_timesteps = []
        H_masks = []
        H_cdr_mask = []

        mask_id = torch.tensor(self.tokenizer.idx_msk, dtype=torch.int64)
        for i, h_x in enumerate(H_tokenized):
            """
            Which only consider the heavy chain. 
            """
            # Randomly generate timestep and indices to mask
            D = len(h_x)  # D should have the same dimensions as each sequence length
            # l_D = len(l_x)
            if D <= 1:  # for sequence length = 1 in dataset
                t = 1
            else:
                t = np.random.randint(1, D)  # randomly sample timestep

            num_mask = (D - t + 1)  # from OA-ARMS
            # Generate H mask.
            mask_arr = np.random.choice(D, num_mask, replace=False)  # Generates array of len num_mask

            h_index_arr = np.arange(0, H_max_len)  # index array [1...seq_len]
            h_mask = np.isin(h_index_arr, mask_arr, invert=False).reshape(
                h_index_arr.shape)  # True represents mask, vice versa
            h_cdr_mask = H_cdr_index[i] != 0
            h_mask = torch.tensor(h_mask, dtype=torch.bool)
            h_before_fix_true_number = h_mask[:D].sum()
            h_mask[:D] = h_mask[:D] * ~h_cdr_mask
            h_after_fix_true_number = h_mask[:D].sum()
            assert h_before_fix_true_number >= h_after_fix_true_number, 'H chain Mask has problem'
            h_num_mask = h_mask[:H_max_len].sum()
            true_num_mask = h_after_fix_true_number
            assert true_num_mask == h_num_mask
            H_masks.append(h_mask)
            H_cdr_mask.append(h_cdr_mask)

            # Generate timestep H.
            h_x_t = ~h_mask[0:D] * h_x + h_mask[0:D] * mask_id
            H_src.append(h_x_t)
            H_timesteps.append(h_num_mask)

        # PAD src out
        H_src = _pad(H_src, self.tokenizer.idx_pad)

        # Pad mask out
        H_masks = _pad(H_masks * 1, 0)  # , self.seq_length, 0)

        # Pad token out
        H_tokenized = _pad(H_tokenized, self.tokenizer.idx_pad)

        # Pad CDR mask for loss.
        H_cdr_mask = _pad(H_cdr_mask * 1, 0)

        # Pad H and L region index.
        H_region = torch.tensor([HEAVY_REGION_INDEX for _ in h_sequences_dict])
        H_timesteps = torch.tensor(H_timesteps)
        return (H_src, H_tokenized, H_region, chain_type,
                H_masks,
                H_cdr_mask,
                H_timesteps)


class OasCamelCollater(object):
    """
    Construct a collector for vhh seq.
    """
    def __init__(self, tokenizer=Tokenizer()):
        self.tokenizer = tokenizer

    def __call__(self, vhh_sequences_dict, is_VHH=True, verbose=True):
        # align_imgt_seq_list = [s_dict['pad_seq'] for s_dict in vhh_sequences_dict]
        failed_idx = [i for i, s_dict in enumerate(vhh_sequences_dict) if s_dict['aho_seq'][-3:] == '---']
        align_aho_seq_list = [s_dict['aho_seq'] for i , s_dict in enumerate(vhh_sequences_dict) if i not in failed_idx]

        # Need to transform.
        aho_torch = torch.stack([torch_masking_BERT_onehot(vhh_seq) for vhh_seq in align_aho_seq_list], dim=0)

        # Need to make sure those align failed seq will not be selected.
        # print(failed_idx)
        H_tokenized = [self.tokenizer.seq2idx(s_dict['pad_seq']) for idx, s_dict in enumerate(vhh_sequences_dict) if idx not in failed_idx]
        H_cdr_index = [torch.tensor(HEAVY_CDR_INDEX) for idx, _ in enumerate(vhh_sequences_dict) if idx not in failed_idx]
        H_cdr_index = torch.stack(H_cdr_index)


        H_tokenized = _pad(H_tokenized, self.tokenizer.idx_pad)
        frame_h = ~(H_cdr_index != 0) * H_tokenized
        frame_pad_true_indices = [[idx.item() for idx in row.nonzero()] for row in (frame_h == 21)]

        H_src = []
        H_masks = []
        H_max_len = max(len(t) for t in H_tokenized)
        H_timesteps = []
        mask_id = torch.tensor(self.tokenizer.idx_msk, dtype=torch.int64)
        for i, h_x in enumerate(H_tokenized):
            """
            Which only consider the heavy chain. 
            """
            # Randomly generate timestep and indices to mask
            # Here the D was limited, because has not align problem.
            D = 150  # D should have the same dimensions as each sequence length
            # l_D = len(l_x)
            if D <= 1:  # for sequence length = 1 in dataset
                t = 1
            else:
                t = np.random.randint(1, D)  # randomly sample timestep

            num_mask = (D - t + 1)  # from OA-ARMS
            # Generate H mask.
            frame_pad = np.array(frame_pad_true_indices[i])
            mask_arr = np.random.choice(D, num_mask, replace=False)  # Generates array of len num_mask
            mask_arr_mask = np.isin(mask_arr, frame_pad, invert=True)
            mask_arr = mask_arr[mask_arr_mask]

            h_index_arr = np.arange(0, H_max_len)  # index array [1...seq_len]
            h_mask = np.isin(h_index_arr, mask_arr, invert=False).reshape(
                h_index_arr.shape)  # True represents mask, vice versa
            h_cdr_mask = H_cdr_index[i] != 0
            h_x_pad_mask = h_x == self.tokenizer.idx_pad
            h_x_pad_mask = h_x_pad_mask * ~h_cdr_mask
            h_cdr_mask = h_cdr_mask.to(torch.int) + h_x_pad_mask.to(torch.int)
            h_cdr_mask = h_cdr_mask.bool()
            h_mask = torch.tensor(h_mask, dtype=torch.bool)
            h_before_fix_true_number = h_mask[:D].sum()
            h_mask[:D] = h_mask[:D] * ~h_cdr_mask[:D]
            h_after_fix_true_number = h_mask[:D].sum()
            assert h_before_fix_true_number >= h_after_fix_true_number, 'H chain Mask has problem'
            h_num_mask = h_mask[:H_max_len].sum()
            true_num_mask = h_after_fix_true_number
            assert true_num_mask == h_num_mask
            H_masks.append(h_mask)

            # Generate timestep H.
            h_x_t = ~h_mask * h_x + h_mask * mask_id
            H_src.append(h_x_t)
            H_timesteps.append(h_num_mask)

        # Pad token out
        H_src = _pad(H_src, self.tokenizer.idx_pad)
        H_masks = _pad(H_masks * 1, 0)  # , self.seq_length, 0)

        # Pad H and L region index.
        H_region = torch.tensor([HEAVY_REGION_INDEX for _ in H_cdr_index])

        H_cdr_mask = H_cdr_index != 0

        H_timesteps = torch.tensor(H_timesteps)

        return H_src, H_masks, H_tokenized, H_region, H_cdr_index, H_cdr_mask, H_timesteps, aho_torch  # align_imgt_seq_list, align_aho_seq_list