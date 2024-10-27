"""The tokenizer for amino-acid sequences."""

import torch
from torch import nn
"""Constants."""

import numpy as np


# mapping between 1-char & 3-char residue names
RESD_MAP_1TO3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}
RESD_MAP_3TO1 = {v: k for k, v in RESD_MAP_1TO3.items()}
RESD_NAMES_1C = sorted(list(RESD_MAP_1TO3.keys()))
RESD_NAMES_3C = sorted(list(RESD_MAP_1TO3.values()))
RESD_NUM = len(RESD_NAMES_1C)  # := 20.
RESD_WITH_X = RESD_NAMES_1C + ['X']
RESD_ORDER_WITH_X = {restype: i for i, restype in enumerate(RESD_WITH_X)}
N_ATOMS_PER_RESD = 14  # TRP
N_ANGLS_PER_RESD = 7  # TRP (omega, phi, psi, chi1, chi2, chi3, and chi4)


class Tokenizer():
    """The tokenizer for amino-acid sequences."""

    def __init__(self, has_bos=False, has_eos=False):
        """Constructor function."""

        # setup configurations
        self.has_bos = has_bos
        self.has_eos = has_eos

        # additional configurations
        # self.tok_bos = '<bos>'
        self.tok_eos = '<eos>'
        self.tok_msk = '<msk>'
        # self.tok_unk = '<unk>'
        self.tok_pad = '-'
        self.toks = [*RESD_WITH_X, self.tok_pad, self.tok_msk]
        self.tok2idx_dict = {tok: idx for idx, tok in enumerate(self.toks)}
        self.idx_msk = self.tok2idx(self.tok_msk)
        self.idx_pad = self.tok2idx(self.tok_pad)


    @property
    def n_toks(self):
        """Get the number of tokens."""

        return len(self.toks)


    def tok2idx(self, tok):
        """Convert a single token into its index."""

        return self.tok2idx_dict[tok]


    def seq2idx(self, aa_seq):
        """Convert the amino-acid sequence into a 1-D vector of token indices."""

        aa_seq_ext = [*aa_seq]
        if self.has_bos:
            aa_seq_ext = [self.tok_bos] + aa_seq_ext
        if self.has_eos:
            aa_seq_ext = aa_seq_ext + [self.tok_eos]
        idx_vec = torch.tensor([self.tok2idx_dict[x] for x in aa_seq_ext])

        return idx_vec


    def seq2idx_batch(self, aa_seq_list):
        """Convert amino-acid sequences into token indices in the batch mode."""

        idx_vec_list = [self.seq2idx(x) for x in aa_seq_list]
        idx_mat = nn.utils.rnn.pad_sequence(
            idx_vec_list, batch_first=True, padding_value=self.idx_pad)

        return idx_mat


    def idx2seq(self, idx_vec):
        """Convert the 1-D vector of token indices into an amino-acid sequence."""

        aa_seq_ext = [self.toks[x] for x in idx_vec.tolist() if x != self.idx_pad]
        if self.has_bos:
            aa_seq_ext = aa_seq_ext[1:]  # skip the <bos> token
        if self.has_eos:
            aa_seq_ext = aa_seq_ext[:-1]  # skip the <eos> token
        aa_seq = ''.join(aa_seq_ext)

        return aa_seq

    def idx2seq_pad(self, idx_vec):
        """Convert the 1-D vector of token indices into an amino-acid sequence."""

        aa_seq_ext = [self.toks[x] for x in idx_vec.tolist()]
        if self.has_bos:
            aa_seq_ext = aa_seq_ext[1:]  # skip the <bos> token
        if self.has_eos:
            aa_seq_ext = aa_seq_ext[:-1]  # skip the <eos> token
        aa_seq = ''.join(aa_seq_ext)

        return aa_seq

    def idx2seq_pad_batch(self, idx_mat):
        """Convert token indices into amino-acid sequences in the batch mode."""

        n_seqs = idx_mat.shape[0]
        aa_seq_list = [self.idx2seq_pad(idx_mat[x]) for x in range(n_seqs)]

        return aa_seq_list

    def idx2seq_batch(self, idx_mat):
        """Convert token indices into amino-acid sequences in the batch mode."""

        n_seqs = idx_mat.shape[0]
        aa_seq_list = [self.idx2seq(idx_mat[x]) for x in range(n_seqs)]

        return aa_seq_list

    def chain_type_idx(self, chain):
        if chain == 'H':
            return 0
        elif chain == 'L':
            return 1
        elif chain == 'K':
            return 2
        else:
            raise TypeError('Chain Type has problem.')

def main():
    """Main entry."""

    # test samples
    aa_seq_list = [
        'EVQLVESGGGLVQPGGSLRLSSAISGFSISSTSIDWVRQAPGKGLEWVARISPSSGSTSYADSVKGRFTISADTSKNTVYLQMNSLRAEDTAVYYTGRPLPEMGFFTQIPAMVDYRGQGTLVTVSS',
        'QVQLQESGGGLVQPGGSLRLSCAASGFTFSSAIMTWVRQAPGKGREWVSTIGSDGSITTYADSVKGRFTISRDNARNTLYLQMNSLKPEDTAVYYCTSAGRRGPGTQVTVSS',
    ]

    # initialization
    tokenizer = Tokenizer()
    print(f'# of tokens: {tokenizer.n_toks}')

    # test w/ <seq2idx_batch>
    idx_mat = tokenizer.seq2idx_batch(aa_seq_list)
    print(f'idx_mat: {idx_mat.shape}')

    # test w/ <idx2seq_batch>
    aa_seq_list_out = tokenizer.idx2seq_batch(idx_mat)
    print(f'sequences: {aa_seq_list_out}')
    for aa_seq, aa_seq_out in zip(aa_seq_list, aa_seq_list_out):
        assert aa_seq == aa_seq_out, f'mismatched amino-acid sequences: {aa_seq} vs. {aa_seq_out}'


if __name__ == '__main__':
    main()