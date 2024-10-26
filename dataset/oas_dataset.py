"""The OAS dataset."""

import os
import re
import random
import logging
from collections import defaultdict
from tqdm import tqdm

from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from torch.utils.data import get_worker_info

from tfold_utils.common_utils import tfold_init
from tfold_utils.torch_utils import inspect_data
from tfold_utils.prot_constants import RESD_NAMES_1C
from utils.anti_numbering import get_regions


def parse_csv_file(path, n_seqs_max=4096):
    """Parse the CSV file."""

    # set of standard amino-acids
    resd_names = set(RESD_NAMES_1C)

    # parse the CSV file
    seq_list = []
    headers = None
    with open(path, 'r', encoding='UTF-8') as i_file:
        for i_line in i_file:
            sub_strs = i_line.strip().split(',')
            if headers is None:
                headers = sub_strs
            else:
                seq_dict = {k: v for k, v in zip(headers, sub_strs)}
                # if len(set(seq_dict['chn']) - resd_names) != 0:
                #     continue
                seq_list.append(seq_dict)
                if (n_seqs_max != -1) and (len(seq_list) >= n_seqs_max):
                    break

    return seq_list


class OasDataset(IterableDataset):
    """The OAS dataset."""

    def __init__(
            self,
            csv_dpath=None,  # directory path to CSV files
            pool_size=65536,  # minimal number of candidate sequences in the pool
            n_seqs_max=4096,  # maximal number of sequences parsed from a single CSV file
            spc_mode='human-only',  # species mode (choices: 'no-human' / 'human-only')
        ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.csv_dpath = csv_dpath
        self.pool_size = pool_size
        self.n_seqs_max = n_seqs_max
        self.spc_mode = spc_mode

        # initialize the dataset
        self.__init_dataset()


    def __iter__(self):
        """Return an iterator of samples in the dataset."""

        # initialization
        # n_files = len(self.file_list)

        # validate the worker information
        worker_info = get_worker_info()
        assert worker_info is None, 'only single-process data loading is supported'

        # initialize a new epoch
        # random.shuffle(self.file_list)
        logging.debug('=== start of epoch ===')

        # initialize the sequence pool
        idx_file = 0
        seq_pool = []  # list of (species, chn_type, seq_dict)-tuples
        while (len(seq_pool) < self.pool_size):
            # species, chn_type, csv_fpath = self.file_list[idx_file]
            csv_fpath = self.csv_dpath
            seq_list = parse_csv_file(csv_fpath, self.n_seqs_max)
            logging.debug('adding %d sequences from %s', len(seq_list), csv_fpath)
            seq_pool.extend([(x['ENTRY'], x['HSEQ'], x['LSEQ']) for x in seq_list])
        random.shuffle(seq_pool)
        logging.debug('# of CSV files: %d (total)', len(seq_pool))

        # parse CSV files and return an iterator of samples
        while len(seq_pool) > 0:
            # build an input dict
            species, H_seq, L_seq = seq_pool.pop()
            try:
                inputs = self.__build_inputs(species, H_seq, L_seq)
                yield inputs
            except:
                continue

            # early-exit if no expansion is needed
            if (len(seq_pool) >= self.pool_size):
                continue

        # indicate the end of current epoch
        logging.debug('=== end of epoch ===')


    def __init_dataset(self):
        """Initialize the dataset."""
        logging.debug('=== CSV files ===')
        logging.debug('load CSV files {}'.format(self.csv_dpath))


    @classmethod
    def __build_inputs(cls, species, H_seq, L_seq):
        """Build an input dict."""

        inputs = {
            'spec': species,
            'hseq': H_seq,
            'lseq': L_seq,
            'hseq_cdr_region': get_regions(H_seq),
            'lseq_cdr_region': get_regions(L_seq)
        }

        return inputs


class OasDatasetShuf(IterableDataset):
    """The OAS dataset for CSV files containing randomly shuffled sequences."""

    def __init__(
            self,
            csv_dpath=None,  # directory path to CSV files
            spc_mode='no-human',  # species mode (choices: 'no-human' / 'human-only')
        ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.csv_dpath = csv_dpath
        self.spc_mode = spc_mode

        # initialize the dataset
        self.__init_dataset()


    def __iter__(self):
        """Return an iterator of samples in the dataset."""

        # initialization
        species = None
        chn_type = None

        # validate the worker information
        worker_info = get_worker_info()
        assert worker_info is None, 'only single-process data loading is supported'

        # initialize a new epoch
        logging.debug('=== start of epoch ===')

        # initialize the sequence pool
        idx_file = 0
        n_files = len(self.file_list)
        random.shuffle(self.file_list)
        species, chn_type, csv_fpath = self.file_list[idx_file]
        seq_list = parse_csv_file(csv_fpath, n_seqs_max=-1)
        logging.debug('adding %d sequences from %s', len(seq_list), csv_fpath)
        idx_seq = 0
        n_seqs = len(seq_list)
        random.shuffle(seq_list)

        # parse CSV files and return an iterator of samples
        while True:
            # build an input dict
            inputs = self.__build_inputs(species, chn_type, seq_list[idx_seq])
            yield inputs

            # early-exit if no expansion is needed
            idx_seq += 1
            if idx_seq < n_seqs:
                continue

            # parse the next CSV file
            idx_file += 1
            if idx_file == n_files:  # no available CSV files left
                break
            species, chn_type, csv_fpath = self.file_list[idx_file]
            seq_list = parse_csv_file(csv_fpath, n_seqs_max=-1)
            logging.debug('adding %d sequences from %s', len(seq_list), csv_fpath)
            idx_seq = 0
            n_seqs = len(seq_list)
            random.shuffle(seq_list)

        # indicate the end of current epoch
        logging.debug('=== end of epoch ===')


    def __init_dataset(self):
        """Initialize the dataset."""

        self.file_list = []
        for species in os.listdir(self.csv_dpath):
            # check whether the current species should be skipped
            if self.spc_mode == 'no-human':
                if species == 'human':
                    continue
            elif self.spc_mode == 'human-only':
                if species != 'human':
                    continue
            else:
                raise ValueError(f'unrecognized species mode: {self.spc_mode}')

            # enumerate all the CSV files for the current species
            for csv_fname in os.listdir(os.path.join(self.csv_dpath, species)):
                chn_type = 'hc' if csv_fname.startswith('heavy') else 'lc'
                csv_fpath = os.path.join(self.csv_dpath, species, csv_fname)
                self.file_list.append((species, chn_type, csv_fpath))

        logging.debug('=== list of CSV files ===')
        logging.debug('\n'.join([str(x) for x in self.file_list]))


    @classmethod
    def __build_inputs(cls, species, chn_type, seq_dict):
        """Build an input dict."""

        inputs = {
            'spec': species,
            'type': chn_type,
            'seq': seq_dict['chn'],
            'regions': {k: v for k, v in seq_dict.items() if k != 'chn'},
        }

        return inputs


def main():
    """Main entry."""

    # configurations
    data_dir_csv = '/test.pkl'

    # initialization
    tfold_init(verb_levl='DEBUG')

    # test w/ <OasDataset>
    n_seqs_dict = defaultdict(int)
    dataset = OasDataset(data_dir_csv, pool_size=512, n_seqs_max=256)  # reduced to 1/16
    for inputs in dataset:
        # n_seqs_dict[inputs['spec']] += 1
        logging.info(inputs['spec'])


    # test w/ DataLoader built from <OasDataset> (batch size: 16)
    data_loader = DataLoader(dataset, batch_size=16, collate_fn=lambda x: x)
    for inputs in tqdm(data_loader):
        inspect_data(inputs, name='inputs')

if __name__ == '__main__':
    main()
