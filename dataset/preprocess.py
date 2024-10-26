"""Build CSV files from OAS dataset, Split the """
import pickle
import os
import json
import logging
import sys
current_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_dir)

import pandas as pd
from tqdm import tqdm
import tempfile

from dataset.abnativ_alignment.align_and_clean import anarci_alignments_of_Fv_sequences
from dataset.abnativ_alignment.mybio import get_SeqRecords
from utils.anti_numbering import get_seq_list_from_SeqRecords

Chn_seqs = set()
Special_count = list()

SEG_names_dict = {
    'H': ['fwh1', 'cdrh1', 'fwh2', 'cdrh2', 'fwh3', 'cdrh3', 'fwh4'],
    'K': ['fwk1', 'cdrk1', 'fwk2', 'cdrk2', 'fwk3', 'cdrk3', 'fwk4'],
    'L': ['fwl1', 'cdrl1', 'fwl2', 'cdrl2', 'fwl3', 'cdrl3', 'fwl4'],
}

def parse_cgz_file(path, Chn_set, mouse=False, is_VHH=False, verbose=True):
    """Parse the GZ-compressed CSV file."""

    # parse the GZ-compressed CSV file
    try:
        data_frame = pd.read_csv(path, header=1, compression='gzip')
    except EOFError:
        logging.warning('corrupted GZ-compressed CSV file: %s', path)
        return []

    # obtain a list of (CSV file name, record index, heavy chain seq., light chain seq.)-tuples
    seq_list = []
    # chn_seqs = set()  # to remove duplicated sequences
    name = os.path.basename(path).replace('.csv.gz', '')
    for row_data in tqdm(data_frame.itertuples(), total=len(data_frame)):
        # heavy chain
        if row_data.locus_heavy == 'L' or row_data.locus_heavy == 'K':
            continue
        else:
            try:
                pos_dict = HEAVY_POSITIONS_dict
                cdr_index = HEAVY_CDR_INDEX
                seg_names = SEG_names_dict[row_data.locus_heavy]
                chn_seq_heavy = row_data.sequence_alignment_aa_heavy
                anarci_outputs = json.loads(row_data.ANARCI_numbering_heavy.replace('\'', '"'))
                seg_seqs_hc = [''.join(anarci_outputs[x].values()) for x in seg_names]
                assert ''.join(seg_seqs_hc) in chn_seq_heavy
                chn_seq_hc = ''.join(seg_seqs_hc)
                # if '1' in [i.strip() for i in anarci_outputs[seg_names[0]].keys()]:
                # This was consider at the human pair, and not consider at the mouse pair.
                if 'X' not in chn_seq_hc:
                    h_pad_initial_seg_seq = ['-'] * len(cdr_index)
                    merged_dict = {key: value for sub_dict in anarci_outputs.values()
                                    for key, value in sub_dict.items()}
                    for key, value in merged_dict.items():
                        key = key.strip()
                        pos_idx = pos_dict[key]
                        h_pad_initial_seg_seq[pos_idx] = value
                    pad_seq_hc = ''.join(h_pad_initial_seg_seq)
                    assert len(pad_seq_hc) == len(cdr_index), 'Pad has problem.'
                else:
                    print(f'heavy chain {chn_seq_heavy}')
                    raise AttributeError
                if mouse:
                    not_align_vh_seq_list = [chn_seq_hc]
                    VH, _, _, faild, mismatch = anarci_alignments_of_Fv_sequences(not_align_vh_seq_list, isVHH=is_VHH,
                                                                            verbose=verbose)
                    # Create a temporary file
                    temp = tempfile.NamedTemporaryFile(mode='w+', suffix='.fa', delete=False)
                    try:
                        print(temp.name)
                        # Write data to the temporary file
                        VH.Print(temp.name, print_header_definition_files=False)
                        # Read the data back from the temporary file
                        vh_seq_info = get_SeqRecords(temp.name)
                        vh_seq_list = get_seq_list_from_SeqRecords(vh_seq_info)
                    finally:
                        # Delete the temporary file
                        os.unlink(temp.name)
                    if len(vh_seq_list) != 0:
                        aho_pad_vh_seq = vh_seq_list[0]
                        print(aho_pad_vh_seq)
                    else:
                        raise ValueError(f'Heavy Aho seq has problem!')

                # else:
                #     print(f'heavy chain {chn_seq_heavy}')
                #     raise AttributeError

            except:
                print('H Length is special.')
                continue

        # light chain
        if row_data.locus_light == 'H':
            continue
        else:
            try:
                pos_dict = LIGHT_POSITIONS_dict
                cdr_index = LIGHT_CDR_INDEX
                seg_names = SEG_names_dict[row_data.locus_light]
                chn_seq_light = row_data.sequence_alignment_aa_light
                anarci_outputs = json.loads(row_data.ANARCI_numbering_light.replace('\'', '"'))
                seg_seqs_lc = [''.join(anarci_outputs[x].values()) for x in seg_names]
                assert ''.join(seg_seqs_lc) in chn_seq_light
                chn_seq_lc = ''.join(seg_seqs_lc)  # remove redundanat leading & trailing AAs
                # if '1' in [i.strip() for i in anarci_outputs[seg_names[0]].keys()]:
                if 'X' not in chn_seq_lc:
                    l_pad_initial_seg_seq = ['-'] * len(cdr_index)
                    merged_dict = {key: value for sub_dict in anarci_outputs.values()
                                    for key, value in sub_dict.items()}
                    for key, value in merged_dict.items():
                        key = key.strip()
                        pos_idx = pos_dict[key]
                        l_pad_initial_seg_seq[pos_idx] = value
                    pad_seq_lc = ''.join(l_pad_initial_seg_seq)
                    assert len(pad_seq_lc) == len(cdr_index), 'Pad has problem.'
                    if mouse: 
                        not_align_vl_seq_list = [chn_seq_lc]
                        _, VK, VL, faild, mismatch = anarci_alignments_of_Fv_sequences(not_align_vl_seq_list, isVHH=is_VHH,
                                                                                verbose=verbose)
                        # Create a temporary file
                        temp = tempfile.NamedTemporaryFile(mode='w+', suffix='.fa', delete=False)
                        try:
                            print(temp.name)
                            # Write data to the temporary file
                            if row_data.locus_light == 'K':
                                VK.Print(temp.name, print_header_definition_files=False)
                            else:
                                VL.Print(temp.name, print_header_definition_files=False)
                            # Read the data back from the temporary file
                            vl_seq_info = get_SeqRecords(temp.name)
                            vl_seq_list = get_seq_list_from_SeqRecords(vl_seq_info)
                        finally:
                            # Delete the temporary file
                            os.unlink(temp.name)
                        if len(vl_seq_list) != 0:
                            aho_pad_vl_seq = vl_seq_list[0]
                            print(aho_pad_vl_seq)
                        else:
                            raise ValueError(f'Light Aho seq has problem!')
                        
                else:
                    print(f'light chain: {chn_seq_light}')
                    raise AttributeError
                # else:
                #     print(f'light chain: {chn_seq_light}')
                #     raise AttributeError
            except:
                continue

        # record the current data entry
        if (chn_seq_hc, chn_seq_lc) not in Chn_set:  # and len(chn_seq_hc) <= length:
            Chn_set.add((chn_seq_hc, chn_seq_lc))
            type_hc, type_lc = row_data.locus_heavy, row_data.locus_light
            if mouse:
                seq_list.append((name, chn_seq_hc, chn_seq_lc,
                             pad_seq_hc, pad_seq_lc,
                             aho_pad_vh_seq, aho_pad_vl_seq,
                             type_hc, type_lc))
                print('-----------')
                print(aho_pad_vh_seq)
                print(aho_pad_vl_seq)
            else:
                seq_list.append((name, chn_seq_hc, chn_seq_lc,
                             pad_seq_hc, pad_seq_lc,
                             type_hc, type_lc))

    return seq_list, Chn_set


HEAVY_POSITIONS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
       '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
       '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
       '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56',
       '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
       '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78',
       '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
       '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100',
       '101', '102', '103', '104', '105', '106', '107', '108', '109',
       '110', '111', '111A', '111B', '111C', '111D', '111E', '111F',
       '111G', '111H', '111I', '111J', '111K', '111L', '112L', '112K',
       '112J', '112I', '112H', '112G', '112F', '112E', '112D', '112C',
       '112B', '112A', '112', '113', '114', '115', '116', '117', '118',
       '119', '120', '121', '122', '123', '124', '125', '126', '127',
       '128']

HEAVY_POSITIONS_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9, '11': 10,
                        '12': 11, '13': 12, '14': 13, '15': 14, '16': 15, '17': 16, '18': 17, '19': 18, '20': 19,
                        '21': 20, '22': 21, '23': 22, '24': 23, '25': 24, '26': 25, '27': 26, '28': 27, '29': 28,
                        '30': 29, '31': 30, '32': 31, '33': 32, '34': 33, '35': 34, '36': 35, '37': 36, '38': 37,
                        '39': 38, '40': 39, '41': 40, '42': 41, '43': 42, '44': 43, '45': 44, '46': 45, '47': 46,
                        '48': 47, '49': 48, '50': 49, '51': 50, '52': 51, '53': 52, '54': 53, '55': 54, '56': 55,
                        '57': 56, '58': 57, '59': 58, '60': 59, '61': 60, '62': 61, '63': 62, '64': 63, '65': 64,
                        '66': 65, '67': 66, '68': 67, '69': 68, '70': 69, '71': 70, '72': 71, '73': 72, '74': 73,
                        '75': 74, '76': 75, '77': 76, '78': 77, '79': 78, '80': 79, '81': 80, '82': 81, '83': 82,
                        '84': 83, '85': 84, '86': 85, '87': 86, '88': 87, '89': 88, '90': 89, '91': 90, '92': 91,
                        '93': 92, '94': 93, '95': 94, '96': 95, '97': 96, '98': 97, '99': 98, '100': 99, '101': 100,
                        '102': 101, '103': 102, '104': 103, '105': 104, '106': 105, '107': 106, '108': 107, '109': 108,
                        '110': 109, '111': 110, '111A': 111, '111B': 112, '111C': 113, '111D': 114, '111E': 115,
                        '111F': 116, '111G': 117, '111H': 118, '111I': 119, '111J': 120, '111K': 121, '111L': 122,
                        '112L': 123, '112K': 124, '112J': 125, '112I': 126, '112H': 127, '112G': 128, '112F': 129,
                        '112E': 130, '112D': 131, '112C': 132, '112B': 133, '112A': 134, '112': 135, '113': 136,
                        '114': 137, '115': 138, '116': 139, '117': 140, '118': 141, '119': 142, '120': 143,
                        '121': 144, '122': 145, '123': 146, '124': 147, '125': 148, '126': 149, '127': 150, '128': 151}

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
HEAVY_CDR_INDEX_NO_TAIL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4]

# Except tail, also need to consider vernier.
# vernier is 5.
HEAVY_CDR_KABAT_VERNIER = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        5, 5, 5, 
                        2,
                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
                        2, 2, 2, 2, 2, 2, 2, 2, 2,
                        0, 5, 0, 5, 0, 5, 0, 5, 0, 0, 0, 0, 5, 
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4]

HEAVY_CDR_KABAT_NO_VERNIER = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 
                            2,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
                            2, 2, 2, 2, 2, 2, 2, 2, 2,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4]



INPAINT_HEAVY_CDR_INDEX = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 4, 0, 4, 0, 0, 2,
                           2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                           2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

LIGHT_POSITIONS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
       '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
       '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
       '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', 
       '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
       '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78',
       '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
       '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100',
       '101', '102', '103', '104', '105', '106', '107', '108', '109',
       '110', '111', '111A', '111B', '111C', '111D', '111E', '111F',
       '112F', '112E', '112D', '112C', '112B', '112A', '112', '113',
       '114', '115', '116', '117', '118', '119', '120', '121', '122',
       '123', '124', '125', '126', '127']

LIGHT_POSITIONS_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9, '11': 10,
                        '12': 11, '13': 12, '14': 13, '15': 14, '16': 15, '17': 16, '18': 17, '19': 18, '20': 19,
                        '21': 20, '22': 21, '23': 22, '24': 23, '25': 24, '26': 25, '27': 26, '28': 27, '29': 28,
                        '30': 29, '31': 30, '32': 31, '33': 32, '34': 33, '35': 34, '36': 35, '37': 36, '38': 37,
                        '39': 38, '40': 39, '41': 40, '42': 41, '43': 42, '44': 43, '45': 44, '46': 45, '47': 46,
                        '48': 47, '49': 48, '50': 49, '51': 50, '52': 51, '53': 52, '54': 53, '55': 54, '56': 55,
                        '57': 56, '58': 57, '59': 58, '60': 59, '61': 60, '62': 61, '63': 62, '64': 63, '65': 64,
                        '66': 65, '67': 66, '68': 67, '69': 68, '70': 69, '71': 70, '72': 71, '73': 72, '74': 73,
                        '75': 74, '76': 75, '77': 76, '78': 77, '79': 78, '80': 79, '81': 80, '82': 81, '83': 82,
                        '84': 83, '85': 84, '86': 85, '87': 86, '88': 87, '89': 88, '90': 89, '91': 90, '92': 91,
                        '93': 92, '94': 93, '95': 94, '96': 95, '97': 96, '98': 97, '99': 98, '100': 99, '101': 100,
                        '102': 101, '103': 102, '104': 103, '105': 104, '106': 105, '107': 106, '108': 107, '109': 108,
                        '110': 109, '111': 110, '111A': 111, '111B': 112, '111C': 113, '111D': 114, '111E': 115,
                        '111F': 116, '112F': 117, '112E': 118, '112D': 119, '112C': 120, '112B': 121, '112A': 122,
                        '112': 123, '113': 124, '114': 125, '115': 126, '116': 127, '117': 128, '118': 129, '119': 130,
                        '120': 131, '121': 132, '122': 133, '123': 134, '124': 135, '125': 136, '126': 137, '127': 138}

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
LIGHT_CDR_INDEX_NO_TAIL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   3, 3, 3, 3, 3, 3, 3, 3, 3,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 4]

LIGHT_CDR_KABAT_VERNIER = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 
                        1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        5, 5, 5, 5,
                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                        2, 2, 2, 2, 
                        0, 0, 0, 0, 0, 0, 0, 0, 
                        5, 0, 5, 0, 0, 0, 5, 5, 0, 5, 
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                        3, 3, 3, 3, 3, 3, 3, 3, 3,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 4]

LIGHT_CDR_KABAT_NO_VERNIER = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 
                            1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                            5, 5, 5, 5,   # Here exists is because observe the situation.
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 
                            0, 0, 0, 0, 0, 0, 0, 0, 
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   # For vernier light sample.
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                            3, 3, 3, 3, 3, 3, 3, 3, 3,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 4]


AHO_CDR_INDEX   = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]