import subprocess
import torch

def get_regions(aa_seq, env_name='antidiff'):
    '''
    :param aa_seq: the sequence of antibody.
    :return: the different regions label.
    '''
    cmd_str = f'ANARCI --sequence {aa_seq}'
    complete_cmd = f'eval "$(conda shell.bash hook)" && conda activate {env_name} && {cmd_str}'
    cmd_out = subprocess.check_output(complete_cmd, shell=True)
    line_strs = cmd_out.decode('utf-8').split('\n')
    assert line_strs[2] == '# Domain 1 of 1'

    sub_strs = line_strs[5].split('|')
    chn_type = sub_strs[2]
    if chn_type == 'K':
        chn_type = 'L'
    idx_resd_beg = int(sub_strs[5])  # inclusive
    idx_resd_end = int(sub_strs[6])  # inclusive

    idx_resd = idx_resd_beg
    labl_vec = torch.zeros(len(aa_seq), dtype=torch.int8)  # 0: framework
    fv1, cdr1, fv2, cdr2, fv3, cdr3, fv4 = 0, 0, 0, 0, 0, 0, 0
    for line_str in line_strs:
        if not line_str.startswith(chn_type):
            continue
        if line_str.endswith('-'):
            continue
        idx_resd_imgt = int(line_str.split()[1])
        if idx_resd_imgt < 27:
            fv1 += 1
        elif 27 <= idx_resd_imgt <= 38:
            labl_vec[idx_resd] = 1  # CDR-1
            cdr1 += 1
        elif 38 < idx_resd_imgt < 56:
            fv2 += 1
        elif 56 <= idx_resd_imgt <= 65:
            labl_vec[idx_resd] = 2  # CDR-2
            cdr2 += 1
        elif 65 < idx_resd_imgt < 105:
            fv3 += 1
        elif 105 <= idx_resd_imgt <= 117:
            labl_vec[idx_resd] = 3  # CDR-3
            cdr3 += 1
        else:
            fv4 += 1
        idx_resd += 1
    sum_length = fv1 + fv2 + fv3 + fv4 + cdr1 + cdr2 + cdr3
    assert idx_resd == idx_resd_end + 1, f'{idx_resd} {idx_resd_beg} {idx_resd_end} {chn_type} {cmd_out}'
    # assert len(aa_seq) == sum_length, 'Acc wrong.'
    if not len(aa_seq) == sum_length:
        assert len(aa_seq) > sum_length, 'AA seq smaller than sum_length'
        fv4 += len(aa_seq) - sum_length

    true_chain_type = sub_strs[2]

    return labl_vec, [fv1, cdr1, fv2, cdr2, fv3, cdr3, fv4], true_chain_type

def get_seq_list_from_SeqRecords(seqrecord):
    seq_list = []
    for seq_info in seqrecord:
        seq = str(seq_info.seq)
        seq_list.append(seq)
    return seq_list


if __name__ == '__main__':
    seq = 'EVQLVESGGGLVQPGGSLRLSSAISGFSISSTSIDWVRQAPGKGLEWVARISPSSGSTSYADSVKGRFTISADTSKNTVYLQMNSLRAEDTAVYYTGRPLPEMGFFTQIPAMVDYRGQGTLVTVSS'
    lable = get_regions(seq)
    print(lable)