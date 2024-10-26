# (c) 2023 Sormannilab and Aubin Ramon
#
# Alignment with ANARCI and cleaning of sequences. 
#
# ============================================================================

import sys 
from .mybio import Anarci_alignment, get_SeqRecords, get_antibodyVD_numbers, clean_anarci_alignment


def anarci_alignments_of_Fv_sequences(fp_seq_not_align_list, seed: bool = True, dont_change_header: bool = True, scheme: str = 'AHo', isVHH: bool = False, clean: bool = True, minimum_added: int = 50, nb_N_gaps: int=1,
                                      add_C_term_missing_VH: str = 'SS', add_C_term_missing_VKappa: str = 'K', add_C_term_missing_VLambda: str = 'L', check_duplicates: bool = False, del_cyst_misalign=False,  verbose: bool=True) :
    '''
    Align sequences from a fasta file using the ANARCI program available from https://github.com/oxpig/ANARCI

    Careful, in most light chains you need at least tolerate_missing_termini_residues=1 in the AHo numbering

    Parameters
    ----------
        - fp_fasta : str
            Filepath to the fasta file with the sequences to align or a single string sequence 
                e.g., 'seqs.fa' or 'QVQE...VSS'
        - seed : bool
            If True, start the numbering with a well defined seed in AHo numbering scheme to get all AHo numbers 
            and then potentially discard unusual sequences by setting dont_change_header to True in add_sequence
        - dont_change_heade: bool
            If True, discard unusual sequences based on the header of the first sequence 
        - scheme : str
            Type of numbering scheme, cleaning only supports AHo numbering 
        - isVHH : bool
            If True, will specify to the heavy chains that they are VHH sequences 
        - clean : bool
            If True, clean sequences based on custom parameters 
        - minimum_added : int
            Minimum size of sequences
        - nb_N_gaps : int
            If not None, allow nb_N_gaps consecutive gap at the N-terminal
        - add_C_term_missing_VH : str
            If not None, add the string motif if missing at the C-terminal (from posi 149 backwards) for Heavy chains
        - add_C_term_missing_VKappa : str 
            If not None, add the string motif if missing at the C-terminal (from posi 148 backwards) for Kappa chains
        - add_C_term_missing_VLambda : str 
            If not None, add the string motif if missing at the C-terminal (from posi 148 backwards) for Lambda chains
        - check_duplicates : bool
            If True, remove duplicates among the same chain type (only!)
        - del_cyst_misalign : bool
            If True, remove the misaligned cysteines sequences (should be set to False if sequence has been mutated at those cysteines positions).
            Default is False for prediction.
        - verbose: bool

    Returns
    -------
        - VH : Anarci_alignment class (see mybio)
        - failed,mischtype : list of tuples
                failed and mischtype are list of tuples like [(j,seq_name,seq,chtype),...] 
                mischtype are for chains that are not H, K, or L according to anarci.
    
    '''
    if clean and scheme!='AHo' :
        sys.stderr.write("**WARNING** in anarci_alignments_of_Fv_sequences clean requested with scheme=%s, but at present only supported for AHo - setting clean to False!"%(scheme))
        clean=False

    Fv_Sequences = fp_seq_not_align_list

    # Fv_Sequences = get_SeqRecords(fp_seq_not_align_list)

    # start run
    VH=Anarci_alignment(seed=seed,scheme=scheme,chain_type='H',isVHH=isVHH)
    VK=Anarci_alignment(seed=seed,scheme=scheme,chain_type='K')
    VL=Anarci_alignment(seed=seed,scheme=scheme,chain_type='L')

    failed, mischtype= list(), list()
    try_to_fix_misalignedCys=False # True makes it much slower and at least for VHHs has no effect on failed rate

    for j,seq in enumerate(Fv_Sequences) :
        if verbose and j%300==0 :
            _=sys.stdout.write(' anarci_alignments_of_Fv_sequences done %d of %d -> %.2lf %% (len(alH)=%d len(failed)=%d len(mischtype)=%d)\n' %(j,len(Fv_Sequences),100.*j/len(Fv_Sequences),len(VH),len(failed),len(mischtype)))
            sys.stdout.flush()
        if hasattr(seq,'seq')  : # seq record
            seq_name=seq.id
            seq=str(seq.seq)
        else :
            seq_name=str(j)
        Fv_res = get_antibodyVD_numbers(seq, scheme=scheme, full_return=True, seqname=seq_name, print_warns=False, auto_detect_chain_type=True)
        if Fv_res is None : # failed
            failed+=[(j,seq_name,seq,None)]
            continue
        for chtype in Fv_res :
            seqind_to_schemnum,schemnum_to_seqind,seqind_regions,warnings, info_dict, eval_table=Fv_res[chtype]
            if chtype=='H' :
                ok=VH.add_processed_sequence(seq, seqind_to_schemnum, seqind_regions, seq_name, minimum_added=minimum_added, dont_change_header=dont_change_header, try_to_fix_misalignedCys=try_to_fix_misalignedCys)
                if ok is None or ok<minimum_added :
                    failed+=[(j,seq_name,seq,chtype)]
            elif chtype=='L' :
                ok=VL.add_processed_sequence(seq, seqind_to_schemnum, seqind_regions, seq_name, minimum_added=minimum_added, dont_change_header=dont_change_header, try_to_fix_misalignedCys=try_to_fix_misalignedCys)
                if ok is None or ok<minimum_added : 
                    failed+=[(j,seq_name,seq,chtype)]
            elif chtype=='K' :
                ok=VK.add_processed_sequence(seq, seqind_to_schemnum, seqind_regions, seq_name, minimum_added=minimum_added, dont_change_header=dont_change_header, try_to_fix_misalignedCys=try_to_fix_misalignedCys)
                if ok is None or ok<minimum_added : 
                    failed+=[(j,seq_name,seq,chtype)]
            else :
                mischtype+=[(j,seq_name,seq,chtype)]
    if verbose :
        _=sys.stdout.write(' anarci_alignments_of_Fv_sequences FINISHED %d of %d -> %.2lf %% (len(alH)=%d len(failed)=%d len(mischtype)=%d)\n' %(j+1,len(Fv_Sequences),100.*(j+1)/len(Fv_Sequences),len(VH),len(failed),len(mischtype)))
        sys.stdout.flush()
    if clean :
        if len(VH)> 0 :
            if verbose: print("\n- Cleaning Heavy -")
            VH, failed_idx = clean_anarci_alignment(VH, warn=verbose,cons_cys_HD=[23,106], del_cyst_misalign=del_cyst_misalign, add_Nterm_missing=None,  add_C_term_missing=add_C_term_missing_VH, isVHH=isVHH, check_duplicates=check_duplicates, nb_N_gaps=nb_N_gaps, verbose=verbose)
            if len(failed_idx) != 0:
                for k in failed_idx:
                    failed+=[(k, None, None, None)]
        if len(VK)> 0 :
            if verbose: print("\n- Cleaning Kappa -")
            VK, failed_idx = clean_anarci_alignment(VK, warn=verbose,cons_cys_HD=[23,106],del_cyst_misalign=del_cyst_misalign, add_Nterm_missing=None,  add_C_term_missing=add_C_term_missing_VKappa, isVHH=False, check_duplicates=check_duplicates, nb_N_gaps=nb_N_gaps, verbose=verbose)
            if len(failed_idx) != 0:
                for k in failed_idx:
                    failed+=[(k, None, None, None)]
        if len(VL)> 0 :
            if verbose: print("\n- Cleaning Lambda -")
            VL, failed_idx = clean_anarci_alignment(VL, warn=verbose,cons_cys_HD=[23,106],del_cyst_misalign=del_cyst_misalign,add_Nterm_missing=None,  add_C_term_missing=add_C_term_missing_VLambda, isVHH=False, check_duplicates=check_duplicates, nb_N_gaps=nb_N_gaps, verbose=verbose)
            if len(failed_idx) != 0:
                for k in failed_idx:
                    failed+=[(k, None, None, None)]
    return VH, VK, VL, failed, mischtype
