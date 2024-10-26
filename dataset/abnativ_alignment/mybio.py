# (c) 2023 Sormannilab

from Bio import SeqIO
from Bio.SeqIO import FastaIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from Bio import Align
from Bio.Align import substitution_matrices  # substitution_matrices.load("BLOSUM62")

from collections import OrderedDict
from traceback import print_exc
import os
import sys
import pickle as pickle
import time
from numpy import sqrt
import numpy
import multiprocessing

from . import misc, csv_dict, structs # plotter
from .aho_consensus import VH_consensus_no_gaps, VH_conservation_index, VL_consensus_no_gaps, VL_conservation_index, VKappa_consensus_no_gaps
from .aho_consensus import VKappa_conservation_index, VLambda_consensus_no_gaps, VLambda_conservation_index, VHH_consensus_no_gaps, VHH_conservation_index
from .blossum import blosum_alph, distance_blosum_normalised

from traceback import print_exc



def subset_sum(numbers, target, partial=[], partial_sum=0):
    """
    find all combinations of numbers that sum to target
    """
    if partial_sum == target:
        yield partial
    if partial_sum >= target:
        return
    for i, n in enumerate(numbers):
        remaining = numbers[i + 1 :]
        yield from subset_sum(remaining, target, partial + [n], partial_sum + n)


def subset_sum_with_repeat(numbers, target, partial=[], partial_sum=0):
    """
    find all combinations of numbers (including repeats) that sum to target
    largely untested
    """
    if partial_sum == target:
        yield partial
    if partial_sum >= target:
        return
    for i, n in enumerate(numbers):
        remaining = numbers[i:]
        yield from subset_sum_with_repeat(
            remaining, target, partial + [n], partial_sum + n
        )


amino_list1 = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

ThreeToOne = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "ASN": "N",
    "GLN": "Q",
    "LYS": "K",
    "THR": "T",
    "PRO": "P",
    "HIS": "H",
    "PHE": "F",
    "ALA": "A",
    "GLY": "G",
    "ILE": "I",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}  # cro is cys proline (no sometimes it is chromophore or similar)


def three_to_one(amino3letter, quiet=True, unk="X"):
    if amino3letter in ThreeToOne:
        return ThreeToOne[amino3letter]
    else:
        if not quiet:
            sys.stderr.write(
                "**WARN** in amino3letter() amino acid %s not standard. Set to %s\n"
                % (amino3letter, unk)
            )
        return unk


# untested
def NucleotideToOne(nucleotide_name, quiet=True, unk="N"):
    if "U" in nucleotide_name:
        return "U"
    if "G" in nucleotide_name:
        return "G"
    if "T" in nucleotide_name:
        return "T"
    if "C" in nucleotide_name:
        return "C"
    if "A" in nucleotide_name:
        return "A"
    if not quiet:
        sys.stderr.write(
            "**WARN** in NucleotideToOne() nucleotide %s not standard. Set to %s\n"
            % nucleotide_name,
            str(unk),
        )
    return unk




def tcoffee_alignment(
    input_sequences_or_file,
    sequence_ids=None,
    outputfilename=None,
    executable_name="t_coffee",
    accurate=False,
    do_analysis=False,
    quiet=False,
    ncpu=None,
):
    """
    perform a multiple sequence alignment using default t_coffee
     - executable_name should also contain absolute path if not installed globally (i.e. added to the PATH)
    input can be a fasta file or a list of SeqRecords, also a list of str can be given (in this case ids are read from sequence_ids, if not given
      sequences are named as seq_1 and so on).
    This function will use all avaialable cores
    return aligned_seqs (SeqRecords from clustal file) but if do_analysis:
    return aligned_seqs, pssm, aa_freq, aa_freq_ignoring_gaps, gap_freqs, consensus,key_seq
      in this case consider also
      plotter.plot_pssm_of_seq(None,pssm, plot_only_top=5)
    accurate=True improve accuracy but makes it very slow
    USEFUL tools can be found at http://www.tcoffee.org/Documentation/t_coffee/t_coffee_tutorial.htm
    in particular pairwise identities can be obtained with:
    t_coffee -other_pg seq_reformat -in sample_aln1.aln -output sim > output.dat
    and the 'triangle' matrix can be extracted with
    grep "TOP" output.dat
    """
    delete_iput = None
    if (
        not type(input_sequences_or_file) is str
    ):  # we need to print the sequences in a temporary input file
        if outputfilename is None:
            outputfilename = "tcoffee_output.aln"
        delete_iput = "tmp_tcoffee_sequences.fasta"
        if (
            type(input_sequences_or_file[0]) is str
        ):  # sequences given as a list of strings, converting to SeqRecords
            if sequence_ids is None:
                sequence_ids = [
                    "Seq_%d" % (i) for i in range(1, len(input_sequences_or_file) + 1)
                ]
            seq_recs = [
                SeqRecord(seq=Seq(se), id=sequence_ids[j], name=sequence_ids[j])
                for j, se in enumerate(input_sequences_or_file)
            ]
        elif isinstance(input_sequences_or_file[0], SeqRecord):
            seq_recs = input_sequences_or_file
        else:
            raise Exception(
                " in tcoffee_alignment() input format not recognised [accepted are fasta file, list of SeqRecord (not Seq) or list of str!\n"
            )
        print_SeqRecords(seq_recs, delete_iput)
        inpf = delete_iput
    elif os.path.isfile(input_sequences_or_file):
        inpf = input_sequences_or_file  # fasta file given as input
    else:
        raise Exception(
            " in tcoffee_alignment() input format not recognised [accepted are fasta file, list of SeqRecord (not Seq) or list of str!\n"
        )
    if ncpu is None:
        ncpu = multiprocessing.cpu_count()
    command = "nice -n 19 %s %s -n_core=%d" % (executable_name, inpf, ncpu)
    if outputfilename is None:
        outputfilename = (
            get_file_path_and_extension(inpf)[1] + ".aln"
        )  # this file is automatically generated if no specific output name is given
    else:
        command += " -outfile %s" % (outputfilename)
    if accurate:
        command += " -mode expresso"
    if quiet:
        command += " -quiet"
    else:
        print(("RUNNING:", command))
    os.system(command)
    if delete_iput is not None:
        os.remove(delete_iput)
    aligned_seqs = get_SeqRecords(outputfilename, file_format="clustal")
    # analysis
    if do_analysis:
        (
            pssm,
            aa_freq,
            aa_freq_ignoring_gaps,
            gap_freqs,
            consensus,
            key_seq,
        ) = get_pssm_fast(
            aligned_seqs, key_id=None
        )  # any non standard amino acid or non '-' is completely ingnored, key_seq is None with key_id None
        # consider also mybio.alignment_correlations_at_position(records,index,key_id=None) , which will provide the subset of alignment with given residue at given position
        return (
            aligned_seqs,
            pssm,
            aa_freq,
            aa_freq_ignoring_gaps,
            gap_freqs,
            consensus,
            key_seq,
        )
    return aligned_seqs




def pairwise_alignment(
    sequence1,
    sequence2,
    matrix_or_function=substitution_matrices.load("BLOSUM62"),
    penalty_to_open_gap=-11.0,
    penalty_to_extend_gap=-1.0,
    local=False,
    calculate_identity_percentage=False,
    identity_percentage_on_shortest_sequence=False,
    calculate_ngaps_in_shorter=False,
):
    """
    it does a pairwise alignment of two sequences
    it RETURNS a list of tuples with [ (seq1_aligned, seq2_aligned , score ,alignment_begin, alignment_end) , another_possible_outcome ]
     if calculate_identity_percentage==True alignment_begin (0 for global alignment) is replaced by the identity percentage
     if calculate_ngaps_in_shorter replace alignment_end with number of gaps in shortest sequence
    matrix_or_function:
     if matrix_or_function is None then identical characters have score of 1, otherwise 0.
     the default matrix blosum62 works only for proteins, one can give None, in which case 1. is given to matching pairs and 0. to non-matching
     the matrix object is actually a dictionary such as {('B', 'N'): 3, ('W', 'L'): -2, ... where the tuples are the pair of amino acids so it is very easy to create a custom matrix
    an example of output is [('ACCGT', 'AC-G-', 3.0, 0, 5), ('ACCGT', 'A-CG-', 3.0, 0, 5)]
    default values as suggested in Pearson (1995) "Comparison of Methods for searching protein sequence databases" Prot. Sci. 4:1145-1160.
    consider using print mybio.print_pairwise(alignment) to get a pretty representation
    if you use as the identity function mybio.identity we suggest you set matrix=mybio.identity, penalty_to_open_gap=-0.5, penalty_to_extend_gap=-0.1
    this choice of parameters yields a more 'normalized' score, whose max is zero for identical sequences, which can be used for sequence similarity comparison
      (tune your gap_parameters or use also calculate_identity_percentage)
    if identity_percentage_on_shortest_sequence is False Percent identity is calculated by multiplying the number of matches in the pair by 100 and dividing by the length of the aligned region, including gaps.
    else dividing by the length of the shortest sequence.
    """
    if matrix_or_function is None:
        if local:
            aligner = Align.PairwiseAligner()
            aligner.mode = 'local'
            aligner.open_gap_score = penalty_to_open_gap
            aligner.extend_gap_score = penalty_to_extend_gap
            if len(sequence1) > len(sequence2):
                alignments = aligner(sequence1, sequence2)
                alignment = sorted(alignments)[0]
            else:
                alignments = aligner(sequence2, sequence1)
                alignment = sorted(alignments)[0]
        else:
            aligner = Align.PairwiseAligner()
            aligner.mode = 'global'
            aligner.open_gap_score = penalty_to_open_gap
            aligner.extend_gap_score = penalty_to_extend_gap
            alignments = aligner(sequence1, sequence2)
            alignment = sorted(alignments)[0]

    else:
        if local:
            aligner = Align.PairwiseAligner()
            aligner.mode = 'local'
            aligner.substitution_matrix = matrix_or_function
            aligner.open_gap_score = penalty_to_open_gap
            aligner.extend_gap_score = penalty_to_extend_gap

            if len(sequence1) > len(sequence2):
                alignments = aligner(sequence1, sequence2)
                alignment = sorted(alignments)[0]
            else:
                alignments = aligner(sequence2, sequence1)
                alignment = sorted(alignments)[0]
        else:
    
            aligner = Align.PairwiseAligner()
            aligner.mode = 'global'
            aligner.substitution_matrix = matrix_or_function
            aligner.open_gap_score = penalty_to_open_gap
            aligner.extend_gap_score = penalty_to_extend_gap
            alignments = aligner(sequence1, sequence2)
            alignment = sorted(alignments)[0]

    if calculate_identity_percentage:
        if identity_percentage_on_shortest_sequence:
            alignment = get_identity_percentage(
                alignment, len(sequence1), len(sequence2)
            )
        else:
            alignment = get_identity_percentage(alignment, None, None)
    if calculate_ngaps_in_shorter:
        if len(sequence1) < len(sequence2):
            i = 0
        else:
            i = 1
        for k, al in enumerate(alignment):
            ngaps = len([s for s in al[i].split("-") if s]) - 1
            alignment[k] = al[:4] + (
                ngaps,
            )  # replace length of alignemnt with number of gaps in shortest sequence
    return alignment


def get_identity_percentage(
    alignment_list_or_tuple, len_seq1, len_seq2, lower_case_when_non_matching=False
):
    """
    returns the identity percentage as the fraction of identical amino acids in the shorter sequence
    if len_seq1 or len_seq2 are None than it is in the alignment length that is at the denominator
    Example:
      A-BCD-
      | | |
      AEB-DF
      len_seq1=4<len_seq2=5 ==> identity=100.*3/4
      len_seq1=None ==> identity=100.*3/6
    """
    if len_seq1 is None or len_seq2 is None:
        den = None
    elif len_seq1 <= len_seq2:
        den = len_seq1
    else:
        den = len_seq2
    if type(alignment_list_or_tuple) is list:
        for k, al in enumerate(alignment_list_or_tuple):
            identity = 0.0
            for j in range(0, len(al[0])):
                if (
                    al[0][j] == al[1][j] and al[0][j] != "-"
                ):  # if they are identical and they are not gaps
                    identity += 1.0
                elif (
                    lower_case_when_non_matching and al[0][j] != "-" and al[1][j] != "-"
                ):
                    al[0] = al[0][:j] + al[0][j].lower() + al[0][j + 1 :]
                    al[1] = al[1][:j] + al[1][j].lower() + al[1][j + 1 :]
            if den is None:
                identity /= 1.0 * len(al[0])
            else:
                identity /= 1.0 * den
            alignment_list_or_tuple[k] = (
                al[:3] + (identity * 100.0,) + al[4:]
            )  # replace position 3 (which is zero for global alignment) with identity score
    elif type(alignment_list_or_tuple) is tuple:
        identity = 0.0
        for j in range(0, len(alignment_list_or_tuple[0])):
            if (
                alignment_list_or_tuple[0][j] == alignment_list_or_tuple[1][j]
                and alignment_list_or_tuple[0][j] != "-"
            ):  # if they are identical and they are not gaps
                identity += 1.0
            elif (
                lower_case_when_non_matching
                and alignment_list_or_tuple[0][j] != "-"
                and alignment_list_or_tuple[1][j] != "-"
            ):
                alignment_list_or_tuple[0] = (
                    alignment_list_or_tuple[0][:j]
                    + alignment_list_or_tuple[0][j].lower()
                    + alignment_list_or_tuple[0][j + 1 :]
                )
                alignment_list_or_tuple[1] = (
                    alignment_list_or_tuple[1][:j]
                    + alignment_list_or_tuple[1][j].lower()
                    + alignment_list_or_tuple[1][j + 1 :]
                )
        if den is None:
            identity /= 1.0 * len(alignment_list_or_tuple[0])
        else:
            identity /= 1.0 * den
        alignment_list_or_tuple = (
            alignment_list_or_tuple[:3]
            + (identity * 100.0,)
            + alignment_list_or_tuple[4:]
        )  # replace position 3 (which is zero for global alignment) with identity score
    else:
        sys.stderr.write(
            "**WARNING** in get_identity_percentage() type of input neither tuple or list\n"
        )
    return alignment_list_or_tuple


def _alinged_seqs_to_mat_inds(aligned_seqs, RegionOfInterest_indices=None,alphabet_dict=blosum_alph) :
    '''
    auxiliary function for downstream rapid sequence distance calculations
    return mat_inds   as a numpy array
    '''
    mat_inds = [] # first covert aligned sequence into indeces of matrix_for_distance, which will be much faster in double loop.
    if RegionOfInterest_indices is not None:
        RegionOfInterest_indices = numpy.array(RegionOfInterest_indices)
        for r in aligned_seqs:
            mat_inds += [
                [
                    alphabet_dict[aa.upper()]
                    for aa in numpy.array(list(r))[RegionOfInterest_indices]
                ]
            ]  # NOTE .upper() for q and other possible guesses
    else:
        for r in aligned_seqs:
            mat_inds += [
                [alphabet_dict[aa.upper()] for aa in r]
            ]  # NOTE .upper() for q and other possible guesses
    return numpy.array(mat_inds)


def distance_between_groups_of_aligned_sequences(
    aligned_seqs1, aligned_seqs2,
    matrix_for_distance=distance_blosum_normalised,
    alphabet_dict=blosum_alph,
    RegionOfInterest_indices=None,
    only_first_N_sequences=False,
    return_number_of_mutations=False,
    quiet=False,
):
    '''
    calculates the distance between all sequences in aligned_seqs1 vs all sequences in aligned_seqs2
    return distance_similarity   OR
    return distance_similarity, distance_n_mutations if return_number_of_mutations (slower)
     if you want just the number of mutation give as matrix_for_distance=mybio._get_matrix_distance_number_of_mutations()
     and only number of mutation will be returned (about 2x faster than returning both)
    returns a matrix of shape len(aligned_seqs1), len(aligned_seqs2)
    should be slightly faster when len(aligned_seqs2) > len(aligned_seqs1)
    example of speed (on M2 macbook air with no fan):
       distance matrix  computed for 1001 vs. 2000000 sequences [only_first_N_sequences=False], took 2078.2 s
       on circe single core (took 20% of RAM):
       distance matrix  computed for 10000 vs. 2000000 sequences [only_first_N_sequences=False], took 28674.9 s (about 8 hours)
       distance matrix  computed for 10000 vs. 2000000 sequences [only_first_N_sequences=False], took 22688.9 s
    '''
    sta=time.time()
    # first covert aligned sequence into indeces of matrix_for_distance, which will be much faster in double loop.
    mat_inds1 = _alinged_seqs_to_mat_inds(aligned_seqs1, RegionOfInterest_indices=RegionOfInterest_indices,alphabet_dict=alphabet_dict) 
    mat_inds2 = _alinged_seqs_to_mat_inds(aligned_seqs2, RegionOfInterest_indices=RegionOfInterest_indices,alphabet_dict=alphabet_dict) 
    
    N1 = len(mat_inds1)
    N2 = len(mat_inds2)
    distance_similarity = numpy.empty((N1,N2))  # allocate your memory
    if not return_number_of_mutations:
        #distance_similarity = (matrix_for_distance[ mat_inds1, mat_inds2 ]) # doen't work unless you amplify mat_inds2 to len of mat_inds1 but perhaps RAM cost outweighs CPU gain
        for j in range(N1):
            distance_similarity[j] = (matrix_for_distance[ mat_inds1[j], mat_inds2 ]).sum(
                axis=1
            )  # mat.shape[1] is the length of the aligned sequences, so we sum over all residue distances (will be 0 for identical residues as that is no distance)
            
            if only_first_N_sequences and j + 1 >= only_first_N_sequences:
                distance_similarity = distance_similarity[: j+1]
                break
        if not quiet:
            print(
                "distance matrix  computed for %d vs. %d sequences [only_first_N_sequences=%s], took %.1lf s"
                % (
                    N1,N2,
                    str(only_first_N_sequences),
                    (time.time() - sta),
                )
            ) 
        sys.stdout.flush()
        return distance_similarity
    else : #if return_number_of_mutations:
        # initialise the matrix for the distance of N mutations
        distance_n_mutations = numpy.empty((N1,N2))
        # get the corresponding matrix distance
        number_of_mutations = _get_matrix_distance_number_of_mutations(alphabet_dict) 
        for j in range(N1):
            distance_similarity[j] = (matrix_for_distance[ mat_inds1[j], mat_inds2 ]).sum(axis=1)  
            # mat.shape[1] is the length of the aligned sequences, so we sum over all residue distances (will be 0 for identical residues as that is no distance)
           
            distance_n_mutations[j] = (number_of_mutations[mat_inds1[j], mat_inds2 ]).sum(axis=1)
            if only_first_N_sequences and j + 1 >= only_first_N_sequences:
                distance_similarity = distance_similarity[: j+1]
                distance_n_mutations = distance_n_mutations[: j+1]
                break
        if not quiet:
            print(
                "distance matrix and distance_n_mutations computed for %d vs. %d sequences [only_first_N_sequences=%s], took %.1lf s"
                % (
                    N1,N2,
                    str(only_first_N_sequences),
                    (time.time() - sta),
                )
            ) 
        sys.stdout.flush()
        return distance_similarity, distance_n_mutations

def _get_matrix_distance_number_of_mutations(alphabet_dict=blosum_alph):
    '''
    return number_of_mutations   a squared distance of shape (len(alphabet_dict),len(alphabet_dict))
    that can be used for the calculation of the number of mutation between two sequences
    '''
    number_of_mutations = numpy.ones((len(alphabet_dict),len(alphabet_dict)))
    numpy.fill_diagonal(number_of_mutations, 0)
    if "X" in alphabet_dict: # 'X' vs 'X' is still considered a mutation
        number_of_mutations[alphabet_dict["X"], alphabet_dict["X"]] = 1
    return number_of_mutations


def pairwise_distance_similarity_from_aln_sequences(
    aln_seqs,
    matrix=substitution_matrices.load("BLOSUM62"),
    normalise_to_same_diagonal=True,
    same_diagonal_score=10,
    both_gaps_score=1,
    one_gap_score=0,
):
    """
    SUPERSEDED by mybio.distance_matrix_from_aligned_sequences
    ISSUE now positions with gaps in both sequences under scrutiny are considered in the scores
        available matrices:
        BENNER22  BENNER6  BENNER74  BLOSUM45  BLOSUM50  BLOSUM62  BLOSUM80  BLOSUM90  DAYHOFF  FENG  GENETIC  GONNET1992
        HOXD70  JOHNSON  JONES  LEVIN  MCLACHLAN  MDM78  NUC.4.4  PAM250  PAM30  PAM70  RAO  RISLER  SCHNEIDER  STR  TRANS
        or can just type 'identity'
        same_diagonal_score has been checked with blosum62 and 10 is the one that retains most information
    """
    if matrix == "identity":
        alphabet = "ARNDCQEGHILKMFPSTWYVBZX*-"
        matrix = numpy.zeros((len(alphabet), len(alphabet))).astype(int)
        for j, aa in enumerate(alphabet):
            matrix[j, j] = 1
    # BLOSUM62=substitution_matrices.load('BLOSUM62') # ARNDCQEGHILKMFPSTWYV BZX*
    else:
        alphabet = matrix.alphabet
    # get index position of amino acids in matrix for later conversion
    alph = {}
    for j, aa in enumerate(alphabet):
        alph[aa] = j
    matrix = numpy.array(matrix)
    # do dirty correction to set all diagonal elements to be the same, so that identity of identical is always max
    if normalise_to_same_diagonal:
        mm = matrix / numpy.abs(numpy.diagonal(matrix)) * 10
        matrixN = numpy.round((mm + mm.T) / 2, 0).astype(
            int
        )  # (mm+mm.T)/2 # normalised with all diagonal element sets to be the same then re-symmetrized. UGLY
    else:
        matrixN = matrix
    if both_gaps_score is not None:
        matrixN[alph["*"], :] = one_gap_score
        matrixN[:, alph["*"]] = one_gap_score
        matrixN[alph["*"], alph["*"]] = both_gaps_score
    # convert sequences to numpy array, and in particular to indices in matrix for faster calculations
    mat = []
    mat_inds = []
    if hasattr(aln_seqs[0], "seq"):
        for j, r in enumerate(aln_seqs):
            mat += [numpy.array(list(r.seq))]
            mat_inds += [[alph["*"] if aa == "-" else alph[aa] for aa in r.seq]]
    else:
        for j, r in enumerate(aln_seqs):
            mat += [numpy.array(list(r))]
            mat_inds += [[alph["*"] if aa == "-" else alph[aa] for aa in r]]
    mat = numpy.array(mat)
    mat_inds = numpy.array(mat_inds)
    N = len(mat_inds)
    Max_score = (
        matrixN[0, 0] * mat_inds.shape[1]
    )  # possible max score assuming that all diagonal elements are the same
    sta = time.time()
    distance_similarity = numpy.empty((N * (N - 1)) // 2)  # allocate your memory
    n = 0
    for j in range(len(mat_inds) - 1):
        distance_similarity[n : n + N - 1 - j] = Max_score - (
            matrixN[mat_inds[j + 1 :], mat_inds[j]]
        ).sum(
            axis=1
        )  # mat.shape[1] is the length of the aligned sequences
        n += N - 1 - j
    print(" took", time.time() - sta)
    return distance_similarity


def get_pairwise_percent_identity(
    seq1_al, seq2_al, exclude_matching_gaps=True, match_with_gap=0
):
    """
        match_with_gap should be in 0,1 if 0 gap in one seq but not in other is considered like mismatch if 1 like match
    NB get_pairwise_blosum_identity does a sort of blosum62 identity...
    """
    identity, den = 0, 0
    for j, aa in enumerate(seq1_al):
        if aa == seq2_al[j]:
            if aa == "-" and exclude_matching_gaps:
                continue  # don't add to den
            identity += 1
        elif aa == "-" or seq2_al[j] == "-":
            identity += match_with_gap
        den += 1
    # if exclude_matching_gaps :
    #    le1,le2=len(seq1_al.replace('-','')),len(seq2_al.replace('-',''))
    #    if le1<le2:den=float(le1)
    #    else : den=float(le2)
    # else : float(len(seq1_al)) # length of alignment
    return 100.0 * identity / den



def pairwise_identity_mat_from_aln_record(
    records,
    do_100_minus=False,
    sequence_identity_function=get_pairwise_percent_identity,
):
    """
    largely SUPERSEDED by pairwise_distance_similarity_from_aln_sequences (at least for large alignments)
    given a list of aligned SeqRecords it computes a sequence identity matrix
    """
    if type(records) is str and os.path.isfile(records):
        records = get_SeqRecords(records)
    index_to_name, name_to_index = {}, {}
    matrix = matrix = numpy.zeros((len(records), len(records)))
    for j, r in enumerate(records):
        s1 = str(r.seq)
        index_to_name[j] = r.id
        if r.id in name_to_index:
            print(
                "warning in pairwise_identity_mat_from_aln_record() overwriting %s in name_to_index"
                % (r.id)
            )
        name_to_index[r.id] = j
        # safety let us calculate the identity also with iself (100%)
        for i in range(j, len(records)):
            s2 = str(records[i].seq)
            ident = sequence_identity_function(s1, s2)
            if do_100_minus:
                ident = 100 - ident
            matrix[j][i] = ident
            matrix[i][j] = ident
    return matrix, index_to_name, name_to_index



def CDR_ranges_from_scheme(scheme):
    """
    select chothia CDR in Chimera:
    select :26-33.H :52-55.H :95-101.H
    select :24-33.L :50-55.L :89-96.L
    select :26-33.G :52-55.G :95-101.G :26-33.B :52-55.B :95-101.B :24-33.A :50-55.A :89-96.A :24-33.F :50-55.F :89-96.F
    return cdr1_scheme,cdr2_scheme,cdr3_scheme
    which look like cdr1_scheme={'H':range(26,34),'L':range(24,34)}
    """
    if scheme.lower() == "imgt":
        cdr1_scheme = {"H": list(range(27, 39)), "L": list(range(27, 39))}
        cdr2_scheme = {"H": list(range(56, 66)), "L": list(range(56, 66))}
        cdr3_scheme = {"H": list(range(105, 118)), "L": list(range(105, 118))}
    elif scheme.lower() == "chothia":
        cdr1_scheme = {"H": list(range(26, 34)), "L": list(range(24, 34))}
        cdr2_scheme = {"H": list(range(52, 56)), "L": list(range(50, 56))}
        cdr3_scheme = {"H": list(range(95, 102)), "L": list(range(89, 97))}
    elif (
        scheme.lower() == "aho"
    ):  # 'AHo' Slightly Different from figure 4 of original paper https://ac.els-cdn.com/S0022283601946625/1-s2.0-S0022283601946625-main.pdf?_tid=c2184c42-26cc-4964-87d5-f07e1d91465d&acdnat=1523884058_a3a1fb168d06d7b535c0da3bca0df0d6
        cdr1_scheme = {"H": list(range(27, 43)), "L": list(range(27, 43))}
        cdr2_scheme = {"H": list(range(57, 70)), "L": list(range(57, 70))}
        cdr3_scheme = {"H": list(range(108, 139)), "L": list(range(108, 139))}
    else:
        raise Exception(
            "scheme %s not implemented in CDR_ranges_from_scheme\n" % (str(scheme))
        )
    return cdr1_scheme, cdr2_scheme, cdr3_scheme


def get_CDR_simple(
    sequence,
    allow=set(["H", "K", "L"]),
    auto_cdr_scheme=True,
    cdr1_scheme={"H": list(range(26, 33)), "L": list(range(24, 35))},
    cdr2_scheme={"H": list(range(52, 57)), "L": list(range(50, 57))},
    cdr3_scheme={"H": list(range(95, 103)), "L": list(range(89, 98))},
    scheme="chothia",
    seqname="",
):
    """
    used by parapred
    From a VH or VL amino acid sequences returns the three CDR sequences as determined from the input numbering (scheme) and the given ranges.
      requires the python module - Available from http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/ANARCI.php

    input cdr_scheme are IGNORED if auto_cdr_scheme is True
    For other numbering schemes see also http://www.bioinf.org.uk/abs/#cdrdef
    Loop    Kabat          AbM    Chothia1    Contact2
    L1    L24--L34    L24--L34    L24--L34    L30--L36
    L2    L50--L56    L50--L56    L50--L56    L46--L55
    L3    L89--L97    L89--L97    L89--L97    L89--L96
    H1    H31--H35B   H26--H35B   H26--H32..34  H30--H35B
    H1    H31--H35    H26--H35    H26--H32    H30--H35
    H2    H50--H65    H50--H58    H52--H56    H47--H58
    H3    H95--H102   H95--H102   H95--H102   H93--H101

    For generic Chothia identification can set auto_detect_chain_type=True and use:
    cdr1_scheme={'H':range(26,33),'L':range(24,35)}
    cdr2_scheme={'H':range(52,57),'L':range(50,57)}
    cdr3_scheme={'H':range(95,103),'L':range(89,98)}
    """
    try:
        import anarci
    except ImportError:
        raise Exception(
            "\n**ImportError** function get_CDR_simple() requires the python module anarci\n Available from http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/ANARCI.php\n\n"
        )

    if auto_cdr_scheme:
        cdr1_scheme, cdr2_scheme, cdr3_scheme = CDR_ranges_from_scheme(scheme)
    res_num_all = anarci.number(sequence, scheme=scheme, allow=allow)
    if not hasattr(res_num_all[0], "__len__"):
        print(
            "ERROR in get_CDR_simple() anarci failed on %s -returned %s chaintype=%s"
            % (seqname, str(res_num_all[0]), str(res_num_all[1]))
        )
        return None
    cdr1, cdr2, cdr3 = "", "", ""
    chain_type = res_num_all[1]
    print(seqname, "chain_type", chain_type, scheme)
    if hasattr(
        cdr1_scheme, "keys"
    ):  # supports dictionary or OrderedDict as input type - assume all cdr ranges are like this
        if chain_type == "K" and chain_type not in cdr1_scheme:
            chain_type = "L"  # Kappa light chain to Lambda light chain for this purpose
        if chain_type not in cdr1_scheme:
            raise Exception(
                "\n chain_type %s not in input cdr1_scheme\n" % (chain_type)
            )
        cdr1_scheme = cdr1_scheme[chain_type]
        cdr2_scheme = cdr2_scheme[chain_type]
        cdr3_scheme = cdr3_scheme[chain_type]
    # extract CDR sequences
    for num_tuple, res in res_num_all[0]:
        if int(num_tuple[0]) in cdr1_scheme:
            cdr1 += res
        elif int(num_tuple[0]) in cdr2_scheme:
            cdr2 += res
        elif int(num_tuple[0]) in cdr3_scheme:
            cdr3 += res
    # put in parapred formta
    cdrs = {"CDR1": cdr1, "CDR2": cdr2, "CDR3": cdr3}
    return cdrs


def DBScan_clustering(dist_mat, eps=None, min_samples=5, grid_eps=False):
    """
    grid_eps will find eps that gives max numb of clusters
    may consider HDBSCAN as a stand-alone library that find eps
    Unsupervised clustering:
    return cluster_labels, dbscan_class
    NOTE: cluster -1: corresponds to non-clusterable samples.
    Density-Based Spatial Clustering of Applications with Noise
     params
     eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        This is not a maximum bound on the distances of points within a cluster.
        This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
        None will use the 5% percentile of the dist_mat
     'metric': 'precomputed', # if input is a disctance matrix
    'min_samples':  5 is by default. The number of samples (or total weight) in a neighborhood
      for a point to be considered as a core point.
    """
    # --- Import libraries:
    from sklearn import cluster

    # Parameters to clustering algoritms:
    if grid_eps:
        m, M = numpy.percentile(dist_mat, [0.3, 10])
        nclus = []
        for e in numpy.linspace(m, M, 20):
            clus, dbs = DBScan_clustering(
                dist_mat, eps=e, min_samples=min_samples, grid_eps=False
            )
            nclus += [(max(clus) + 1, e)]
        nclus = sorted(nclus, reverse=True)
        print("best nclu,eps:", nclus[4])
        eps = nclus[0][1]
    elif eps is None:
        if dist_mat.shape[0] > 200:
            p = 1
        else:
            p = 5
        eps = numpy.percentile(dist_mat, p)
    params = {
        "eps": eps,  # The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster.
        "metric": "precomputed",  # if input is a disctance matrix
        "min_samples": min_samples,
    }  # 5 is by default. The number of samples (or total weight) in a neighborhood for a point to be considered as a core point
    # --- Clustering:
    dbscan = cluster.DBSCAN(
        eps=params["eps"], metric=params["metric"], min_samples=params["min_samples"]
    )
    # dbscan.fit_predict(dist_mat, sample_weight=None)
    dbscan.fit(dist_mat)
    cluster_labels = dbscan.labels_  # the computed categories
    # Meaning of cluster -1: corresponds to non-clusterable samples.
    return cluster_labels, dbscan


def Kmeans_clustering(
    dist_mat, n_clusters, return_centroids=False, plot=False, **plot_kwargs
):
    """
    return cluster_labels, kmeans_results_class, index_to_cluster,clusters
    if return_centroids it then
    return cluster_labels, kmeans_results_class,cluster_centroids_as_indices,cluster_size, index_to_cluster,clusters

    """
    try:
        import sklearn.cluster
    except ImportError:
        sys.stderr.write(
            "\n\n**ERROR** for mybio.Kmeans_clustering you need sklearn installed\n"
        )
        raise
    kmeans = sklearn.cluster.KMeans(n_clusters)
    results = kmeans.fit(dist_mat)
    cluster_labels = results.labels_
    if plot:
        if len(dist_mat.shape) > 1:
            N = dist_mat.shape[0]
        else:
            raise Exception(
                "**ERROR** in Kmedoids_clustering asked to plot requires a squared matrix, current shape %s\n"
                % (str(dist_mat.shape))
            )
        # sort and prepare for plot
        argsorter = numpy.argsort(cluster_labels)
        plot_mat = dist_mat[argsorter.reshape(-1, 1), argsorter]
        sorted_selected_clusters = cluster_labels[argsorter]
        clust_ids, counts = numpy.unique(
            sorted_selected_clusters, return_counts=True
        )  # check size of each cluster
        vgrid = numpy.cumsum(counts)
        # ylabels=None #[x.replace('Seq:','') for x in fl[:cut_first]] [::-1] ,ylabels=ylabels,xlabels=[x.split(':')[1] for x in fl[:cut_first]],xlabels_rotation='vertical',xlabel='Seq ID',ylabel='Seq ID : ReadCount',
        plotter.plot_matrix(
            plot_mat,
            hgrid=[len(plot_mat) - x for x in vgrid if len(plot_mat) - x > 0],
            vgrid=[x for x in vgrid if x < len(plot_mat)],
            figure_size=(9, 9),
            **plot_kwargs
        )
    index_to_cluster, clusters = {}, {}
    for j, cl in enumerate(cluster_labels):
        index_to_cluster[j] = cl
        if cl not in clusters:
            clusters[cl] = []
        clusters[cl] += [j]
    if return_centroids:
        centroid_indices = numpy.array(
            [
                (numpy.linalg.norm(dist_mat - cent, axis=1)).argmin()
                for cent in results.cluster_centers_
            ]
        )
        cluster_size = {}
        for cl in clusters:
            cluster_size[cl] = len(clusters[cl])
        cluster_centroids_as_indices = {}  # clusters are keys and centroid index values
        for j, c in enumerate(centroid_indices):
            if c not in clusters[j]:
                print("ISSUE")
            cluster_centroids_as_indices[j] = c
        return (
            cluster_labels,
            results,
            cluster_centroids_as_indices,
            cluster_size,
            index_to_cluster,
            clusters,
        )
    return cluster_labels, results, index_to_cluster, clusters


def Kmedoids_clustering(
    dist_mat, n_clusters, return_centroids=False, plot=False, **plot_kwargs
):
    """
    This python package implements k-medoids clustering with PAM. It can be used with arbitrary dissimilarites, as it requires a dissimilarity matrix as input.
    return cluster_labels, kmedoids_fasterpam_class, index_to_cluster,clusters
    if return_centroids it then
    return cluster_labels, kmedoids_fasterpam_class,cluster_centroids_as_indices,cluster_size, index_to_cluster,clusters

    """
    try:
        import kmedoids
    except ImportError:
        sys.stderr.write(
            "\n\n**ERROR** for mybio.kmedoids_clustering you need kmedoids installed - pip install kmedoids which requires rust to be intalled as a programming language!\n"
        )
        raise
    c = kmedoids.fasterpam(dist_mat, n_clusters)
    # print("Loss is:", c.loss)
    cluster_labels = c.labels
    if plot:
        if len(dist_mat.shape) > 1:
            N = dist_mat.shape[0]
        else:
            raise Exception(
                "**ERROR** in Kmedoids_clustering asked to plot requires a squared matrix, current shape %s\n"
                % (str(dist_mat.shape))
            )
        # sort and prepare for plot
        argsorter = numpy.argsort(cluster_labels)
        plot_mat = dist_mat[argsorter.reshape(-1, 1), argsorter]
        sorted_selected_clusters = cluster_labels[argsorter]
        clust_ids, counts = numpy.unique(
            sorted_selected_clusters, return_counts=True
        )  # check size of each cluster
        vgrid = numpy.cumsum(counts)
        # ylabels=None #[x.replace('Seq:','') for x in fl[:cut_first]] [::-1] ,ylabels=ylabels,xlabels=[x.split(':')[1] for x in fl[:cut_first]],xlabels_rotation='vertical',xlabel='Seq ID',ylabel='Seq ID : ReadCount',
        plotter.plot_matrix(
            plot_mat,
            hgrid=[len(plot_mat) - x for x in vgrid if len(plot_mat) - x > 0],
            vgrid=[x for x in vgrid if x < len(plot_mat)],
            figure_size=(9, 9),
            **plot_kwargs
        )
    index_to_cluster, clusters = {}, {}
    for j, cl in enumerate(cluster_labels):
        index_to_cluster[j] = cl
        if cl not in clusters:
            clusters[cl] = []
        clusters[cl] += [j]
    if return_centroids:
        cluster_size = {}
        for cl in clusters:
            cluster_size[cl] = len(clusters[cl])
        cluster_centroids_as_indices = {}  # clusters are keys and centroid index values
        centroid_indices = c.medoids
        for j, c in enumerate(centroid_indices):
            if c not in clusters[j]:
                print("ISSUE")
            cluster_centroids_as_indices[j] = c
        return (
            cluster_labels,
            c,
            cluster_centroids_as_indices,
            cluster_size,
            index_to_cluster,
            clusters,
        )
    return cluster_labels, c, index_to_cluster, clusters



def Hierarchical_clustering(
    dist_mat,
    n_clusters,
    linkage="average",
    return_centroids=False,
    plot=False,
    **plot_kwargs
):
    """
    consider linkage='complete' (also 'ward', 'complete', 'average', 'single' suggested default was 'ward' but only works with euclidean distance and not with pre-computed affinities such as a distance matrix)
    return cluster_labels, hierarchical_class, index_to_cluster,clusters
    if return_centroids it then
    return cluster_labels, clustering,cluster_centroids_as_indices,cluster_size, index_to_cluster,clusters
    where hier_centroids_indices are indices of the rows of dist_mat corresponding to the cluster centroids
      which are defined as the element of the cluster with the smallest median distance from all other elements (unless centroids_old_way=True, which does a very ugly approximation).
    if plot it plots the clustered matrix
    """
    # --- Import libraries:
    from sklearn import cluster

    # Parameters to clustering algoritms:
    params = {
        "n_clusters": n_clusters,  # The number of clusters (use dendrogram as support)
        "linkage": linkage,  # The linkage criterion
        "affinity": "precomputed",
    }  # If input is a disctance matrix
    # --- Clustering:
    clustering = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"],
        linkage=params["linkage"],
        affinity=params["affinity"],
    )
    clustering.fit(dist_mat)
    cluster_labels = clustering.labels_
    index_to_cluster, clusters = {}, {}
    for j, cl in enumerate(cluster_labels):
        index_to_cluster[j] = cl
        if cl not in clusters:
            clusters[cl] = []
        clusters[cl] += [j]
    if (
        return_centroids
    ):  # highly sub-optimal way to get potential centroids Sum the vectors in each cluster
        if len(dist_mat.shape) > 1 and dist_mat.shape[0] == dist_mat.shape[1]:
            N = dist_mat.shape[0]
        else:
            raise Exception(
                "**ERROR** in Hierarchical_clustering asked to return_centroids requires a squared matrix, current shape %s\n"
                % (str(dist_mat.shape))
            )
        # new way to get centroids
        cluster_size = {}  # will contain the lengths for each cluster
        cluster_centroids_as_indices = (
            {}
        )  # keys are cluster number, values are indices
        for clno in clusters:
            inds = numpy.array(clusters[clno]).astype(
                int
            )  # get all elements of a cluster
            cluster_size[clno] = len(inds)
            MA = dist_mat[inds]
            MA = MA[
                :, inds
            ]  # create sub-matrix of distances between cluster elements
            # print ("DEB:",clno,len(inds),MA.shape, (MA-MA.T).sum(), inds[numpy.median(MA,axis=0).argmin()],inds[numpy.mean(MA,axis=0).argmin()])
            centroid = inds[
                numpy.median(MA, axis=0).argmin()
            ]  # Centroid is defined as the elmenet with the smallest median distance from all others (could e mean but then if you have few very close but far from most that form a more spread group the centroid will likely be among the few)
            cluster_centroids_as_indices[clno] = centroid
    if plot:
        if len(dist_mat.shape) > 1:
            N = dist_mat.shape[0]
        else:
            raise Exception(
                "**ERROR** in Hierarchical_clustering asked to plot requires a squared matrix, current shape %s\n"
                % (str(dist_mat.shape))
            )
        # sort and prepare for plot
        argsorter = numpy.argsort(cluster_labels)
        plot_mat = dist_mat[argsorter.reshape(-1, 1), argsorter]
        sorted_selected_clusters = cluster_labels[argsorter]
        clust_ids, counts = numpy.unique(
            sorted_selected_clusters, return_counts=True
        )  # check size of each cluster
        vgrid = numpy.cumsum(counts)
        # ylabels=None #[x.replace('Seq:','') for x in fl[:cut_first]] [::-1] ,ylabels=ylabels,xlabels=[x.split(':')[1] for x in fl[:cut_first]],xlabels_rotation='vertical',xlabel='Seq ID',ylabel='Seq ID : ReadCount',
        fig = plotter.plot_matrix(
            plot_mat,
            hgrid=[len(plot_mat) - x for x in vgrid if len(plot_mat) - x > 0],
            vgrid=[x for x in vgrid if x < len(plot_mat)],
            figure_size=(9, 9),
            **plot_kwargs
        )
        if return_centroids:
            cent_mask = numpy.zeros(plot_mat.shape)
            cent_inds = numpy.zeros(plot_mat.shape[0]).astype(
                int
            )  # very convoluted because we need to argsort it.
            cent_inds[
                numpy.array(list(cluster_centroids_as_indices.values())).astype(int)
            ] = 1  # [still need sorting
            # print ('DEB:', numpy.array(list(cluster_centroids_as_indices.values())) ,cent_mask.shape,len(cent_inds),cent_inds[argsorter],'argsorter=',argsorter)
            cent_inds = cent_inds[argsorter].astype(bool)
            cent_mask[cent_inds, cent_inds] = 1
            m = plotter.create_color_map(
                minval=0, maxval=1, colors=[(1, 1, 1, 0), (1, 0, 0, 1)]
            )
            fig = plotter.plot_matrix(
                cent_mask,
                cmap=m,
                vmin=0.5,
                plot_colorbar=False,
                figure=fig[0],
                **plot_kwargs
            )
    if return_centroids:
        return (
            cluster_labels,
            clustering,
            cluster_centroids_as_indices,
            cluster_size,
            index_to_cluster,
            clusters,
        )
    return cluster_labels, clustering, index_to_cluster, clusters



# antibody alignments and nubmerings (rely on anarci program available from https://github.com/oxpig/ANARCI )
def get_antibodyVD_numbers(
    sequence,
    scheme="chothia",
    allow=set(["H", "K", "L", "A", "B", "G", "D"]),
    full_return=True,
    seqname="",
    auto_cdr_scheme=True,
    cdr1_scheme=list(range(27, 39)),
    cdr2_scheme=list(range(56, 66)),
    cdr3_scheme=list(range(105, 118)),
    print_warns=True,
    auto_detect_chain_type=False,
):
    """
    Other useful scheme is 'AHo'
    numbers according to the given scheme the VL or VH part of an antibody sequence
     when ran on full sequence second part will correpond to constant domain and only first part will be numbered
    input cdr_scheme are IGNORED if auto_cdr_scheme is Tru
    give a gapless sequence and
    if full_return
     return  seq_dict
     a dict whose keys are chain type, typically in ["H", "K", "L"] and values are tuples with (seqind_to_schemnum,schemnum_to_seqind,seqind_regions,warnings, info_dict, eval_table )
     see also merge_scFv_antibodyVD_numbers to merge results in sigle dictionaries:
     seqind_to_schemnum,schemnum_to_seqind,seqind_regions,warnings= mybio.merge_scFv_antibodyVD_numbers( seq_dict,sequence )
    otherwise
     return seqind_to_schemnum,schemnum_to_seqind,seqind_regions,warnings
    if seqind_to_schemnum[j] is None it means that no numbering was assigned for residue index j (probably part of constant domain)
     and if seqind_to_schemnum[j] is "N" it means that this was an N-terminus residue without numbers assigned, maybe an expression tag or similar.
     seqname can be used for debug purposes and if give is printed in potential printed warnings.
    the returned warnings may be a list of tuples with residue index and last-assigned IMGT number (j,schemnum_to_seqind.keys()[-1])
    seqind_regions is a list of the length of the sequence containing either '' (unassigned number) or 'fr' (framework) or 'CDR1','CDR2','CDR3'
     for each residue

    this function requires the module anarci
    For other numbering schemes see also http://www.bioinf.org.uk/abs/#cdrdef
    """
    try:
        import anarci
    except Exception:
        sys.stderr.write(
            "\n\n**ERROR** functions get_antibodyVD_numbers() and get_CDR() require the python module anarci\n Available from https://github.com/oxpig/ANARCI\n\n"
        )
        sys.stderr.flush()
        raise ImportError
    sequence = sequence.upper()
    if auto_cdr_scheme:
        cdr1_scheme, cdr2_scheme, cdr3_scheme = CDR_ranges_from_scheme(scheme)
    if full_return:
        nam = seqname
        if nam == "":
            nam = "input_seq"
        ok, aa_wrong = is_protein(sequence)
        if not ok:
            sys.stderr.write(
                "*Error in get_antibodyVD_numbers** Unkown amion acid '%s' in sequence %s - not porcessing\n"
                % (aa_wrong, nam)
            )
            return None
        res_num_different = anarci.anarci(
            [(nam, sequence)], scheme=scheme, allow=allow
        )  # returns more things but it is also radically different (e.g.if input is scFv sequence it numbers both heavy and light and returns two lists res_num_all[0][0][0] and res_num_all[0][0][1]  - other infos are res_num_all[1][0][0] and res_num_all[1][0][1]
        if res_num_different is None or not hasattr(res_num_different[0][0], "__len__"):
            if print_warns:
                sys.stderr.write(
                    "ERROR in get_antibodyVD_numbers anarci failed %s -returned %s\n"
                    % (seqname, str(res_num_different))
                )  # anarci failed
            return None
        elif (
            len(res_num_different[0][0]) > 1
        ):  # more than one antibody chain! (e.g. scFv sequence as input)
            print(
                "in get_antibodyVD_numbers input sequence %s has more than one Fv chain! types %s"
                % (
                    seqname,
                    str([(d["chain_type"], d["id"]) for d in res_num_different[1][0]]),
                )
            )
            # SHOULD add fix for situations where [('H', 'rat_H'), ('K', 'human_K')] and e.g. rat_H is almost as good as human_H while rat_K would make no sense (thus all antibody should be human)
            # in this format res_num_all[0]==res_num_different[0][0][0][0]
            # res_num_all[2][0] will contain both (or all chains mixed together, but distinguishable according to query_start query_end
            mutli_chain_results = OrderedDict()
            for j in range(len(res_num_different[0][0])):
                ch = res_num_different[1][0][j]["chain_type"]
                if ch in mutli_chain_results:
                    print("**WARNING** chain_type %s previously found " % (ch), end=" ")
                    ch = csv_dict.new_key_for_dict(mutli_chain_results, ch)
                    print("adding %s instead" % (ch))
                start_ind, end_ind = (
                    res_num_different[1][0][j]["query_start"],
                    res_num_different[1][0][j]["query_end"],
                )
                (
                    seqind_to_schemnum,
                    schemnum_to_seqind,
                    seqind_regions,
                    warnings,
                ) = aux_get_antibodyVD_numbers(
                    sequence,
                    (
                        res_num_different[0][0][j][0],
                        res_num_different[1][0][j]["chain_type"],
                    ),
                    cdr1_scheme,
                    cdr2_scheme,
                    cdr3_scheme,
                    seqname=seqname + ch,
                    start_ind=start_ind,
                    end_ind=end_ind,
                    auto_detect_chain_type=False,
                    print_warns=print_warns,
                )
                # get relevant lines of res_num_all[2][0] plus header
                inf2 = [res_num_different[2][0][0]] + [
                    l
                    for l in res_num_different[2][0][1:]
                    if start_ind == l[-2] and end_ind == l[-1]
                ]
                mutli_chain_results[ch] = (
                    seqind_to_schemnum,
                    schemnum_to_seqind,
                    seqind_regions,
                    warnings,
                    res_num_different[1][0][j],
                    inf2,
                )
            return mutli_chain_results
        else:
            res = {}
            res_num_all = (
                res_num_different[0][0][0][0],
                res_num_different[1][0][0]["chain_type"],
            )
            start_ind, end_ind = (
                res_num_different[1][0][0]["query_start"],
                res_num_different[1][0][0]["query_end"],
            )
            (
                seqind_to_schemnum,
                schemnum_to_seqind,
                seqind_regions,
                warnings,
            ) = aux_get_antibodyVD_numbers(
                sequence,
                res_num_all,
                cdr1_scheme,
                cdr2_scheme,
                cdr3_scheme,
                seqname=seqname,
                start_ind=start_ind,
                end_ind=end_ind,
                auto_detect_chain_type=False,
                print_warns=print_warns,
            )
            # res_num_different[1][0][0] is a dict like {'bitscore': 181.1, 'query_start': 16, 'query_end': 129, 'chain_type': 'H', 'id': 'rat_H', 'bias': 0.7, 'query_name': 'ab', 'species': 'rat', 'scheme': 'chothia', 'evalue': 1.2e-56, 'description': ''}
            # res_num_all[2][0] is a list of list where [0] is the header ['id', 'description', 'evalue', 'bitscore', 'bias', 'query_start', 'query_end'] and entries are sorted accordint to the evalues
            res[res_num_different[1][0][0]["chain_type"]] = [
                seqind_to_schemnum,
                schemnum_to_seqind,
                seqind_regions,
                warnings,
                res_num_different[1][0][0],
                res_num_different[2][0],
            ]
            return res
    else:
        res_num_all = anarci.number(sequence, scheme=scheme, allow=allow)
    return aux_get_antibodyVD_numbers(
        sequence,
        res_num_all,
        cdr1_scheme,
        cdr2_scheme,
        cdr3_scheme,
        seqname=seqname,
        auto_detect_chain_type=auto_detect_chain_type,
        print_warns=print_warns,
    )


def merge_scFv_antibodyVD_numbers(
    Fv_res, sequence, add_chain_type_to_numbers=True, force_L_instead_of_K=True
):
    """
    return seqind_to_schemnum,schemnum_to_seqind,seqind_regions,warnings
     where the list are merged results with both the light and heavy chain - if add_chain_type_to_numbers is False
     the chain type is NOT added to the values of seqind_to_schemnum, HOWEVER it is still added to the keys of schemnum_to_seqind
      as otherwise these would be rewritten
    the input Fv_res is a
     a dict whose keys are chain type, typically in ["H", "K", "L"] and values are
     tuples with (seqind_to_schemnum,schemnum_to_seqind,seqind_regions,warnings, info_dict, eval_table )
    """
    if len(Fv_res) == 1:
        (
            seqind_to_schemnum,
            schemnum_to_seqind,
            seqind_regions,
            warnings,
            info_dict,
            eval_table,
        ) = list(Fv_res.values())[0]
        return seqind_to_schemnum, schemnum_to_seqind, seqind_regions, warnings
    else:
        ks = list(Fv_res.keys())
        if len(ks) > 2:
            print(
                "**WARNING** in merge_scFv_antibodyVD_numbers found more than 2 chain types %s\n"
                % (str(ks))
            )
    seqind_to_schemnum, schemnum_to_seqind, seqind_regions = (
        OrderedDict(),
        OrderedDict(),
        [],
    )
    found_one = False
    warned = False
    for j, a in enumerate(sequence):
        schemnum, region = None, ""
        for k in ks:
            if "00" in k:
                if not warned:
                    sys.stderr.write(
                        "\n**WARNING** in merge_scFv_antibodyVD_numbers found k named %s among %s - possilby these are VHH or VLs in tandem! SKIPPING second chain!"
                        % (k, str(list(ks)))
                    )
                    warned = True
                region = ""
                schemnum = None
                if Fv_res[k][0][j] not in ["N", None]:  # seqind_to_schemnum
                    found_one = True
            else:
                if Fv_res[k][0][j] not in ["N", None]:  # seqind_to_schemnum
                    found_one = True
                    if schemnum is None:
                        if add_chain_type_to_numbers:
                            ch = k
                            if force_L_instead_of_K and k == "K":
                                ch = "L"
                            schemnum = (
                                ch + ":" + str(Fv_res[k][0][j][0]),
                                Fv_res[k][0][j][1],
                            )  # tuples like (1, ' ') [first res of H chain when H is after L]
                        else:
                            schemnum = Fv_res[k][0][j]
                    else:
                        print(
                            "**Warn in merge_scFv_antibodyVD_numbers() for residue %s at seq index %d - schemnum previoulsy set to %s but chain type %s suggests %s - leaving unchanged"
                            % (a, j, str(schemnum), k, Fv_res[k][0][j])
                        )
                        # if add_chain_type_to_numbers :  schemnum=( k+':'+str(Fv_res[k][0][j][0]),Fv_res[k][0][j][1]) # tuples like (1, ' ') [first res of H chain when H is after L]
                        # else : schemnum=Fv_res[k][0][j]
                if Fv_res[k][2][j] != "":  # seqind_regions
                    if region == "":
                        region = k + ":" + Fv_res[k][2][j]
                    else:
                        print(
                            "**Warn in merge_scFv_antibodyVD_numbers() for residue %s at seq index %d - region previoulsy set to %s but chain type %s suggests %s - leaving unchanged"
                            % (a, j, region, k, Fv_res[k][2][j])
                        )

        seqind_regions += [region]
        if schemnum is None and not found_one:
            schemnum = "N"
        seqind_to_schemnum[j] = schemnum
        if schemnum not in ["N", None]:
            if (
                "00" in k
            ):  # should never happen because schemnun has been set to N or None if this is the case
                if not warned:
                    sys.stderr.write(
                        "**ERROR**2 in merge_scFv_antibodyVD_numbers found k named %s among %s - possilby these are VHH or VLs in tandem! SKIPPING second chain!"
                        % (k, str(list(ks)))
                    )
                    warned = True
                continue
            if (
                not add_chain_type_to_numbers
            ):  # here we must add anyway as it is a dictionary
                ch = k
                if force_L_instead_of_K and k == "K":
                    ch = "L"
                schemnum = (ch + ":" + schemnum[0], schemnum[1])
            if schemnum in schemnum_to_seqind:
                print(
                    "Error schemnum %s already in schemnum_to_seqind" % (str(schemnum))
                )
            schemnum_to_seqind[schemnum] = j
    flatten = lambda l: [item for sublist in l for item in sublist]
    warnings = flatten([Fv_res[k][3] for k in ks])
    return seqind_to_schemnum, schemnum_to_seqind, seqind_regions, warnings


def aux_get_antibodyVD_numbers(
    sequence,
    res_num_all,
    cdr1_scheme,
    cdr2_scheme,
    cdr3_scheme,
    seqname="",
    start_ind=None,
    end_ind=None,
    auto_detect_chain_type=True,
    print_warns=True,
):
    if not hasattr(res_num_all[0], "__len__"):
        if print_warns:
            print(
                "ERROR in aux_get_antibodyVD_numbers anarci failed %s -returned %s"
                % (seqname, str(res_num_all[0]))
            )
        if auto_detect_chain_type:
            return None, None, None, ["failed"], None
        return None, None, None, ["failed"]
    outa = list(zip(*res_num_all[0]))
    if not hasattr(outa, "__len__") or len(outa) != 2:
        if print_warns:
            sys.stderr.write(
                "ERROR in aux_get_antibodyVD_numbers anarci failed %s -returned %s"
                % (seqname, str(res_num_all[0]))
            )
        if auto_detect_chain_type:
            return None, None, None, ["failed"], None
        return None, None, None, ["failed"]
    res_num, map_seq = outa
    chain_type = res_num_all[1]
    warnings = []
    seqind_to_schemnum = OrderedDict()
    schemnum_to_seqind = OrderedDict()
    seqind_regions = []  # can be "fr" for framework, "CDR1",2,3
    n, jfirst = 0, -1
    for j, a in enumerate(sequence):
        if start_ind is not None and j < start_ind:
            seqind_to_schemnum[
                j
            ] = "N"  # for N-terminus (may be tag or residues not numbered)
            seqind_regions += [""]
            continue
        elif end_ind is not None and j > end_ind:
            seqind_to_schemnum[j] = None
            seqind_regions += [""]
            continue
        if n is None:  # after variable domain
            seqind_to_schemnum[j] = None
            seqind_regions += [""]
            continue
        while map_seq[n] == "-":
            if res_num[n] in schemnum_to_seqind:
                print(
                    "ERRORg in aux_get_antibodyVD_numbers number %s already in schemnum_to_seqind (j=%d and n=%d) %s"
                    % (str(res_num[n]), j, n, seqname)
                )
            schemnum_to_seqind[res_num[n]] = "-"
            n += 1
        if a == map_seq[n]:
            jfirst = j
            seqind_to_schemnum[j] = res_num[n]
            if res_num[n] in schemnum_to_seqind:
                print(
                    "ERROR in aux_get_antibodyVD_numbers number %s already in schemnum_to_seqind (j=%d and n=%d) %s"
                    % (str(res_num[n]), j, n, seqname)
                )
            schemnum_to_seqind[res_num[n]] = j
            if hasattr(
                cdr1_scheme, "keys"
            ):  # supports dictionary or OrderedDict as input type - assume all cdr ranges are like this
                if chain_type == "K" and chain_type not in cdr1_scheme:
                    ch = "L"  # Kappa light chain to Lambda light chain for this purpose
                else:
                    ch = chain_type
                if ch not in cdr1_scheme:
                    raise Exception(
                        "\n chain_type %s not in input cdr1_scheme\n" % (chain_type)
                    )
                cdr1_scheme = cdr1_scheme[ch]
                cdr2_scheme = cdr2_scheme[ch]
                cdr3_scheme = cdr3_scheme[ch]
            if res_num[n][0] in cdr1_scheme:
                seqind_regions += ["CDR1"]
            elif res_num[n][0] in cdr2_scheme:
                seqind_regions += ["CDR2"]
            elif res_num[n][0] in cdr3_scheme:
                seqind_regions += ["CDR3"]
            else:
                seqind_regions += ["fr"]
            n += 1
        elif jfirst == -1:
            seqind_to_schemnum[
                j
            ] = "N"  # for N-terminus (may be tag or residues not numbered)
            seqind_regions += [""]
        else:
            try:
                print(
                    "WARNING in aux_get_antibodyVD_numbers aa %s ind %d not mapped on numbers [prev_aa=%s prevk=%s=%s] %s - POSSIBLE MISALIGNMENTS!!"
                    % (
                        a,
                        j,
                        sequence[j - 1],
                        str(list(schemnum_to_seqind.keys())[-1]),
                        sequence[list(schemnum_to_seqind.values())[-1]],
                        seqname,
                    )
                )
            except Exception:
                print(
                    "WARNING in aux_get_antibodyVD_numbers aa %s ind %d not mapped on numbers [prev_aa=%s prevk=%s= gap?] %s - POSSIBLE MISALIGNMENTS!!"
                    % (
                        a,
                        j,
                        sequence[j - 1],
                        str(list(schemnum_to_seqind.keys())[-1]),
                        seqname,
                    )
                )
                pass
            warnings += [(j, list(schemnum_to_seqind.keys())[-1])]
            seqind_to_schemnum[j] = None
            seqind_regions += [""]
        if len(map_seq) == n:
            n = None  # outside variable domain, assign None
    if jfirst == -1:
        print(
            "ERROR in aux_get_antibodyVD_numbers jfirst==-1 => sequence not mapped!! %s"
            % (seqname)
        )
    # try a fix for some rare cases where seqs begins with SS
    if (
        0 not in seqind_to_schemnum
        and 1 in seqind_to_schemnum
        and sequence[0] == sequence[1]
    ):
        # print ' here0',
        if seqind_to_schemnum[1] == (2, " ") and (1, " ") not in schemnum_to_seqind:
            a = (1, " ")
        elif seqind_to_schemnum[1] == (1, " ") and (2, " ") not in schemnum_to_seqind:
            a = (2, " ")
        else:
            a = None
        if a is not None:
            schemnum_to_seqind[a] = 0
            seqind_to_schemnum[0] = a
            seqind_to_schemnum = OrderedDict(sorted(seqind_to_schemnum.items()))
            schemnum_to_seqind = OrderedDict(sorted(schemnum_to_seqind.items()))
            print(" .. forcing %s with index %d" % (str(a), 0))
    elif (
        1 not in seqind_to_schemnum
        and 0 in seqind_to_schemnum
        and sequence[0] == sequence[1]
    ):
        # print ' here1',sequence[0]==sequence[1],
        if seqind_to_schemnum[0] == (2, " ") and (1, " ") not in schemnum_to_seqind:
            a = (1, " ")
        elif seqind_to_schemnum[0] == (1, " ") and (2, " ") not in schemnum_to_seqind:
            a = (2, " ")
        else:
            a = None
        if a is not None:
            schemnum_to_seqind[a] = 1
            seqind_to_schemnum[1] = a
            seqind_to_schemnum = OrderedDict(sorted(seqind_to_schemnum.items()))
            schemnum_to_seqind = OrderedDict(sorted(schemnum_to_seqind.items()))
            print(" .. forcing %s with index %d" % (str(a), 1))
    if seqind_regions == []:
        seqind_regions = None  # failure
    if auto_detect_chain_type:
        return (
            seqind_to_schemnum,
            schemnum_to_seqind,
            seqind_regions,
            warnings,
            chain_type,
        )
    return seqind_to_schemnum, schemnum_to_seqind, seqind_regions, warnings


def fix_anarci_misaligned_seq(
    seq,
    consensus_al,
    conservation_index,
    seq_name="",
    cons_index_cutoff=0.9,
    key_indeces=None,
    max_allowed_off=3,
    debug=1,
):
    """
    return re_aligned_seq, success - if success is False the re_aligned_seq will be identical to seq.
    useful for sequences that went in the alignment but whose cysteines are off.
    conservation_index and its cutoff are only used to get key_indeces if these are not given as input
        key_indeces are understud as the indices in the consensus sequence that correspond to positions highly conserved
        where we may expect a near-perfect alignment
    useful inputs (could be refined, these have been obtained from a redundant set of PDB complete cleaned sequences):
    VH_consensus_no_gaps = 'QVQLVESSGGGLVQPGGSLRLSCAASGYFTFSSSTLSGYYMHWVRQAPGKGLEWVGYISPSAGNGGSTYYADSVKGRFTISRDNSKNTAYLQMNSLRSEDTAVYYCARDGGYYGSDGGVAYYAADFFGEYYYYYYFDYWGQGTLVTVSS'
    VH_conservation_index=[0.648, 0.87, 0.829, 0.946, 0.556, 0.724, 0.859, 0.256, 0.916, 0.525, 0.61, 0.708, 0.715, 0.602, 0.872, 0.762, 0.393, 0.763, 0.666, 0.539, 0.673, 0.743, 1.015, 0.471, 0.681, 0.844, 0.875, 0.257, 0.5, 0.485, 0.657, 0.425, 0.302, 0.241, 0.256, 0.257, 0.255, 0.231, 0.595, 0.25, 0.592, 0.36, 1.002, 0.678, 0.832, 0.936, 0.546, 0.929, 0.815, 0.696, 0.626, 0.796, 0.932, 0.847, 0.534, 0.532, 0.128, 0.868, 0.272, 0.337, 0.142, 0.243, 0.257, 0.247, 0.304, 0.522, 0.19, 0.59, 0.322, 0.904, 0.505, 0.467, 0.517, 0.547, 0.715, 0.666, 0.806, 0.555, 0.777, 0.713, 0.617, 0.522, 0.925, 0.473, 0.617, 0.527, 0.613, 0.664, 0.452, 0.723, 0.745, 0.635, 0.642, 0.518, 0.709, 0.808, 0.508, 0.404, 0.758, 0.987, 0.786, 0.913, 0.602, 0.961, 0.822, 1.015, 0.73, 0.626, 0.161, 0.142, 0.139, 0.158, 0.125, 0.134, 0.166, 0.204, 0.212, 0.235, 0.247, 0.251, 0.257, 0.257, 0.258, 0.258, 0.257, 0.257, 0.252, 0.244, 0.231, 0.213, 0.189, 0.164, 0.157, 0.201, 0.215, 0.47, 0.69, 0.537, 0.981, 0.922, 0.817, 0.924, 0.904, 0.443, 0.863, 0.916, 0.945, 0.892, 0.795]
    VL_consensus_no_gaps = 'DIVLTQSPSSLSASPGERVTISCRASSSQSISSSNNGKNYLAWYQQKPGQAPKLLIYDKSGAAADGASNRASGVPDRFSGSGSGSSTDFTLTISSLQAEDFATYYCQQYDSSLSVSAAAAAAAAAAAAAGGGESSPYTFGGGTKLEIKR'
    VL_conservation_index=[0.565, 0.599, 0.58, 0.663, 0.902, 0.969, 0.557, 0.683, 0.331, 0.534, 0.624, 0.661, 0.457, 0.678, 0.521, 0.924, 0.519, 0.532, 0.677, 0.708, 0.678, 0.529, 1.016, 0.496, 0.567, 0.727, 0.208, 0.265, 0.54, 0.41, 0.567, 0.232, 0.181, 0.21, 0.239, 0.258, 0.235, 0.231, 0.362, 0.52, 0.543, 0.401, 1.02, 0.85, 0.876, 0.87, 0.736, 0.818, 0.809, 0.475, 0.479, 0.908, 0.625, 0.745, 0.782, 0.906, 0.792, 0.229, 0.264, 0.262, 0.265, 0.266, 0.266, 0.266, 0.264, 0.262, 0.5, 0.643, 0.338, 0.612, 0.331, 0.596, 0.92, 0.755, 0.902, 0.44, 0.989, 0.993, 0.75, 0.914, 0.844, 0.656, 0.807, 0.877, 0.258, 0.258, 0.719, 0.546, 0.651, 0.598, 0.912, 0.675, 0.961, 0.602, 0.379, 0.612, 0.637, 0.434, 0.849, 0.994, 0.424, 0.821, 0.438, 0.98, 0.799, 1.016, 0.6, 0.604, 0.349, 0.266, 0.264, 0.197, 0.249, 0.264, 0.265, 0.265, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.266, 0.265, 0.265, 0.265, 0.262, 0.219, 0.16, 0.53, 0.273, 0.657, 0.99, 0.933, 0.493, 0.937, 0.932, 0.861, 0.702, 0.7, 0.636, 0.696, 0.628]
    VHH_consensus_no_gaps = 'QVQLQESGGGGLVQAGGSLRLSCAASGSRTFSSYFGDTYAMGWFRQAPGKEREFVAAISSSGSSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAAGRGGSGSSGYCGVAAAAIHAAYTSPGEYDYWGQGTQVTVSS'
    VHH_conservation_index=[0.741, 0.917, 0.826, 0.957, 0.627, 0.826, 0.888, 0.26, 0.912, 0.901, 0.887, 0.713, 0.923, 0.913, 0.627, 0.902, 0.811, 0.906, 0.962, 0.87, 0.949, 0.871, 1.013, 0.713, 0.782, 0.874, 0.733, 0.26, 0.32, 0.512, 0.507, 0.474, 0.21, 0.255, 0.259, 0.26, 0.255, 0.252, 0.485, 0.244, 0.736, 0.543, 1.007, 0.643, 0.961, 0.923, 0.803, 0.965, 0.885, 0.863, 0.656, 0.819, 0.933, 0.425, 0.881, 0.649, 0.249, 0.802, 0.316, 0.304, 0.205, 0.257, 0.258, 0.249, 0.493, 0.463, 0.264, 0.706, 0.398, 0.903, 0.63, 0.835, 0.817, 0.858, 0.885, 0.874, 0.963, 0.97, 0.884, 0.9, 0.878, 0.762, 0.952, 0.782, 0.783, 0.834, 0.805, 0.825, 0.644, 0.811, 0.956, 0.853, 0.969, 0.839, 0.699, 0.941, 0.816, 0.84, 0.948, 0.987, 0.933, 0.892, 0.659, 0.969, 0.827, 1.013, 0.637, 0.511, 0.159, 0.132, 0.083, 0.101, 0.089, 0.113, 0.136, 0.177, 0.218, 0.247, 0.258, 0.259, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.259, 0.253, 0.239, 0.208, 0.156, 0.141, 0.15, 0.106, 0.161, 0.432, 0.38, 0.639, 0.887, 0.904, 0.877, 0.912, 0.954, 0.858, 0.951, 0.955, 0.947, 0.898, 0.873]
    """
    if key_indeces is None:
        key_indeces = numpy.where(numpy.array(conservation_index) >= cons_index_cutoff)[
            0
        ]
    seq = numpy.array(list(seq))
    consensus_al = numpy.array(list(consensus_al))
    chunks = misc.contiguous_regions(
        seq != "-"
    )  # chunks of contiguous sequence separated by gap regions, Returns a 2D array where the first column is the start index of the region and the second column is the end index
    if debug:  # useful for printing
        highlight = numpy.array([" " for a in seq])
        highlight[key_indeces] = "|"
        initial_situa = seq == consensus_al
        initial_score = (initial_situa[key_indeces]).sum()
        if debug > 1:
            print("initial situation " + seq_name)
            print(
                ("".join(highlight))
                + "\n"
                + ("".join(seq))
                + "  seq\n"
                + ("".join(consensus_al))
                + "  cons.\nscore=",
                initial_score,
                "/",
                len(key_indeces),
                " identity=",
                initial_situa.sum(),
                "/",
                len(initial_situa),
            )
            print("chunks=", chunks)
    # now move one chunck at a time and see when/if it improves
    best_chunck_offset = {}
    for cj, chunk in enumerate(chunks):
        chunk_key_inds = (
            key_indeces - chunk[0]
        )  # rescale key indices so that they (those >=0 and <len(chunk) ) point to this chunk directly
        if cj > 0:
            # min_off= -1 * min([max_allowed_off, chunk[0]-chunks[cj-1][1]]) # this is a very safe way of assigning boundaries for offset, however it's problematic because sometimes you only have 1 gap but you need to move by 2 as the previous bits are moved by 1 so that 2 gaps actually become available
            min_off = -1 * max_allowed_off
        else:
            min_off = -1 * min([max_allowed_off, chunk[0]])
        if cj < len(chunks) - 1:
            # max_off = min([max_allowed_off, chunks[cj+1][0]-chunk[1] ])# this is a very safe way of assigning boundaries for offset, however it's problematic because sometimes you only have 1 gap but you need to move by 2 as the previous bits are moved by 1 so that 2 gaps actually become available
            max_off = max_allowed_off
        else:
            max_off = min([max_allowed_off, len(seq) - chunk[1]])
        # if debug >1 :
        #    print('\nchunk_key_inds=',chunk_key_inds[ (chunk_key_inds>=0) & (chunk_key_inds<len(situa))],'chunk=',chunk,cj) #,('\n'+(''.join([' ' if j not in chunk_key_inds else '|' for j in range(len(situa))])))+'\n'+(''.join(seq[chunk[0]:chunk[1]]))+'\n'+(''.join(consensus_al[chunk[0]:chunk[1]]))+'\nscore=',score,'/',len(chunk_key_inds),' identity=',situa.sum(),'/',len(situa))
        if debug > 1:
            print("cj=", cj, "chunk=", chunk, "min_off=", min_off, "max_off=", max_off)
        for off in range(min_off, max_off + 1):
            situa1 = (
                seq[chunk[0] : chunk[1]]
                == consensus_al[chunk[0] + off : chunk[1] + off]
            )
            cc = chunk_key_inds - off
            cc = cc[(cc >= 0) & (cc < len(situa1))]
            score1 = situa1[cc].sum()
            if debug > 1:
                print(
                    " off=",
                    off,
                    (
                        "\n "
                        + (
                            "".join(
                                [
                                    " " if j not in cc else "|"
                                    for j in range(len(situa1))
                                ]
                            )
                        )
                    )
                    + "\n "
                    + ("".join(seq[chunk[0] : chunk[1]]))
                    + " seq\n "
                    + ("".join(consensus_al[chunk[0] + off : chunk[1] + off]))
                    + " cons\n score=",
                    score1,
                    "/",
                    len(cc),
                    " identity=",
                    situa1.sum(),
                    "/",
                    len(situa1),
                )
            if len(cc) == 0:
                offset_confidence_score = -1
            else:
                offset_confidence_score = score1 * 1.0 / len(cc)
            if cj not in best_chunck_offset:
                best_chunck_offset[cj] = [
                    (score1, situa1.sum(), off, len(cc), offset_confidence_score)
                ]
            else:
                best_chunck_offset[cj] += [
                    (score1, situa1.sum(), off, len(cc), offset_confidence_score)
                ]
        if cj in best_chunck_offset:  # sort and get best offset per chunk
            best_chunck_offset[cj] = sorted(
                best_chunck_offset[cj],
                key=lambda x: (max([1, (chunk[1] - chunk[0]) / 4]) * x[0] + x[1]),
                reverse=True,
            )  # max(1,len_of_the_chunk/4)* number_of_same_key_ind + overall_identity , this is done so that if overall identity is high but number_of_same_key_ind is 0 the highest identity may still be top ranking
            if debug > 1:
                print(
                    "best_chunck_offset for cj=",
                    cj,
                    "best_chunck_offset=",
                    best_chunck_offset[cj][0][2],
                    (chunks[cj][1] - chunks[cj][0]),
                    "all scores=",
                    best_chunck_offset[cj],
                )

    selected_offsets_per_chunk = {}
    double_check = []
    current_offset = 0
    for j in best_chunck_offset:
        selected_offsets_per_chunk[j] = current_offset  # default
        if len(best_chunck_offset[j]) > 1:
            if (
                best_chunck_offset[j][0][4] > 0
                and best_chunck_offset[j][1][4] < best_chunck_offset[j][0][4] / 2.0
            ):  # look at offset_confidence_score
                selected_offsets_per_chunk[j] = best_chunck_offset[j][0][
                    2
                ]  # select new offset
            elif (
                best_chunck_offset[j][0][2] != current_offset
            ):  # best is not the one we are applying but we don't have much confidence in it
                double_check += [j]
        current_offset = selected_offsets_per_chunk[j]

    for (
        j
    ) in (
        double_check
    ):  # get highest ranking between offset of previous, 0, and offset of next - all equal we favour previous and then 0.
        # we favour next_offset over prev_offset only if they have equal scores and there are fewer gaps separating from next_chunk then from prev_chunk
        n = j - 1
        prev_off = 0
        while n > 0:
            if n not in double_check and n in selected_offsets_per_chunk:
                prev_off = selected_offsets_per_chunk[n]
                break
            n -= 1
        next_off = 0
        nn = j + 1
        while nn < len(selected_offsets_per_chunk):
            if nn not in double_check and nn in selected_offsets_per_chunk:
                next_off = selected_offsets_per_chunk[nn]
                break
            nn += 1
        scores = {}
        for off in misc.uniq([prev_off, 0, next_off]):
            scores[off] = [tup for tup in best_chunck_offset[j] if tup[2] == off]
            if scores[off] == []:
                del scores[off]  # this offset was not an option for this chunk
            elif len(scores[off]) == 1:
                scores[off] = scores[off][0]
            else:
                sys.stderr.write(
                    "**ERROR** in fix_misaligned_seq unusual scores[off] with multiple values for same offset! [seq_name='%s']\n"
                    % (str(seq_name))
                )
                sys.stderr.flush()
                return seq, False
        sorted_scores_values = sorted(
            scores.values(), key=lambda x: x[:2], reverse=True
        )
        if prev_off in scores:
            if sorted_scores_values[0][:2] > scores[prev_off][:2]:
                selected_offsets_per_chunk[j] = sorted_scores_values[0][
                    2
                ]  # save new offset as it is actually better
            elif (
                next_off in scores
                and next_off != prev_off
                and scores[prev_off][:2] == scores[next_off][:2]
                and j > 0
                and j + 1 < len(chunks)
                and chunks[j][0] - chunks[j - 1][1] > chunks[j][1] - chunks[j + 1][0]
            ):
                selected_offsets_per_chunk[
                    j
                ] = next_off  # save the one with least number of gaps in between
            else:
                selected_offsets_per_chunk[j] = prev_off
        elif 0 in scores:
            if sorted_scores_values[0][:2] > scores[0][:2]:
                selected_offsets_per_chunk[j] = sorted_scores_values[0][
                    2
                ]  # save new offset as it is better
            else:
                selected_offsets_per_chunk[j] = 0
        else:
            selected_offsets_per_chunk[j] = sorted_scores_values[0][
                2
            ]  # this should never be the case and it should always have been saved above
    if debug > 1:
        print("selected_offsets_per_chunk=", selected_offsets_per_chunk)
    # now apply offset unless conflicting
    new_seq = numpy.copy(seq)
    for cj, chunk in enumerate(chunks):
        off = selected_offsets_per_chunk[cj]
        new_seq[chunk[0] : chunk[1]] = "-"
        new_seq[chunk[0] + off : chunk[1] + off] = seq[
            chunk[0] : chunk[1]
        ]  # that + or - should remain within length and be >0 should have been enforced earlier

    # now check that we did not mess up
    new_seq_str = "".join(new_seq)
    seq_str = "".join(seq)
    wrong = False
    if len(seq_str) != len(new_seq_str):
        sys.stderr.write(
            "**Error** in fix_misaligned_seq procedure affected seq length (%d is now %d) - new sequence will be dropped and this won't be 'corrected' [seq_name='%s']\n"
            % (len(seq_str), len(new_seq_str), str(seq_name))
        )
        wrong = True
    elif seq_str.replace("-", "") != new_seq_str.replace("-", ""):
        sys.stderr.write(
            "**Error** in fix_misaligned_seq procedure affected sequence amino acids (gapless length %d is now %d, if these numbers are the same the two gapless sequences are still different) - new sequence will be dropped and this won't be 'corrected' [seq_name='%s']\n"
            % (
                len(seq_str.replace("-", "")),
                len(new_seq_str.replace("-", "")),
                str(seq_name),
            )
        )
        wrong = True
        sys.stderr.flush()
    if debug:
        try:
            situa = new_seq == consensus_al
            score = (situa[key_indeces]).sum()
            print("fix_anarci_misaligned_seq final situation: " + seq_name)
            print(
                ("".join(highlight))
                + "\n"
                + ("".join(consensus_al))
                + " Cons\n"
                + ("".join(new_seq))
                + " Fixed\n"
                + ("".join(seq))
                + " Old\n"
                + "New fixed score (based on highly conserved '|' residues)=",
                score,
                "/",
                len(key_indeces),
                " identity=",
                situa.sum(),
                "/",
                len(situa),
                " Old: score=",
                initial_score,
                "/",
                len(key_indeces),
                " identity=",
                initial_situa.sum(),
                "/",
                len(initial_situa),
            )
        except Exception:
            pass
    if wrong:
        return seq, False
    return new_seq, True


def warn(self, message, out=sys.stderr):
        """
        print warning message'
        """
        out.write("*WARNING* " + message)
        out.flush()

      
def clean_anarci_alignment(al, cons_cys_HD=[23,106], del_cyst_misalign=False, add_Nterm_missing='q',try_to_realign_misalignedCys=True, add_C_term_missing='SS',isVHH=False,cons_index_cutoff=0.9 ,consensus_and_consindex_tuple_for_clean=None, warn=True, debug=False, nb_N_gaps=None, check_duplicates=True, verbose=True) :
    '''
    function that checks if an input anarci alignment is consistent and remove sequences with incomplete termini
     or misaligned key Cys residues. (delete those only if del_cyst_misalign=True)
    in AHo numbering at least, VL and VH have both Cys at numbers 23 and 106
     However, for VL one should change the default add_Nterm_missing and add_C_term_missing
    set add_Nterm_missing to None to skip the check of the N terminus
    same for add_C_term_missing
    consensus_seq_for_clean can be a consensus sequence given to clean the alignment,
     if not given it will first use the al.chain_type to use hard-coded one (from AHo scheme)
    '''

    del_k_list = []
    if len(al)==0 :
        sys.stdout.write("EMPTY alignment in clean_anarci_alignment() nothing to clean\n")
        return al
    consensus_al=None
    if consensus_and_consindex_tuple_for_clean is not None :
        consensus_seq_for_clean, conservation_index= consensus_and_consindex_tuple_for_clean
        if len(consensus_seq_for_clean)!=len(conservation_index) :
            sys.stderr.write("\n***ERROR*** in clean_anarci_alignment provided consensus_and_consindex_tuple_for_clean but len(consensus)=%d while len(cons_index)=%d\n"%(len(consensus_seq_for_clean),len(conservation_index)))
        if len(consensus_seq_for_clean)==len(al.HD) :
            consensus_al=consensus_seq_for_clean
        else :
            sys.stderr.write("\n***ERROR*** in clean_anarci_alignment provided consensus_and_consindex_tuple_for_clean with consensus of length %d but alignment has legnth of %d. IGNORING consensus_seq_for_clean and trying to go for hardcoded or compute from alignment itself\n\n" % (len(consensus_seq_for_clean),len(al.HD)))      
    elif al.chain_type=='H' :
        if len(VH_consensus_no_gaps)== len(al.HD) : # go for the hard-coded
            if isVHH : # not too different but a bit
                consensus_al=VHH_consensus_no_gaps
                conservation_index=VHH_conservation_index
            else :
                consensus_al=VH_consensus_no_gaps
                conservation_index=VH_conservation_index
    elif al.chain_type=='K' :
        if len(VKappa_consensus_no_gaps)== len(al.HD) : # go for the hard-coded
            consensus_al=VKappa_consensus_no_gaps
            conservation_index=VKappa_conservation_index
    elif al.chain_type=='L' :
        if len(VLambda_consensus_no_gaps)== len(al.HD) : # go for the hard-coded
            consensus_al=VLambda_consensus_no_gaps
            conservation_index=VLambda_conservation_index
    elif try_to_realign_misalignedCys :
        sys.stderr.write("***WARNING*** clean_anarci_alignment() Alignment chain_type %s not recognised setting try_to_realign_misalignedCys=False\n"%(str(al.chain_type)))
        try_to_realign_misalignedCys=False
    if al.chain_type in ['L','K'] :
        if add_C_term_missing is not None and 'S' in add_C_term_missing : 
            sys.stderr.write("***WARNING*** clean_anarci_alignment() Alignment chain_type %s LIKELY NOT COMPATIBLE with input add_C_term_missing=%s\n"%(str(al.chain_type),str(add_C_term_missing)))
        if add_Nterm_missing is not None and 'q' in add_Nterm_missing.lower() : 
            sys.stderr.write("***WARNING*** clean_anarci_alignment() Alignment chain_type %s LIKELY NOT COMPATIBLE with input add_Nterm_missing=%s\n"%(str(al.chain_type),str(add_Nterm_missing)))
    if try_to_realign_misalignedCys and consensus_al is None : # determine consensus of this alignment under scrutiny
        tmp_clea = clean_anarci_alignment(al, cons_cys_HD=cons_cys_HD, add_Nterm_missing=add_Nterm_missing,try_to_realign_misalignedCys=False, add_C_term_missing=add_C_term_missing, warn=False)
        pssm, log2_pssm, conservation_index ,consensus, aa_freq, aa_freq_ignoring_gaps, gap_freqs=tmp_clea.get_pssm(plot=False)
        alph=numpy.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        consensus_al=''.join(alph[pssm[1:,].argmax(axis=0)]) # don't consider gaps so don't use consensus from above
    
    al_clean=al.copy(deep=False)
    cys=numpy.array([ al.HD.index(c) if c in al.HD else al.HD.index(str(c)) for c in cons_cys_HD ])
    wrong_cys=0
    fixed_wrong_cys=0
    missing_N_term=0
    fixed_missing_N_term=0
    missing_C_term=0
    fixed_missing_C_term=0
    duplicates=0 
    key_indeces=None # in this way we calcualte key_indices only once rather than every time
    if key_indeces is None : 
        key_indeces=numpy.where(numpy.array(conservation_index) >= cons_index_cutoff)[0]
        key_indeces=numpy.unique(list(key_indeces)+list(cys)) # Returns the sorted unique elements
    nodupli_seqs = set()
    for k in al :
        al_clean[k]=al[k][:]
        # check duplicates 
        seq = "".join(al_clean[k])
        if check_duplicates:
            if seq not in nodupli_seqs:
                nodupli_seqs.add(seq)
            else:
                del al_clean[k]
                duplicates +=1
                del_k_list.append(k)
                if warn: sys.stderr.write('Skipping %s as duplicate \n'%(seq))
                  

                continue
        # check Cysteines
        if len(cons_cys_HD):
            if any(numpy.array(al[k])[cys]!='C') :
                wrong_cys+=1
                if try_to_realign_misalignedCys :
                    re_aligned_seq , success = fix_anarci_misaligned_seq( al[k], consensus_al, conservation_index ,seq_name=k,key_indeces=key_indeces,cons_index_cutoff=cons_index_cutoff, debug=debug)
                    if success and all(numpy.array(re_aligned_seq)[cys]=='C'):
                        al_clean[k]=list(re_aligned_seq)[:]
                        fixed_wrong_cys+=1
                    else :
                        #print("THIS ONE ABOVE\n\n")
                        if warn : 
                            sys.stderr.write('Attention %s as no cysteines %s at indices %s =IgG numbering %s\n'%(k,str(numpy.array(al[k])[cys]),str(cys),str(cons_cys_HD)))
                            sys.stderr.write('If the position has not been mutated on purpose, consider that the alignment is erroneous which will impact the whole AbNatiV profile.\n')
                        if del_cyst_misalign:
                            del al_clean[k]
                            del_k_list.append(k)
                        continue                    
                else :
                    if warn : 
                        sys.stderr.write('Attention %s as no cysteines %s at indices %s =IgG numbering %s\n'%(k,str(numpy.array(al[k])[cys]),str(cys),str(cons_cys_HD)))
                        sys.stderr.write('If the position has not been mutated on purpose, consider that the alignment is erroneous which will impact the whole AbNatiV profile.\n')
                    if del_cyst_misalign:
                        del al_clean[k]
                        del_k_list.append(k)
                    continue
        # check N terminus
        if add_Nterm_missing is not None :
            n=0 
            while al[k][n]=='-' and n<len(al[k]) : n+=1
            if n>0 : 
                missing_N_term+=1
                if n<=len(add_Nterm_missing) :
                    n1=0 
                    while al[k][n1]=='-' and n1<len(al[k]) : 
                        al_clean[k][n1]=add_Nterm_missing[n1]
                        n1+=1
                    if warn : sys.stderr.write('Added %s at N terminus of %s to replace %d gaps\n'%(add_Nterm_missing[:n1],k,n))
                    fixed_missing_N_term+=1
                else :
                    if warn : sys.stderr.write('Skipping %s as too many (%d) gaps at N-terminus %s\n'%(k,n,str(al[k][:n+2])))
                    del al_clean[k]
                    del_k_list.append(k)
                    continue

        if add_Nterm_missing is None and type(nb_N_gaps)==int:
            n = 0 
            while al[k][n]=='-' and n<len(al[k]) : n+=1
            if n >nb_N_gaps:
                if warn : sys.stderr.write('Skipping %s as too many (%d) gaps at N-terminus %s\n'%(k,n,str(al[k][:n+2])))
                del al_clean[k]
                missing_N_term=+1
                del_k_list.append(k)
                continue
    
            


        # check C terminus
        if add_C_term_missing and al.chain_type in ['H']:
            n=0 
            while al[k][-1-n]=='-' and n<len(al[k]) : n+=1
            if n>0 : 
                missing_C_term+=1
                if n<=len(add_C_term_missing) :
                    n1=0 
                    while al[k][-1-n1]=='-' and n1<len(al[k]) : 
                        al_clean[k][-1-n1]=add_C_term_missing[-1-n1]
                        n1+=1
                    if warn : sys.stderr.write('Added %s at C terminus of %s to replace %d gaps\n'%(add_C_term_missing[-n1:],k,n))
                    fixed_missing_C_term+=1
                else :
                    if warn : sys.stderr.write('Skipping %s as too many (%d) gaps at C-terminus %s\n'%(k,n,str(al[k][-n-2:])))
                    del al_clean[k]
                    del_k_list.append(k)
                    continue
        if add_C_term_missing and al.chain_type in ['L','K']:
            n=0 
            while al[k][-2-n]=='-' and n<len(al[k])-1 : 
                n+=1 #always have a missing C-term residue
            if n>0 : 
                missing_C_term+=1
                if n<=len(add_C_term_missing) :
                    n1=0 
                    while al[k][-2-n1]=='-' and n1<len(al[k])-1: 
                        al_clean[k][-2-n1]=add_C_term_missing[-1-n1]
                        n1+=1
                    if warn : sys.stderr.write('Added %s at C terminus of %s to replace %d gaps\n'%(add_C_term_missing[-n1:],k,n))
                    fixed_missing_C_term+=1
                else :
                    if warn : sys.stderr.write('Skipping %s as too many (%d) gaps at C-terminus %s\n'%(k,n,str(al[k][-n-2:])))
                    del al_clean[k]
                    del_k_list.append(k)
                    continue
    if verbose: sys.stdout.write("clean_anarci_alignment:\n Processed %d sequence, of which %d saved in cleaned (%.1lf%%)\n" % (len(al),len(al_clean),100.*len(al_clean)/len(al) ))
    if wrong_cys >0 and warn :
        sys.stdout.write(" wrong_cys=%d (%.1lf%% of total), fixed %d of them (%.1lf%% of wrong ones)\n" % (wrong_cys,100.*wrong_cys/len(al),fixed_wrong_cys, 100.*fixed_wrong_cys/wrong_cys ))
    if missing_N_term > 0 and warn :
        sys.stdout.write(" missing_N_term=%d (%.1lf%% of total), fixed %d of them (%.1lf%% of wrong ones)\n" % (missing_N_term,100.*missing_N_term/len(al),fixed_missing_N_term, 100.*fixed_missing_N_term/missing_N_term ))
    if add_C_term_missing is not None  and missing_C_term > 0 and warn :
        sys.stdout.write(" missing_C_term=%d (%.1lf%% of total), fixed %d of them (%.1lf%% of wrong ones)\n" % (missing_C_term,100.*missing_C_term/len(al),fixed_missing_C_term, 100.*fixed_missing_C_term/missing_C_term ))
    if duplicates > 0 and warn :
        sys.stdout.write(" duplicates=%d (%.1lf%% of total))\n" % (duplicates,100.*duplicates/len(al)))

    return al_clean, del_k_list





class Anarci_alignment(OrderedDict):
    def __init__(
        self,
        separate_cdr=False,
        scheme="AHo",
        seed=False,
        isVHH=False,
        filename=None,
        minimum_added_res_per_seq=10,
        chain_type=None,
        _dict=None,
        **argd
    ):
        """
        another interesting scheme is 'AHo' or 'chothia' 'IMGT'
        """
        self.HD = []
        self.seq_region = []  # includes CDR info, it's an header line
        self.filename = ""
        self.separate_cdr = separate_cdr
        self.minimum_added_res_per_seq = minimum_added_res_per_seq
        self.mat = None
        self.filename = filename
        self.scheme = scheme
        self.key_column_hd_name = "ID\\" + str(scheme)
        if isVHH :
            if chain_type is None :
                chain_type='H'
            elif chain_type!='H' :
                warn('isVHH=True but chain_type=%s - will set that to H'%(chain_type))
                chain_type='H'
        self.chain_type = chain_type
        self.clusters = None
        self.consensus = None
        self.conservation_index = None
        self.identity_mat = None  # can create by calling calculate_identity_matrix()
        self.index_to_name = {}  # it will return the key (name) from the index
        self.Fv_info = (
            {}
        )  # will contain the Fv_res of each sequence added with add_sequence (and won't be printed or saved so must be used immediately or saved separately) - keys are seq names
        self.seq_length_info = (
            {}
        )  # filled by calling get_seql_length_info() will contain the length of each sequence and its CDRs (and won't be printed or saved so must be used immediately or saved separately) - keys are seq names
        self.constant_region = (
            OrderedDict()
        )  # where the sequence outside the aligned Fv region is stored
        copy = None
        if "copy" in argd:
            copy = argd["copy"]
            del argd["copy"]
        if _dict is not None:
            super(Anarci_alignment, self).__init__(
                _dict, **argd
            )  # this inits the OrderedDict
        else:
            super(Anarci_alignment, self).__init__(**argd)  # this inits the OrderedDict
        if copy is not None:
            self.copy(
                other_alignment=copy, deep=False
            )  # copy content from other class but not entries
        if seed :
            AHo_seed_seq = "QVQLVQSSGAEVKKPGSSVKVSCKASGRGTFGTSSSSSYIITWVRQAPGQGLEWLGGIIPISSSFGSPNYAQNFQGRVTITADESTTTAYMDLSSLRPDDTAVYYCAKDQGDYDISTWAGSGSSDRRGLEAAPFYFDSWGQGTLVTVSS"  # useful to start an Anarci_alignment in AHo numbering scheme to get all AHo numbers and then potentially discard unusual sequences by setting dont_change_header to True in add_sequence
            self.add_sequence(AHo_seed_seq, name="Seed",print_warnings=False)
            self.chain_type = chain_type # reset chain type
            del self["Seed"]
            if self.chain_type =='H' :
                if isVHH :
                    self.consensus= VHH_consensus_no_gaps
                    self.conservation_index=VHH_conservation_index
                else :
                    self.consensus= VH_consensus_no_gaps
                    self.conservation_index=VH_conservation_index
            elif self.chain_type =='K' :
                self.consensus= VKappa_consensus_no_gaps
                self.conservation_index=VKappa_conservation_index
            elif self.chain_type =='L' :
                self.consensus= VLambda_consensus_no_gaps
                self.conservation_index=VLambda_conservation_index 
    
    def shape(self):
        return (len(self), len(self.HD))
    
    def warn(self, message, out=sys.stderr):
        """
        print warning message'
        """
        out.write("*WARNING* " + message)
        out.flush()

    def err(self, message, out=sys.stderr, stop=False):
        """
        print error message'
        """
        if stop:
            raise Exception("*ERROR* " + message + "\n")
        out.write("*ERROR* " + message + "\n")
        out.flush()

    def __repr__(self):
        s = "<Anarci_alignment class with %d entries from file %s>" % (
            len(self),
            str(self.filename),
        )
        if len(self) > 0:
            s += " Lenght of aligned sequences is %d" % (len(list(self.values())[0]))
        return s

    def __str__(self):
        return "<Anarci_alignment class with %d entries from file %s>" % (
            len(self),
            str(self.filename),
        )

    def copy(self, other_alignment=None, deep=False):
        """
        if other_alignment is not None it copies that class into this class, otherwise it returns a copy of this class
        if deep is not True it does not copy the items
        if you use the other_alignment option with deep=True it will copy the new class into this one without earinsing this first
        """
        if other_alignment is not None:
            self.HD = other_alignment.HD[:]
            self.seq_region = other_alignment.seq_region
            self.filename = other_alignment.filename
            self.separate_cdr = other_alignment.separate_cdr
            self.minimum_added_res_per_seq = other_alignment.minimum_added_res_per_seq
            self.scheme = other_alignment.scheme
            if other_alignment.mat is not None:
                self.mat = other_alignment.mat.copy()
            if other_alignment.identity_mat is not None:
                self.identity_mat = other_alignment.identity_mat.copy()
            if other_alignment.index_to_name is not None:
                self.index_to_name = other_alignment.index_to_name.copy()
            if deep:
                for k in other_alignment:
                    if k in self:
                        sys.stderr.write(
                            "**Warn in deep copy - overwriting k=%s\n" % (str(k))
                        )
                    self[k] = other_alignment[k][:]
                    if k in other_alignment.constant_region:
                        self.constant_region[k] = other_alignment.constant_region[k]
                    if k in other_alignment.Fv_info:
                        self.Fv_info[k] = other_alignment.Fv_info[k]
        else:
            other_alignment = Anarci_alignment()
            other_alignment.HD = self.HD[:]
            other_alignment.seq_region = self.seq_region
            other_alignment.filename = self.filename
            other_alignment.separate_cdr = self.separate_cdr
            other_alignment.minimum_added_res_per_seq = self.minimum_added_res_per_seq
            other_alignment.scheme = self.scheme
            if self.mat is not None:
                other_alignment.mat = self.mat.copy()
            if self.identity_mat is not None:
                other_alignment.identity_mat = self.identity_mat.copy()
            if self.index_to_name is not None:
                other_alignment.index_to_name = self.index_to_name.copy()
            if deep:
                for k in self:
                    other_alignment[k] = self[k][:]
                    if k in self.constant_region:
                        other_alignment.constant_region[k] = self.constant_region[k]
            return other_alignment

    def sort(self, column_index=None, reverse=False, key=lambda x: x):
        """
        may be use to sort alignemt according to aa in one specific column or by entry names.
         more useful with key=something for example to sort it according to clusters
        Unlike the list sort function this function does not sort
        the dictionary, it returns a sorted copy
        to sort it do data=data.sort()
        column_index=None will sort it by keys
        otherwise it will be sorted by the entry in that column
        """
        if column_index is None or (
            column_index == "keys" and column_index not in self.hd
        ):
            if "copy" in self:
                raise Exception(
                    "Cannot have 'copy' as keys in Data class - protected keyword."
                )
            tmp = Anarci_alignment(
                copy=self,
                _dict=OrderedDict(
                    sorted(list(self.items()), key=lambda t: key(t[0]), reverse=reverse)
                ),
            )
            tmp.index_to_name = {}
            return tmp
        else:
            if type(column_index) is str:
                column_index = self.hd[column_index]
            if "copy" in self:
                raise Exception(
                    "Cannot have 'copy' as keys in Data class - protected keyword."
                )
            tmp = Anarci_alignment(
                copy=self,
                _dict=OrderedDict(
                    sorted(
                        list(self.items()),
                        key=lambda t: key(t[1][column_index]),
                        reverse=reverse,
                    )
                ),
            )
            tmp.index_to_name = {}
            return tmp

    def load(self, alignment_fname, chain_type=None, **kwargs):
        self.filename = alignment_fname
        self.chain_type = chain_type
        if self.chain_type is None:
            if "VH" in self.filename and "VL" not in self.filename:
                self.chain_type = "H"
            elif "VL" in self.filename and "VH" not in self.filename:
                self.chain_type = "L"
            elif (
                "light" in self.filename.lower()
                and "heavy" not in self.filename.lower()
            ):
                self.chain_type = "L"
            elif (
                "heavy" in self.filename.lower()
                and "light" not in self.filename.lower()
            ):
                self.chain_type = "H"
            elif (
                "hseq" in self.filename.lower() and "lseq" not in self.filename.lower()
            ):
                self.chain_type = "H"
            elif (
                "lseq" in self.filename.lower() and "hseq" not in self.filename.lower()
            ):
                self.chain_type = "L"
        _, alid, ext = get_file_path_and_extension(alignment_fname)
        if ext in [".csv", ".txt", ".tsv"]:
            if "csv" in ext:
                delimiter = ","
            else:
                delimiter = "\t"
            with open(alignment_fname) as fil:
                for j, line in enumerate(fil):
                    if len(line) < 2 or line[0] == "#":
                        continue
                    splits = line[:-1].split(delimiter)  # ends with '\n'
                    if j == 0:
                        if "\\" in splits[0]:
                            id_info, scheme = splits[0].split("\\")
                            if len(self) == 0:
                                self.scheme = scheme
                            elif scheme != self.scheme:
                                sys.stderr.write(
                                    "**WARNING** in Anarci_alignment.load for file %s, scheme in file is %s but current alignment has scheme set to %s and already contains %d sequences!! LOADING ANYWAY and leaving scheme unchanged!\n"
                                    % (self.filename, scheme, self.scheme, len(self))
                                )
                            chtype = None
                            if "ID:" in id_info:
                                chtype = id_info.replace("ID:", "")
                            elif len(id_info.replace(":", "")) == 1:
                                chtype = id_info.replace(":", "")
                            if chtype is not None and chtype != "None":
                                if self.chain_type is None:
                                    self.chain_type = chtype
                                elif self.chain_type != chtype and chtype is not None:
                                    # print("DEB:",chtype,type(chtype))
                                    sys.stderr.write(
                                        "**WARNING** in load of anarci al inferred chain type %s from fname, but found %s in key_column_hd_name, using the latter!\n"
                                        % (self.chain_type, chtype)
                                    )
                                    self.chain_type = chtype
                        self.HD = [
                            a
                            for ih, a in enumerate(splits)
                            if ih > 0 and a[0].isdigit() or "CDR" in a
                        ]
                    else:
                        k = splits[0]
                        if k in self:
                            self.warn("OVERWRITING k %s in load()\n" % (k))
                        self[k] = splits[1:]

            if self.index_to_name == {} or self.index_to_name is None:
                for j, k in enumerate(self):
                    self.index_to_name[j] = k
        else: # assume fasta or clustal
            recs = get_SeqRecords(alignment_fname, **kwargs)
            Len=None
            for r in recs :
                self[r.id] = list(r.seq)
                if Len!=len(self[r.id]) :
                    if Len is None :
                        Len=len(self[r.id])
                    else :
                        self.warn('input file %s format %s but sequences dont seem aligned as lenght changes (now %d previous %d)\n'%(alid,ext,len(self[r.id]),Len))
            #for r in recs:
            #    self.add_sequence(
            #        str(r.seq).replace("-", ""), name=r.id
            #    )  # effectively re-align
        if self.HD is None or self.HD==[] :
            self.HD=list(range(1,Len+1))
        try :
            self.HD_to_seq_region()
        except Exception :
            self.warn("input file %s format %s number of columns %d - as numbering scheme is not known cannot assign Fv_regions\n"%(alid,ext,Len))
        return

    def update_hd(
        self, new_h, index_replaced_by_insertion, new_aa_symbol="-", debug=False
    ):
        self.HD[index_replaced_by_insertion:index_replaced_by_insertion] = [new_h]
        if debug:
            print(
                "    DEBupdate_hd replacing index %d --> %s"
                % (
                    index_replaced_by_insertion,
                    str(
                        self.HD[
                            misc.positive(
                                index_replaced_by_insertion - 1
                            ) : index_replaced_by_insertion
                            + 2
                        ]
                    ),
                )
            )
        for k in self:
            self[k][index_replaced_by_insertion:index_replaced_by_insertion] = [
                new_aa_symbol
            ]  # put gap
        return

    def HD_to_seq_region(self, framework_name="fr"):
        cdr1_scheme, cdr2_scheme, cdr3_scheme = CDR_ranges_from_scheme(self.scheme)
        self.seq_region = []
        for a in self.HD:
            if ":" in a:
                ch, a = a.split(":")
            else:
                ch = self.chain_type
            if ch == "K":
                ch = "L"  # identical numberings for Kappa and Lambda chains
            if ch is None:
                sys.stderr.write(
                    "**WARNING** in HD_to_seq_region unspecified self.chain_type (None) and cannot infer from header! ASSUMING VH (Heavy Chains)!\n"
                )
                ch = "H"
                if self.chain_type is None:
                    self.chain_type = "H"
            if a[-1].isdigit():
                num = int(a)
            elif "CDR" in a:
                self.seq_region += [a]
                continue
            else:
                num = int(a[:-1])
            if num in cdr1_scheme[ch]:
                self.seq_region += ["CDR1"]
            elif num in cdr2_scheme[ch]:
                self.seq_region += ["CDR2"]
            elif num in cdr3_scheme[ch]:
                self.seq_region += ["CDR3"]
            else:
                self.seq_region += [framework_name]
        return

    def add_from_records(
        self,
        seq_records_or_file,
        names=None,
        overwrite=True,
        minimum_added=10,
        skip_if_different_type=False,
        **kwargs
    ):
        if type(seq_records_or_file) is str:
            seq_records_or_file = get_SeqRecords(seq_records_or_file)
        different_type = []
        failed = []
        if names is not None:
            for j, rec in enumerate(seq_records_or_file):
                self.add_sequence(
                    rec,
                    name=names[j],
                    overwrite=overwrite,
                    minimum_added=minimum_added,
                    skip_if_different_type=skip_if_different_type,
                    different_type=different_type,
                    failed=failed,
                    **kwargs
                )
        else:
            for rec in seq_records_or_file:
                self.add_sequence(
                    rec,
                    overwrite=overwrite,
                    minimum_added=minimum_added,
                    skip_if_different_type=skip_if_different_type,
                    different_type=different_type,
                    failed=failed,
                    **kwargs
                )
        if len(failed) > 0:
            sys.stderr.write(
                "Failed adding %d sequences: failed=%s\n" % (len(failed), str(failed))
            )
        if len(different_type) > 0:
            sys.stderr.write(
                "%d sequences raised different_type warning: different_type=%s\n"
                % (len(different_type), str(different_type))
            )
        return

    def get_complete(self, return_as_class=False):
        """
        filter for those sequences that are complete VH or VL
        it returns the keys (unless return_as_class=True where it returns a copy of the alignment with only those keys)
        """
        if return_as_class:
            c = [
                k
                for k in self
                if (
                    self[k][0] not in ["-", ""]
                    or self[k][1] not in ["-", ""]
                    or self[k][2] not in ["-", ""]
                )
                and (
                    self[k][-1] not in ["-", ""]
                    or self[k][-2] not in ["-", ""]
                    or self[k][-3] not in ["-", ""]
                )
            ]
            ot = Anarci_alignment(separate_cdr=self.separate_cdr)
            ot.HD = self.HD[:]
            for k in c:
                ot[k] = self[k][:]
            return ot
        return [
            k
            for k in self
            if (
                self[k][0] not in ["-", ""]
                or self[k][1] not in ["-", ""]
                or self[k][2] not in ["-", ""]
            )
            and (
                self[k][-1] not in ["-", ""]
                or self[k][-2] not in ["-", ""]
                or self[k][-3] not in ["-", ""]
            )
        ]

    def add_sequence(
        self,
        sequence,
        name=None,
        overwrite=True,
        minimum_added=None,
        return_seqind_regions=False,
        return_seqind_to_schemnum=False,
        skip_if_different_type=False,
        allow=set(["H", "K", "L", "A", "B", "G", "D"]),
        different_type=[],
        failed=[],
        dont_change_header=False,
        print_warnings=2,
        try_to_fix_misalignedCys=False,
        debug=False,
        **kwargs
    ):
        """
        adds a new sequence to the alignment
         seqind_regions is a list of the length of the sequence containing either '' (unassigned number) or 'fr'
         (framework) or 'CDR1','CDR2','CDR3'
         try_to_fix_misalignedCys can be used to fix misalignments on the fly, it is often better to just clean the alignment at the end but this is useful if one is adding a new sequence to compare it with an existing alignment - works only for AHo scheme, has no effect on other schemes.
        """
        if minimum_added is None:
            minimum_added = self.minimum_added_res_per_seq
        if "scheme" in kwargs:
            if kwargs["scheme"] != self.scheme:
                if len(self) == 0:
                    self.warn(
                        "in add_sequence given numbering scheme %s different from init %s - setting new scheme to %s for the whole class\n"
                        % (kwargs["scheme"], self.scheme, kwargs["scheme"])
                    )
                    self.scheme = kwargs["scheme"]
                else:
                    self.err(
                        "in add_sequence given numbering scheme %s different from class scheme %s"
                        % (kwargs["scheme"], self.scheme),
                        True,
                    )
            del kwargs["scheme"]

        if isinstance(sequence, SeqRecord):
            if name is None:
                name = sequence.id
            sequence = str(sequence.seq)
        if name is None:
            name = str(len(self))
        if name in self:
            if not overwrite:
                new_name = csv_dict.new_key_for_dict(self, name)
                self.warn(
                    "DUPLICATED key k %s in add_sequence - already present - adding %s instead\n"
                    % (name, new_name)
                )
                name = new_name
            else:
                self.warn("OVERWRITING k %s in add_sequence\n" % (name))
        Fv_res = get_antibodyVD_numbers(
            sequence,
            seqname=name,
            scheme=self.scheme,
            auto_cdr_scheme=True,
            full_return=True,
            allow=allow,
            **kwargs
        )
        if Fv_res is None:
            failed += [name]
            self.warn("failed %s len=%d\n" % (name, len(sequence)))
            return 0
        if len(Fv_res) != 1:
            chain_type = list(Fv_res.keys())  # chain_type ['H','K']
            if self.chain_type is None:
                self.warn(
                    "in add_sequence %s anarci identified more than one variable domains %s - setting up as scFv alignment\n"
                    % (name, str(list(Fv_res.keys())))
                )
        else:
            chain_type = list(Fv_res.keys())[0]
        # name += '_'+str(chain_type)
        if self.chain_type is None:
            self.chain_type = chain_type
        elif chain_type != self.chain_type:
            different_type += [name]
            if print_warnings == 2:
                self.warn(
                    "in anarci add_sequence adding new sequence %s of different type, alignment set to %s new type %s\n"
                    % (name, str(self.chain_type), str(chain_type))
                )
            if skip_if_different_type:
                if print_warnings == 2:
                    sys.stderr.write(
                        "  skipped %s (skip_if_different_type=True)\n" % (name)
                    )
                return
            if set(chain_type) == set(["H", "K"]) and set(self.chain_type) == set(
                ["H", "L"]
            ):  # some single chains are mixed lambda and Kappa - cannot be added in this way
                Fv_res["L"] = Fv_res["K"]
                del Fv_res["K"]
            elif set(chain_type) == set(["H", "L"]) and set(self.chain_type) == set(
                ["H", "K"]
            ):
                Fv_res["K"] = Fv_res["L"]
                del Fv_res["L"]
        (
            seqind_to_schemnum,
            schemnum_to_seqind,
            seqind_regions,
            warnings,
        ) = merge_scFv_antibodyVD_numbers(
            Fv_res, sequence, add_chain_type_to_numbers=True
        )  # WILL DO NOTHING if single sequence
        if warnings != []:
            if warnings[0] == "failed":
                return 0
            if len(warnings) == 2 and [w[1] for w in warnings] == 2 * [(1, " ")]:
                pass
            elif (
                len(warnings) == 1
                and warnings[0][0] == 1
                and sequence[0] == sequence[1]
            ):
                pass
            else:
                failed += [name]
                self.warn("skipping %s\n" % (name))
                return 0
        if self.Fv_info is not None and hasattr(self.Fv_info, "keys"):
            if overwrite or name not in self.Fv_info:
                self.Fv_info[name] = {}
                for k in Fv_res:
                    self.Fv_info[name][k] = Fv_res[k][
                        3:
                    ]  # no point in saving numbers and regions as this has been aligned
                del Fv_res
        return self.add_processed_sequence(
            sequence,
            seqind_to_schemnum,
            seqind_regions,
            name,
            minimum_added=minimum_added,
            return_seqind_regions=return_seqind_regions,
            return_seqind_to_schemnum=return_seqind_to_schemnum,
            dont_change_header=dont_change_header,
            try_to_fix_misalignedCys=try_to_fix_misalignedCys,
            failed=failed,
            debug=debug,
        )

    def add_processed_sequence(
        self,
        sequence,
        seqind_to_schemnum,
        seqind_regions,
        name,
        minimum_added=10,
        return_seqind_regions=False,
        return_seqind_to_schemnum=False,
        dont_change_header=False,
        try_to_fix_misalignedCys=False,
        cons_cys_HD=[23, 106],
        failed=[],
        debug=False,
    ):
        """
        returns the number of amino acid residues actually added to the alignment - if this is smaller than minimum_added nothing is added and the sequence is skipped.
        seqind_regions can be None if self.separate_cdr is False
        try_to_fix_misalignedCys can be used to fix misalignments on the fly, it is often better to just clean the alignment at the end but this is useful if one is adding a new sequence to compare it with an existing alignment - works only for AHo scheme, has no effect on other schemes.
        input can be obtained with the function get_antibodyVD_numbers
        """
        if isinstance(sequence, SeqRecord):
            if name is None:
                name = sequence.id
            sequence = str(sequence.seq)
        # if self.separate_cdr :
        added = 0
        jj = None
        n, j = 0, 0
        self[name] = []
        self.constant_region[name] = ""
        Fv_added = False
        while j < len(sequence):
            aa = sequence[j]
            # print 'deb',aa,j,n,added,jj
            if seqind_to_schemnum[j] in ["N", None, ""]:  # skip as outside Fv
                if (
                    not Fv_added
                    and seqind_to_schemnum[j] != "N"
                    and len(self.constant_region[name]) < j
                    and len(self[name]) > 0
                ):  # add chain break symbol to say that what comes afterwards is after the Fv region
                    self.constant_region[name] += "/"
                    Fv_added = True
                elif (
                    Fv_added
                    and j > 0
                    and seqind_to_schemnum[j - 1] not in ["N", None, ""]
                ):
                    self.constant_region[
                        name
                    ] += "/"  # may be linker within ligth and heavy chain in scFv
                self.constant_region[name] += aa
                j += 1
                continue
            h = "".join(
                map(str, seqind_to_schemnum[j])
            ).strip()  # convert to string in standard format like 110A - if scFv it will be H:110A for example
            if self.separate_cdr and "CDR" in seqind_regions[j]:
                if seqind_regions[j] in self.HD:
                    jj = self.HD.index(seqind_regions[j])
                elif (
                    len(self.HD) <= j
                ):  # probably first sequence or first sequence this long
                    jj = len(self.HD)
                    self.HD += [seqind_regions[j]]
                else:
                    self.err(
                        "in add_sequence for k %s cannot add CDR %s - skipping sequence\n\n"
                        % (name, seqind_regions[j])
                    )
                    del self[name]
                    return
                while n < jj:
                    self[name] += ["-"]  # add gap
                    n += 1
                if jj == n:
                    self[name] += [
                        ""
                    ]  # add empty string to be filled with CDR loop sequence
                else:
                    self.err(
                        "in add_sequence for k %s CDRzone n>jj %d %d h=%s- skipping sequence\n\n"
                        % (name, jj, n, str(h))
                    )
                    del self[name]
                    return
                while (
                    j < len(seqind_regions) and self.HD[jj] == seqind_regions[j]
                ):  # add whole CDR loop at this location as a str
                    self[name][-1] += sequence[j]
                    added += 1
                    j += 1
                # if len(self[name][-1])<=1 and '!' in separate_cdr_gap:self[name][-1]+='!' # so that CDRs can be recognised as longer than 1 without looking at the header
                n += 1
                continue
            if h in self.HD:
                jj = self.HD.index(h)
                while n < jj:
                    self[name] += ["-"]  # add gap
                    n += 1
                if jj == n:
                    self[name] += [aa]
                    added += 1
                    n += 1
                else:
                    self.err(
                        "in add_sequence for k %s n>jj %d %d h=%s- skipping sequence\n\n"
                        % (name, jj, n, str(h))
                    )
                    del self[name]
                    return
            elif (
                len(self.HD) <= n
            ):  # probably first sequence or first sequence this long
                # if h=='121A' :print "Adding 121A here %d n=%d %s\n%s" %(j,n,name,str(self.HD))
                if (
                    dont_change_header
                ):  # discards any sequences that require header dilation; useful if you specify a seed as longest header and do not want dilations such as 44B, 121C etc
                    self.warn(
                        "discarding sequence %s as header needed change! (added %d residues but will remove!)\n"
                        % (name, added)
                    )
                    failed+=[j] # add to failed
                    Tmp_file = open("failed_sequences.fasta", "a")  # append mode
                    Tmp_file.write(
                        "\n>%s\n%s\n" % (name, sequence)
                    )  # print failed/discarded sequence in fasta format
                    Tmp_file.close()
                    if name in self:  # delete entry in case it was added
                        del self[name]
                    return  # return here so that nothing else is done for this sequence
                self.update_hd(h, n, debug=debug)
                self[name][-1] = aa  # the update_hd has added a '-' to name!
                # self.HD+=[h]
                # self[name]+=[aa]
                added += 1
                n += 1
            else:  # i am already at the n index where I should add the new column
                # if h=='121A' :print "Adding 121A here2 %d n=%d %s\n%s" %(j,n,name,str(self.HD))
                if (
                    dont_change_header
                ):  # discards any sequences that require header dilation; useful if you specify a seed as longest header and do not want dilations such as 44B, 121C etc
                    self.warn(
                        "discarding sequence %s as header needed change! (added %d residues but will remove!)\n"
                        % (name, added)
                    )
                    failed+=[j] # add to failed
                    Tmp_file = open("failed_sequences.fasta", "a")  # append mode
                    Tmp_file.write(
                        "\n>%s\n%s\n" % (name, sequence)
                    )  # print failed/discarded sequence in fasta format
                    Tmp_file.close()
                    if name in self:  # delete entry in case it was added
                        del self[name]
                    return  # return here so that nothing else is done for this sequence
                self.update_hd(h, n, debug=debug)
                self[name][-1] = aa  # the update_hd has added a '-' to name!
                added += 1
                n += 1
            j += 1
        if added < minimum_added:
            self.warn(
                "discarding new sequence %s as only added %d residues\n" % (name, added)
            )
            failed+=[j] # add to failed
            del self[name]
            return 0
        while len(self[name]) < len(self.HD):
            self[name] += ["-"]
        self.index_to_name[len(self) - 1] = name
        # try to fix possible misalignments on the fly
        if (
            try_to_fix_misalignedCys
            and self.scheme == "AHo"
            and (
                numpy.array(self[name])[
                    numpy.array(
                        [
                            self.HD.index(c) if c in self.HD else self.HD.index(str(c))
                            for c in cons_cys_HD
                        ]
                    )
                ]
                != "C"
            ).any()
        ):
            if (self.consensus is None and len(self) >= 20) or (
                len(self) >= 20 and len(self.consensus) < len(self.HD)
            ):
                (
                    pssm,
                    log2_pssm,
                    conservation_index,
                    consensus,
                    aa_freq,
                    aa_freq_ignoring_gaps,
                    gap_freqs,
                ) = self.get_pssm(plot=False)
                alph = numpy.array(
                    [
                        "A",
                        "C",
                        "D",
                        "E",
                        "F",
                        "G",
                        "H",
                        "I",
                        "K",
                        "L",
                        "M",
                        "N",
                        "P",
                        "Q",
                        "R",
                        "S",
                        "T",
                        "V",
                        "W",
                        "Y",
                    ]
                )
                self.consensus = "".join(
                    alph[
                        pssm[
                            1:,
                        ].argmax(axis=0)
                    ]
                )  # don't consider gaps
                self.conservation_index = conservation_index
            elif len(self)<20 and self.consensus is None :
                if self.chain_type =='H' :
                    self.consensus= VH_consensus_no_gaps
                    self.conservation_index=VH_conservation_index
                    sys.stderr.write("*Potential Warning* to fix_anarci_misaligned_seq setting consensus to hard-coded VH_consensus_no_gaps and VH_conservation_index which are human, for chain_type H (note VHH may have different consensus!)\n")
                elif self.chain_type =='K' :
                    self.consensus= VKappa_consensus_no_gaps
                    self.conservation_index=VKappa_conservation_index
                    sys.stderr.write("*Potential Warning* to fix_anarci_misaligned_seq setting consensus to hard-coded VKappa_consensus_no_gaps and VKappa_onservation_index which are human for chain_type K (kappa light)\n")
                elif self.chain_type =='L' :
                    self.consensus= VLambda_consensus_no_gaps
                    self.conservation_index=VLambda_conservation_index 
                    sys.stderr.write("*Potential Warning* to fix_anarci_misaligned_seq setting consensus to hard-coded VLambda_consensus_no_gaps and VLambda_onservation_index which are human for chain_type L (lambda light)\n")
            if self.consensus is not None:
                if self.conservation_index is not None and len(
                    self.conservation_index
                ) == len(self.consensus) == len(self.HD):
                    re_aligned_seq, success = fix_anarci_misaligned_seq(
                        self[name],
                        self.consensus,
                        self.conservation_index,
                        debug=debug,
                        seq_name=name,
                    )
                    if success:  # overwrite
                        self[name] = list(re_aligned_seq)
                else:
                    sys.stderr.write(
                        " **Potential Error in anarci_alignment cannot try_to_fix_misalignedCys with type(self.conservation_index)=%s\n"
                        % (str(type(self.conservation_index)))
                    )
                    if self.conservation_index is not None:
                        sys.stderr.write(
                            " len(self.conservation_index),len(self.consensus),len(self.HD)=%s but should be all the same! This may happen if a new sequences has changed the header\n"
                            % (
                                str(
                                    len(self.conservation_index),
                                    len(self.consensus),
                                    len(self.HD),
                                )
                            )
                        )
            else:
                sys.stderr.write(
                    " **Potential Error in anarci_alignment cannot try_to_fix_misalignedCys with %d aligned sequences. To overcome this you may set self.consensus and self.conservation_index to the hardcoded ones mybio.VH_consensus_no_gaps, self.VH_conservation_index and similar ones for VHH and VL.\n"
                    % (len(self))
                )
        if return_seqind_to_schemnum and return_seqind_regions:
            return added, seqind_regions, seqind_to_schemnum
        if return_seqind_to_schemnum:
            return added, seqind_to_schemnum
        if return_seqind_regions:
            return added, seqind_regions
        if len(self.HD) != len(self.seq_region):
            self.HD_to_seq_region()
        return added
    
    def numpymat(self, only_k=None, remove_cdr_if_separate=True):
        """
        saves in self.mat the numpy matrix corresponding to the whole aligment (no HD)
        """
        if only_k is None:
            only_k = self
        self.mat = numpy.array([self[k] for k in only_k])
        if self.separate_cdr and remove_cdr_if_separate:
            remove = sorted(
                [j for j, k in enumerate(self.HD) if "CDR" in k], reverse=True
            )
            self.mat = numpy.array(numpy.delete(self.mat, remove, axis=1))
        return

    def get_fake_consensus(self, only_k=None):
        if self.mat is None:
            self.numpymat(only_k=only_k, remove_cdr_if_separate=True)
        if only_k is None:
            only_k = self
        fake_consensus = ""
        for j in range(self.mat.shape[1]):
            mat_values, mat_counts = numpy.unique(
                self.mat[:, j], return_counts=True
            )  # return_counts only in numpy versions >= 1.9
            most_common_ind = numpy.argmax(mat_counts)
            most_common, numb_occurrences = (
                mat_values[most_common_ind],
                mat_counts[most_common_ind],
            )  # prints the most frequent element
            if numb_occurrences == len(only_k):
                fake_consensus += most_common  # they are all identical
            elif numb_occurrences >= len(only_k) / 2.0:
                fake_consensus += (
                    most_common.lower()
                )  # this is more than 50% conserved, hence lower case
            elif "-" in self.mat[:, j]:
                fake_consensus += (
                    ","  # non-conserved site where at least one sequence has a gap
                )
            else:
                fake_consensus += "*"  # non-conserved site where no sequence has a gap
        return fake_consensus

    def to_csv_data(self):
        db = csv_dict.Data()
        if self.key_column_hd_name == "ID\\" + str(self.scheme):
            self.key_column_hd_name = (
                "ID:" + str(self.chain_type) + "\\" + str(self.scheme)
            )
        db.key_column_hd_name = self.key_column_hd_name
        db.filename = self.filename
        db.hd = csv_dict.list_to_dictionary(self.HD)
        for k in self:
            db[k] = self[k][:]
        return db

    def Print(
        self,
        fname=sys.stdout,
        print_header_definition_files="auto",
        delimiter=None,
        print_seq_region=True,
    ):
        """
        prints alignment, if ext in ['.fasta','.pir','.pap','.aln','.ali','.fa'] will print the sequences in that format
        otherwise it will print a csv or tab separated table with header containign the numbering
        """
        if type(fname) is str:
            _, nam, ext = get_file_path_and_extension(fname)
            if ext in [".fasta", ".pir", ".pap", ".aln", ".ali", ".fa"]:
                _ = self.to_recs(outfile=fname, file_format=ext[1:])
                if print_header_definition_files == "auto" or (
                    type(print_header_definition_files) is bool
                    and print_header_definition_files is True
                ):
                    # Unfortunatley only single characters are supported, so I don't know how to add the numberings
                    # out=open(nam+'_numbers.bin','w')
                    # col='Dim Gray'
                    # out.write('#\n# Header file of %s numbering.\n# Use this header definition file with %s\n#\n' % (self.scheme,fname))
                    # out.write('name: %s\nstyle: %s\n' % (self.scheme,'character'))
                    # for j,k in enumerate(self.HD) :
                    #    out.write('\t%d\t%s\t%s\n' % (j+1,k,col))
                    # out.close()
                    out = open(nam + "_regions.bin", "w")
                    col_fram, col_cdr = "blue", "red"
                    sym_fram, sym_cdr = "circle", "star"
                    out.write(
                        "#\n# Header file of CDRs and framework regions.\n# Use this header definition file with %s\n#\n"
                        % (fname)
                    )
                    out.write(
                        "name: CDR/framework %s\nstyle: %s\n" % (self.scheme, "symbol")
                    )
                    for j, k in enumerate(self.seq_region):
                        if "CDR" in k:
                            out.write("\t%d\t%s\t%s\n" % (j + 1, sym_cdr, col_cdr))
                        if "fr" in k:
                            out.write("\t%d\t%s\t%s\n" % (j + 1, sym_fram, col_fram))
                    out.close()
                    os.remove(nam + "_regions.bin")
                return
            # returned above
            out = open(fname, "w")
            closeit = True
            if delimiter is None:
                if "csv" in ext:
                    delimiter = ","
                else:
                    delimiter = "\t"
        else:
            closeit = False
            out = fname
            if delimiter is None:
                delimiter = "\t"
        # print header :
        if self.key_column_hd_name == "ID\\" + str(self.scheme):
            self.key_column_hd_name = (
                "ID:" + str(self.chain_type) + "\\" + str(self.scheme)
            )
        out.write(
            delimiter.join(list(map(str, [self.key_column_hd_name] + self.HD))) + "\n"
        )
        if print_seq_region:
            out.write(
                delimiter.join(list(map(str, ["#Seq_region:"] + self.seq_region)))
                + "\n"
            )
        for k in self:
            out.write(delimiter.join(list(map(str, [k] + self[k]))) + "\n")
        if closeit:
            out.close()
        return

    def align_cdr(self):
        """
        Very slow and not better than keeping the numbers from anarci...
        only if separate_cdr==True
        aligns CDR individually and replace the existing ones with aligned versions of them (i.e. with gaps '-' and so on so that they are aligned at the best of their possiblity
        """
        if not self.separate_cdr:
            self.warn(
                "align_cdr possible only for alignments made with separate_cdr=True - leaving class unchanged\n"
            )
            return
        for cd in ["CDR1", "CDR2", "CDR3"]:
            if cd not in self.HD:
                self.warn("cdr %s not in self.HD - skipping\n" % (cd))
                continue
            jj = self.HD.index(cd)
            aligned_seqs = tcoffee_alignment(
                [self[k][jj] for k in self], list(self.keys()), quiet=False
            )
            for j, k in enumerate(self):
                if self[k][jj] != str(aligned_seqs[j].seq).replace("-", ""):
                    self.err(
                        " self[k][jj]!=aligned_seqs[j].replace('-','') %s %s"
                        % (self[k][jj], str(aligned_seqs[j].seq).replace("-", ""))
                    )
                self[k][jj] = aligned_seqs[j]
        return

    def to_recs(
        self,
        outfile=None,
        only_k=None,
        gapless=False,
        upper_case=False,
        file_format="fasta",
    ):
        if only_k is None:
            only_k = self
        records = []
        for rid in only_k:
            try:
                if gapless:
                    s = ("".join(self[rid])).replace("-", "")
                else:
                    s = "".join(self[rid])
                if upper_case:
                    s = s.upper()
            except Exception:
                self.err("at k %s\n" % (rid))
                raise
            records += [SeqRecord(id=rid, name=rid, seq=Seq(s), description="")]
        if outfile is not None:
            print_SeqRecords(records, filename=outfile, file_format=file_format)
        return records

    def align_constant_region(
        self, add_to_main_alignment=True, gaps_to_those_with_no_constant=True
    ):
        """
        align the constant region (if present) using tcoffee
        creates self.aligned_constant_regions as a list
        and adds everyting to the C terminus (if add_to_main_alignment) [give warning if something is found at n-terminus]
        """
        cterms, seq_ids = (
            [],
            [],
        )  # should be the constant regions, as these are after the C terminus of the Fv regions
        linkers, link_ids = [], []
        nterms = (
            []
        )  # should be empty as constant regions are after Fv region, but may have expression/purification tags or similar
        for k in self.constant_region:
            # if len(self.constant_region[k])>2 : # at least two residues, considering one is likely /
            links = []
            if "/" in self.constant_region[k]:
                spl = self.constant_region[k].split("/")
                if len(self.chain_type) > 1:  # linkers are allowed only in this case
                    nterm = spl[0]
                    for jl in range(
                        1, len(self.chain_type)
                    ):  # chain type of scFv can be HL or LH I guess
                        links += [spl[jl]]
                    if jl < len(spl) - 1:
                        cterm = spl[jl + 1]
                    else:
                        cterm = ""
                    if jl + 1 < len(spl) - 1:
                        self.warn(
                            "too many separate constant regions found for %s - found %d\n"
                            % (k, len(spl))
                        )
                else:
                    nterm, cterm = spl
            else:
                nterm, cterm = "", self.constant_region[k]
            nterms += [nterm]
            if len(cterm) >= 1:
                cterms += [cterm]
                seq_ids += [k]
            if links != []:
                linkers += [links]
                link_ids += [k]
        if len(seq_ids) == 0 and len(linkers) == 0:
            self.warn(
                "No constant regions found to align among %d entries - nothing done\n"
                % (len(self))
            )
            return
        elif len(seq_ids) > 0 and len(seq_ids) != len(self):
            self.warn(
                "Suitable constant regions to align identified for %d sequences among %d\n"
                % (len(seq_ids), len(self))
            )
        if any([n != "" and n != "-" for n in nterms]):
            self.warn(
                "in align_constant_region found some non-empty sequences at Nterminus of variable regions - INGORING\n"
            )
            self.nterms = nterms
        # align linkers
        if len(link_ids) > 0:
            self.aligned_linkers = []
            for lj, linkf in enumerate(zip(*linkers)):
                if any(numpy.array([len(l) for l in linkf]) > 2):
                    self.aligned_linkers += [
                        tcoffee_alignment(
                            linkf, link_ids, do_analysis=False, quiet=True
                        )
                    ]
                else:
                    self.aligned_linkers += [linkf]
                ins = None
                al_length = len(self.aligned_linkers[-1][0])
                if (
                    add_to_main_alignment
                ):  # add linkers to main alignment - here we need to insert them between chain types..
                    for j, a in enumerate(self.HD[:-1]):
                        c, i = a.split(":")
                        cn, ine = self.HD[j + 1].split(":")
                        if (
                            int(i) > int(ine) and cn != "li"
                        ):  # li will be already a linker
                            ins = (
                                j + 1
                            )  # using HD[ins:ins]=[li...] will insert between j and j+1
                    if ins is None:
                        self.err(
                            "cannot find suitable insertion site for linkers index %d (last chain type %s)"
                            % (lj, c)
                        )
                    self.HD[ins:ins] = ["li" + str(j + 1) for j in range(al_length)]
                    self.seq_region[ins:ins] = ["link" for j in range(al_length)]
                    i = 0
                    for k in self:
                        if i < len(link_ids) and k == link_ids[i]:
                            self[k][ins:ins] = list(self.aligned_linkers[-1][i])
                            i += 1
                        else:
                            self[k][ins:ins] = ["-"] * al_length
                    if i < len(self.aligned_linkers[-1]):
                        self.warn(
                            "NOT all sequences aligned_constant_regions had linker index %d - added %d of %d\n"
                            % (lj, i, len(self.aligned_constant_regions))
                        )

        # align c-terminus
        if len(seq_ids) > 0:
            if any(numpy.array([len(c) for c in cterms]) > 2):
                if len(cterms) == 1:
                    self.aligned_constant_regions = cterms
                elif len(cterms) == 2:
                    al = pairwise_alignment(
                        cterms[0], cterms[1], one_alignment_only=True
                    )
                    self.aligned_constant_regions = [al[0][0], al[0][1]]
                else:
                    print("DEB: tcoffee", seq_ids, [len(c) for c in cterms])
                    self.aligned_constant_regions = tcoffee_alignment(
                        cterms, seq_ids, do_analysis=False, quiet=True
                    )
            else:
                self.aligned_constant_regions = cterms
            if add_to_main_alignment:
                al_length = len(self.aligned_constant_regions[0])
                last = misc.get_numbers_from_string(self.HD[-1], force_float=False)[
                    0
                ]  # int() should have been sufficient
                self.HD += list(map(str, list(range(last + 1, last + al_length + 1))))
                self.seq_region += ["constant"] * al_length
                i = 0
                for k in self:
                    if i < len(seq_ids) and k == seq_ids[i]:
                        self[k] += list(self.aligned_constant_regions[i])
                        i += 1
                    else:
                        self[k] += ["-"] * al_length
                if i < len(self.aligned_constant_regions):
                    self.warn(
                        "NOT all sequences aligned_constant_regions added - added %d of %d\n"
                        % (i, len(self.aligned_constant_regions))
                    )
        return

    def calculate_identity_matrix(
        self,
        sequence_identity_function=get_pairwise_percent_identity,
        exclude_CDRs=False,
        outfile=None,
        plot=False,
        xlabels_rotation="vertical",
        value_labels=True,
        plot_colorbar=None,
        figure_size=(8, 8),
        save=None,
        frame=False,
        **kwargs
    ):
        """
                other sequence_identity_function:
                    get_pairwise_percent_identity
                    get_number_of_mutations  # this will actually fill the matrix with the number of mutations counting also gaps (unless both have gap)!
                    get_pairwise_blosum_identity # basically a work in progress with a rather random normalisation
                this would be a nseq*nseq matrix containing the pairwise identity between any two sequences
                see also mybio.pssm_from_anarci_alignment
        NEED TO add more customise value labels and colorbar
                return identity_mat,index_to_nameH,name_to_indexH
                craates self.identity_mat
        """
        if exclude_CDRs:
            recs = []
            inds = numpy.array(
                [i for i, r in enumerate(self.seq_region) if "CDR" not in r]
            )
            for k in self:
                if len(self.seq_region) != len(self[k]):
                    self.warn(
                        "seq_regions not properly set as len(self.seq_region) != len(self[k]) %d %d for k= %s - will generate ERRORS with exclude_CDRs=True\n"
                        % (len(self.seq_region), len(self[k]), k)
                    )
                recs += [
                    SeqRecord(seq="".join(numpy.array(self[k])[inds]), id=k, name=k)
                ]
        else:
            recs = self.to_recs(outfile=None, only_k=None, upper_case=True)
        (
            identity_mat,
            index_to_nameH,
            name_to_indexH,
        ) = pairwise_identity_mat_from_aln_record(
            recs,
            do_100_minus=False,
            sequence_identity_function=sequence_identity_function,
        )
        self.identity_mat = identity_mat
        self.index_to_name = index_to_nameH
        # print identity tables
        if outfile is not None:
            out = open(outfile, "w")
        l2 = "\t"
        xlabels, ylabels = [], []
        for j, c in enumerate(identity_mat):
            line = "%s\t" % (index_to_nameH[j])
            ylabels += [index_to_nameH[j]]
            for i, v in enumerate(c):
                if j == 0:
                    xlabels += [index_to_nameH[i]]
                    l2 += "%s\t" % (index_to_nameH[i])
                if i > j:
                    line += "\t"
                else:
                    line += "%d\t" % (v)
            line = line[:-1] + "\n"
            if j == 0:
                l2 = l2[:-1] + "\n"
            if outfile is not None:
                out.write(line)
        if outfile is not None:
            out.write(l2)
            out.close()
        ylabels = ylabels[::-1]
        if plot:
            vlabs = None
            if value_labels is True:
                if (
                    numpy.array(identity_mat, dtype="int") == numpy.array(identity_mat)
                ).all():  # if all int
                    vlabs = numpy.array(identity_mat, dtype="int").astype("str")
                else:
                    vlabs = numpy.round(identity_mat, 1).astype(
                        "str"
                    )  # round to 1 decimal
                vlabs[
                    numpy.arange(len(vlabs)), numpy.arange(len(vlabs))
                ] = ""  # remove labels from diagonal
                if plot_colorbar is None:
                    plot_colorbar = False
            if "clusters" in dir(self) and self.clusters is not None:
                hgrid = numpy.cumsum([len(c) for c in self.clusters])
            else:
                hgrid = None
            plotter.plot_matrix(
                identity_mat,
                value_labels=vlabs,
                xlabels=xlabels,
                ylabels=ylabels,
                frame=frame,
                x_minor_tick_every=False,
                hgrid=hgrid,
                vgrid=hgrid,
                xlabels_rotation=xlabels_rotation,
                figure_size=figure_size,
                save=save,
                plot_colorbar=plot_colorbar,
                **kwargs
            )
        return identity_mat, index_to_nameH, name_to_indexH

    def get_pssm(
        self,
        key_seq_name=None,
        exclude_incomplete_termini=False,
        pssm_file_save_k=None,
        plot=False,
        plot_only_top=8,
        **plot_kwargs
    ):
        """
        return pssm, log2_pssm, conservation_index ,consensus, aa_freq, aa_freq_ignoring_gaps, gap_freqs
        # will use mybio.pssm_from_anarci_alignment
        # pssm should be ['-','A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        alph=numpy.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])
        consensus_no_gaps=''.join(alph[pssm[1:,].argmax(axis=0)]) # don't consider gaps
        pssm_file_save_k is to print the two pssm (PWM and PSSM) to file
         filenames will be pssm_file_save_k+'PWM_frequencypssm.txt' and pssm_file_save_k+'PSSM_loglikelihoodpssm.txt'
        """
        # if self.mat  is None : self.numpymat()
        (
            pssm,
            seqind_regions,
            aa_freq,
            aa_freq_ignoring_gaps,
            gap_freqs,
            consensus,
            key_seq,
            al,
        ) = pssm_from_anarci_alignment(
            self,
            sequence=None,
            seq_name=key_seq_name,
            exclude_incomplete_termini=exclude_incomplete_termini,
            upper_case=True,
            return_aa_frequency_by_sequence_index=False,
        )
        conservation_index = conservation_index_variance_based(
            pssm, num_sequences=len(self), gapless=True
        )
        log2_pssm = ppsm_freq_to_loglikelihood(
            pssm, background="auto", value_for_zerofreqs=True
        )
        if pssm_file_save_k is not None :
            other_columns={ 'Fv_region':self.seq_region, self.scheme+' numbering':self.HD}
            if key_seq_name is None :
                key_seq_name='Consensus'
                key_seq=consensus
            else :
                other_columns['Consensus']=consensus
            print_pssm( pssm, 
                        pssm_file_save_k+'PWM_frequencypssm.txt',
                        sequence_numbering=None,
                        sequence_numbering_name=None,
                        key_sequence=key_seq,
                        key_seq_name=key_seq_name,
                        conservation_index=conservation_index,
                        Nseqs=len(self),
                        other_columns=other_columns,
                        round_to=5,
                    )
            print_pssm( log2_pssm, 
                        pssm_file_save_k+'PSSM_loglikelihoodpssm.txt',
                        sequence_numbering=None,
                        sequence_numbering_name=None,
                        key_sequence=key_seq,
                        key_seq_name=key_seq_name,
                        conservation_index=conservation_index,
                        Nseqs=len(self),
                        other_columns=other_columns,
                        round_to=5,
                    )
        if plot:
            try:
                xlabels = self.HD
                _ = plotter.plot_pssm_of_seq(
                    consensus,
                    pssm,
                    plot_only_top=plot_only_top,
                    plot_conservation_index_num_seqs=conservation_index,
                    xlabels=xlabels,
                    xlabels_rotation="vertical",
                    **plot_kwargs
                )
                if "save" in plot_kwargs:
                    save = (
                        plot_kwargs["save"]
                        .replace(".png", "_loglikelihood.png")
                        .replace(".pdf", "_loglikelihood.pdf")
                    )
                    del plot_kwargs["save"]
                else:
                    save = None
                _ = plotter.plot_pssm_of_seq(
                    consensus,
                    log2_pssm,
                    plot_only_top=plot_only_top,
                    plot_conservation_index_num_seqs=conservation_index,
                    center_cmap_on_value=0,
                    save=save,
                    xlabels=xlabels,
                    xlabels_rotation="vertical",
                    **plot_kwargs
                )
            except Exception:
                sys.stderr.write(
                    "\n\nException while plotting pssm of anarci_alignment\n"
                )
                print_exc(file=sys.stderr)
                sys.stderr.write("Cannot plot!\n")
        return (
            pssm,
            log2_pssm,
            conservation_index,
            consensus,
            aa_freq,
            aa_freq_ignoring_gaps,
            gap_freqs,
        )

    def block_pssm_by_cdr_length(
        self,
        seq_key,
        do_cdr1=True,
        do_cdr2=True,
        do_cdr3=True,
        framework_option="common_hits12",
        return_also_loglikelihood=True,
        do_not_process_gaps_of_key_seq=False,
        restrict_cdrlen_to_aligned_positions=False,
        exclude_incomplete_termini=True,
        add_pseudocounts=True,
        plot=True,
        plot_only_top=8,
        save=None,
        **plot_pssm_kwargs
    ):
        """
        framework_option are 'all', 'common_hits12','common_hits' respectively for all sequences, only those that have same lengths in CDR1 and CDR2, only those that have same length in CDR1 and CDR2 and CDR3
        do_not_process_gaps_of_key_seq will return a pssm without columns corresponding to gap positions into input seq_key
        return pssm,loglikelihood_pssm,individual_pssms,individual_loglikelihood_pssms, consensus, key_seq, cdr_boundary_inds, hd_numbers_retained, seq_numbers
        individual_pssms = [pssm_fr1, pssm_cdr1, pssm_fr2, pssm_cdr2, pssm_fr3, pssm_cdr3, pssm_fr4 ]
         consensus is NOT calculated from loglikelihood_pssm and if return_also_loglikelihood is False loglikelihood_pssm is None
        consider also mybio.allowed_residues_from_pssm mybio.ppsm_freq_to_loglikelihood() and see also mybio.pssm_from_anarci_alignment to get the pssm of the whole alignment
        """
        (
            hits1,
            hits2,
            hits3,
            common_hits12,
            common_hits,
            common_hits_keys,
        ) = self.filter_by_key_seq_cdr_length(
            seq_key,
            do_cdr1=do_cdr1,
            do_cdr2=do_cdr2,
            do_cdr3=do_cdr3,
            restrict_cdrlen_to_aligned_positions=restrict_cdrlen_to_aligned_positions,
        )
        all_keys = list(self.keys())
        ki = all_keys.index(seq_key)
        key_seq_al = numpy.array(self[seq_key])
        hd_numbers_retained = numpy.array(self.HD[:])
        seq_numbers = {}
        if do_not_process_gaps_of_key_seq:
            hd_numbers_retained = hd_numbers_retained[numpy.where(key_seq_al != "-")[0]]
        elif exclude_incomplete_termini:
            tmps = "".join(key_seq_al)
            stri = tmps.strip("-")
            fi = tmps.find(stri)
            fe = fi + len(stri)
            hd_numbers_retained = hd_numbers_retained[fi:fe]
        if framework_option == "all":
            frinds = numpy.arange(len(self))
        elif framework_option == "common_hits12":
            frinds = common_hits12
        elif framework_option == "common_hits":
            frinds = common_hits
        elif do_cdr1 and framework_option == "cdr1":
            frinds = hits1
        elif do_cdr2 and framework_option == "cdr2":
            frinds = hits2
        elif do_cdr3 and framework_option == "cdr3":
            frinds = hits3
        else:
            raise Exception(
                "block_pssm_by_cdr_length framework_option %s not recognised\n"
                % (framework_option)
            )
        if ki not in frinds:
            self.error(
                "block_pssm_by_cdr_length() index %d of seq_key %s not selected for framework\n"
                % (ki, seq_key)
            )
            ki = None
        framework_mat = self.mat[frinds]
        print(
            "seq_key = %s\nFramework option %s --> using %d of %d sequences for framework alignment (%.2lg %%)"
            % (
                seq_key,
                framework_option,
                len(frinds),
                len(self),
                100.0 * len(frinds) / len(self),
            )
        )
        seq_numbers["fr"] = len(frinds)
        nt, ct = 0, 0
        if exclude_incomplete_termini:
            nt, ct = 1, 2  # only N-terminus termini for fr1 and c-terminus for fr4
            fr1_mat = framework_mat[:, : self.inds1[0]]
            pssm_fr1, _, _, _, consensus_fr1, key_seq_fr1 = process_aln_records(
                [
                    SeqRecord(id=all_keys[frinds[j]], seq="".join(s))
                    for j, s in enumerate(fr1_mat)
                ],
                key_id=seq_key,
                add_pseudocounts=add_pseudocounts,
                exclude_incomplete_termini=nt,
                do_not_process_gaps_of_key_seq=do_not_process_gaps_of_key_seq,
                print_log=False,
            )
        else:
            (
                pssm_fr1,
                _,
                _,
                _,
                consensus_fr1,
                key_seq_fr1,
            ) = get_pssm_fast = get_pssm_fast(
                framework_mat[:, : self.inds1[0]],
                key_id=ki,
                add_pseudocounts=add_pseudocounts,
                do_not_process_gaps_of_key_seq=do_not_process_gaps_of_key_seq,
                print_log=False,
            )
        # CDR1:
        if hits1 is None:
            hits1 = frinds
        else:
            seq_numbers["CDR1"] = len(hits1)
            print(
                " CDR1 found %d of %d sequences of same length (%.2lg %%)"
                % (len(hits1), len(self), 100.0 * len(hits1) / len(self))
            )
            cdr1_mat = self.mat[hits1][:, self.inds1]
        pssm_cdr1, _, _, _, consensus_cdr1, key_seq_cdr1 = get_pssm_fast(
            cdr1_mat,
            key_id=ki,
            add_pseudocounts=add_pseudocounts,
            do_not_process_gaps_of_key_seq=do_not_process_gaps_of_key_seq,
            print_log=False,
        )
        # framework 2
        fr2_mat = framework_mat[:, self.inds1[-1] + 1 : self.inds2[0]]
        pssm_fr2, _, _, _, consensus_fr2, key_seq_fr2 = get_pssm_fast(
            fr2_mat,
            key_id=ki,
            add_pseudocounts=add_pseudocounts,
            do_not_process_gaps_of_key_seq=do_not_process_gaps_of_key_seq,
            print_log=False,
        )
        # CDR2:
        if hits2 is None:
            hits2 = frinds
        else:
            seq_numbers["CDR2"] = len(hits2)
            print(
                " CDR2 found %d of %d sequences of same length (%.2lg %%)"
                % (len(hits2), len(self), 100.0 * len(hits2) / len(self))
            )
            cdr2_mat = self.mat[hits2][:, self.inds2]
        pssm_cdr2, _, _, _, consensus_cdr2, key_seq_cdr2 = get_pssm_fast(
            cdr2_mat,
            key_id=ki,
            add_pseudocounts=add_pseudocounts,
            do_not_process_gaps_of_key_seq=do_not_process_gaps_of_key_seq,
            print_log=False,
        )
        # framework 3
        fr3_mat = framework_mat[:, self.inds2[-1] + 1 : self.inds3[0]]
        pssm_fr3, _, _, _, consensus_fr3, key_seq_fr3 = get_pssm_fast(
            fr3_mat,
            key_id=ki,
            add_pseudocounts=add_pseudocounts,
            do_not_process_gaps_of_key_seq=do_not_process_gaps_of_key_seq,
            print_log=False,
        )

        # CDR3:
        if hits3 is None:
            hits3 = frinds
        else:
            seq_numbers["CDR3"] = len(hits3)
            print(
                " CDR3 found %d of %d sequences of same length (%.2lg %%)"
                % (len(hits3), len(self), 100.0 * len(hits3) / len(self))
            )
            cdr3_mat = self.mat[hits3][:, self.inds3]
        pssm_cdr3, _, _, _, consensus_cdr3, key_seq_cdr3 = get_pssm_fast(
            cdr3_mat,
            key_id=ki,
            add_pseudocounts=add_pseudocounts,
            do_not_process_gaps_of_key_seq=do_not_process_gaps_of_key_seq,
            print_log=False,
        )

        # framework last bit
        fr4_mat = framework_mat[:, self.inds3[-1] + 1 :]
        if exclude_incomplete_termini:
            pssm_fr4, _, _, _, consensus_fr4, key_seq_fr4 = process_aln_records(
                [
                    SeqRecord(id=all_keys[frinds[j]], seq="".join(s))
                    for j, s in enumerate(fr4_mat)
                ],
                key_id=seq_key,
                add_pseudocounts=add_pseudocounts,
                exclude_incomplete_termini=ct,
                do_not_process_gaps_of_key_seq=do_not_process_gaps_of_key_seq,
                print_log=False,
            )
        else:
            pssm_fr4, _, _, _, consensus_fr4, key_seq_fr4 = get_pssm_fast(
                fr4_mat,
                key_id=ki,
                add_pseudocounts=add_pseudocounts,
                do_not_process_gaps_of_key_seq=do_not_process_gaps_of_key_seq,
                print_log=False,
            )
        # combine
        cdr_boundary_inds = numpy.cumsum(
            [
                0,
                pssm_fr1.shape[1],
                pssm_cdr1.shape[1],
                pssm_fr2.shape[1],
                pssm_cdr2.shape[1],
                pssm_fr3.shape[1],
                pssm_cdr3.shape[1],
                pssm_fr4.shape[1],
            ]
        )
        individual_pssms = [
            pssm_fr1,
            pssm_cdr1,
            pssm_fr2,
            pssm_cdr2,
            pssm_fr3,
            pssm_cdr3,
            pssm_fr4,
        ]
        pssm = numpy.hstack(individual_pssms)
        consensus = (
            consensus_fr1
            + consensus_cdr1
            + consensus_fr2
            + consensus_cdr2
            + consensus_fr3
            + consensus_cdr3
            + consensus_fr4
        )
        key_seq = (
            key_seq_fr1
            + key_seq_cdr1
            + key_seq_fr2
            + key_seq_cdr2
            + key_seq_fr3
            + key_seq_cdr3
            + key_seq_fr4
        )
        # get log-likelihood, get background from pssm of whole alignment
        loglikelihood_pssm = None
        if return_also_loglikelihood:
            (
                overall_alignment_pssm,
                _,
                _,
                _,
                _,
                overall_consensus,
                key_seq_overall,
                _,
            ) = pssm_from_anarci_alignment(
                self,
                sequence=None,
                seq_name=seq_key,
                do_not_process_gaps_of_key_seq=False,
                return_aa_frequency_by_sequence_index=False,
                exclude_incomplete_termini=exclude_incomplete_termini,
            )
            if self.mat.shape[1] != overall_alignment_pssm.shape[1]:
                self.warn(
                    " return_also_loglikelihood self.mat.shape[1]!=overall_alignment_pssm.shape[1] %d and %d\n"
                    % (self.mat.shape[1], overall_alignment_pssm.shape[1])
                )
            ov_framework_mat = overall_alignment_pssm.copy()
            cdr3_background = overall_alignment_pssm[:, self.inds3].mean(axis=1)
            loglike_cdr3 = ppsm_freq_to_loglikelihood(
                pssm_cdr3, background=cdr3_background, value_for_zerofreqs=True
            )
            ov_framework_mat = numpy.delete(ov_framework_mat, self.inds3, axis=1)
            cdr2_background = overall_alignment_pssm[:, self.inds2].mean(axis=1)
            loglike_cdr2 = ppsm_freq_to_loglikelihood(
                pssm_cdr2, background=cdr2_background, value_for_zerofreqs=True
            )
            ov_framework_mat = numpy.delete(ov_framework_mat, self.inds2, axis=1)
            cdr1_background = overall_alignment_pssm[:, self.inds1].mean(axis=1)
            loglike_cdr1 = ppsm_freq_to_loglikelihood(
                pssm_cdr1, background=cdr1_background, value_for_zerofreqs=True
            )
            ov_framework_mat = numpy.delete(ov_framework_mat, self.inds1, axis=1)
            fr_background = ov_framework_mat.mean(axis=1)
            loglike_fr1 = ppsm_freq_to_loglikelihood(
                pssm_fr1, background=fr_background, value_for_zerofreqs=True
            )
            loglike_fr2 = ppsm_freq_to_loglikelihood(
                pssm_fr2, background=fr_background, value_for_zerofreqs=True
            )
            loglike_fr3 = ppsm_freq_to_loglikelihood(
                pssm_fr3, background=fr_background, value_for_zerofreqs=True
            )
            loglike_fr4 = ppsm_freq_to_loglikelihood(
                pssm_fr4, background=fr_background, value_for_zerofreqs=True
            )
            individual_loglikelihood_pssms = [
                loglike_fr1,
                loglike_cdr1,
                loglike_fr2,
                loglike_cdr2,
                loglike_fr3,
                loglike_cdr3,
                loglike_fr4,
            ]
            loglikelihood_pssm = numpy.hstack(individual_loglikelihood_pssms)
            if plot:
                cbar_label = "log-likelihood"  # Residue enrichment (log2)
                center_cmap_on_value = 0
                savef = save
                if type(save) is str:
                    savef = save.replace(".png", "_loglikelihood.png")
                f = plotter.plot_pssm_of_seq(
                    key_seq,
                    loglikelihood_pssm,
                    plot_only_top=plot_only_top,
                    xlabels=hd_numbers_retained,
                    vline=0.5 + cdr_boundary_inds,
                    xlabels_rotation=90,
                    center_cmap_on_value=center_cmap_on_value,
                    cbar_label=cbar_label,
                    save=savef,
                    **plot_pssm_kwargs
                )
        if plot:
            cbar_label = "Obs. Frequency"
            f = plotter.plot_pssm_of_seq(
                key_seq,
                pssm,
                plot_only_top=plot_only_top,
                xlabels=hd_numbers_retained,
                vline=0.5 + cdr_boundary_inds,
                xlabels_rotation=90,
                cbar_label=cbar_label,
                save=save,
                **plot_pssm_kwargs
            )
        return (
            pssm,
            loglikelihood_pssm,
            individual_pssms,
            individual_loglikelihood_pssms,
            consensus,
            key_seq,
            cdr_boundary_inds,
            hd_numbers_retained,
            seq_numbers,
        )

    def filter_by_key_seq_cdr_length(
        self,
        seq_key,
        do_cdr1=True,
        do_cdr2=True,
        do_cdr3=True,
        restrict_cdrlen_to_aligned_positions=False,
    ):
        """
        will run numpymat() and
        return return hits1,hits2,hits3,common_hits,common_hits12,common_hits_keys
        besides common_hits_keys all are index of the sequences in numpy.mat
        it also saves self.mat, self.inds1, self.inds2, self.inds3 the last 3 are the index position of CDR1,2,3 respectively (not saved if the corresponding do_cdr is False)
        so one can get for example self.mat[hits3][:,self.inds3] for the aligned cdr3 sequences of those sequences in the alignment with a CDR3 of the same length.
        see also mybio.pssm_from_anarci_alignment
        """
        if seq_key not in self:
            raise Exception(
                "**ERROR** filter_by_key_seq_cdr_length(): given seq_key %s not a key of alignment\n"
                % (seq_key)
            )
        if self.separate_cdr:
            raise Exception(
                "**ERROR** filter_by_key_seq_cdr_length(): this class has separate_cdr=True, filtering by length not implemented\n"
            )
        all_keys = list(self.keys())
        ki = all_keys.index(seq_key)
        kseq = numpy.array(self[seq_key])
        self.numpymat()
        if not all(self.mat[ki] == kseq):
            self.err(
                " in filter_by_key_seq_cdr_length not all(self.mat[ki]==kseq) for ki=%d and seq_key %s\n"
                % (ki, seq_key)
            )
        hits1, hits2, hits3 = None, None, None
        self.inds1 = numpy.where(numpy.array(self.seq_region) == "CDR1")[0]
        self.inds2 = numpy.where(numpy.array(self.seq_region) == "CDR2")[0]
        self.inds3 = numpy.where(numpy.array(self.seq_region) == "CDR3")[0]
        if do_cdr1:
            cdr1 = kseq[self.inds1]
            cdr1_len = (cdr1 != "-").sum()
            if (
                restrict_cdrlen_to_aligned_positions
            ):  # typically the same for CDR1/2 or for AHo scheme
                cdr1_inds = numpy.array([j for j in self.inds1 if kseq[j] != "-"])
                hits1 = numpy.where(
                    ((self.mat[:, self.inds1] != "-").sum(axis=1) == cdr1_len)
                    & ((self.mat[:, cdr1_inds] != "-").sum(axis=1) == len(cdr1_inds))
                )[0]
            else:
                hits1 = numpy.where(
                    (self.mat[:, self.inds1] != "-").sum(axis=1) == cdr1_len
                )[0]
        if do_cdr2:
            cdr2 = kseq[self.inds2]
            cdr2_len = (cdr2 != "-").sum()
            if (
                restrict_cdrlen_to_aligned_positions
            ):  # typically the same for CDR1/2 or for AHo scheme
                cdr2_inds = numpy.array([j for j in self.inds2 if kseq[j] != "-"])
                hits2 = numpy.where(
                    ((self.mat[:, self.inds2] != "-").sum(axis=1) == cdr2_len)
                    & ((self.mat[:, cdr2_inds] != "-").sum(axis=1) == len(cdr2_inds))
                )[0]
            else:
                hits2 = numpy.where(
                    (self.mat[:, self.inds2] != "-").sum(axis=1) == cdr2_len
                )[0]
        if do_cdr3:
            cdr3 = kseq[self.inds3]
            cdr3_len = (cdr3 != "-").sum()
            if (
                restrict_cdrlen_to_aligned_positions
            ):  # typically the same for CDR1/2 or for AHo scheme
                cdr3_inds = numpy.array([j for j in self.inds3 if kseq[j] != "-"])
                hits3 = numpy.where(
                    ((self.mat[:, self.inds3] != "-").sum(axis=1) == cdr3_len)
                    & ((self.mat[:, cdr3_inds] != "-").sum(axis=1) == len(cdr3_inds))
                )[0]
            else:
                hits3 = numpy.where(
                    (self.mat[:, self.inds3] != "-").sum(axis=1) == cdr3_len
                )[0]
        common_hits = None
        if hits1 is not None:
            common_hits = hits1
        if hits2 is not None:
            if common_hits is None:
                common_hits = hits2
            else:
                common_hits = [j for j in common_hits if j in hits2]
        common_hits12 = common_hits[:]
        if hits3 is not None:
            if common_hits is None:
                common_hits = hits3
            else:
                common_hits = [j for j in common_hits if j in hits3]
        common_hits_keys = numpy.array(all_keys)[common_hits]
        return hits1, hits2, hits3, common_hits12, common_hits, common_hits_keys

    def get_seq_length_info(self):
        """
        fills self.seq_length_inf with:
        H=['seq_length', 'cdr1_length', 'cdr2_length', 'cdr3_length']
        return seq_length_info, seq_lengths, cdr1_lengths, cdr2_lengths, cdr3_lengths
        """
        self.seq_length_info = {}  # reset in case
        self.inds1 = numpy.where(numpy.array(self.seq_region) == "CDR1")[0]
        self.inds2 = numpy.where(numpy.array(self.seq_region) == "CDR2")[0]
        self.inds3 = numpy.where(numpy.array(self.seq_region) == "CDR3")[0]
        self.numpymat()
        seq_lengths = (self.mat != "-").sum(axis=1)
        cdr1_lengths = (self.mat[:, self.inds1] != "-").sum(axis=1)
        cdr2_lengths = (self.mat[:, self.inds2] != "-").sum(axis=1)
        cdr3_lengths = (self.mat[:, self.inds3] != "-").sum(axis=1)
        for j, k in enumerate(self):
            self.seq_length_info[k] = [
                seq_lengths[j],
                cdr1_lengths[j],
                cdr2_lengths[j],
                cdr3_lengths[j],
            ]
        return (
            self.seq_length_info,
            seq_lengths,
            cdr1_lengths,
            cdr2_lengths,
            cdr3_lengths,
        )

    def add(self, other_al, overwrite=False, print_warnings=True,rename_to_add=False,add_to_start_of_name=''):
        # first update the two HD
        n, j = 0, 0
        if self.separate_cdr != other_al.separate_cdr:
            self.err(
                "cannot add another alignment with different separate_cdr option! (self=%s other=%s)\n\n"
                % (str(self.separate_cdr), str(other_al.separate_cdr)),
                stop=True,
            )
        while n < len(self.HD) or j < len(other_al.HD):
            if self.HD[n] != other_al.HD[j]:
                if (
                    self.HD[n][:3] == "112" and other_al.HD[j][:3] == "112"
                ):  # opposite ordering
                    if self.HD[n] > other_al.HD[j]:
                        other_al.update_hd(self.HD[n], j)
                    else:
                        self.update_hd(other_al.HD[j], n)
                else:
                    if self.HD[n] < other_al.HD[j]:
                        other_al.update_hd(self.HD[n], j)
                    else:
                        self.update_hd(other_al.HD[j], n)
            n += 1
            j += 1
        rn=0
        for j,k in enumerate(other_al):
            nk=k
            if add_to_start_of_name!='' :
                nk=add_to_start_of_name+str(k)
            if nk in self:
                if rename_to_add :
                    while str(rn)+nk in self : # will also apply to next ones
                        rn+=1
                    nk=str(rn)+nk
                else :
                    if not overwrite:
                        if print_warnings:
                            self.warn("skipping %s already present\n" % (nk))
                        continue
                    elif print_warnings:
                        self.warn("overwriting %s\n" % (nk))
            self[nk] = other_al[k][:]
        return




def get_SeqRecords(
    fp_or_seq, check_sequences=False, file_format="fasta", return_as_dictionary=False
):
    """
    get the sequence records from a file.
    It returns a list of the SeqRecord of the sequences. records can be accessed with .seq (sequence, i.e. Seq, for actual string use str(records[].seq)),
     .id id in the file, .name (sometimes equal to id, depends on format), .description (the description, when included) and .dbxref which (if present) is a list with the cross-referenced database.
    return_as_dictionary  uses records.to_dict() to convert the record into a dictionary and return it(from Bio import SeqIO)
    For a list of available file_format see http://biopython.org/wiki/SeqIO#File_Formats .aln is generally 'clustal'

    """
    if os.path.isfile(fp_or_seq):
        handle = open(fp_or_seq, "r")
        records = list(SeqIO.parse(handle, file_format))
        handle.close()
        del handle
    else:
        records = list(SeqIO)
    if check_sequences:
        for record in records:
            if record.seq[-1] == "*":
                record.seq = record.seq[:-1]
    if return_as_dictionary:
        return SeqIO.to_dict(records)
    return records


def print_SeqRecords(list_of_SeqRecord, filename, **kwargs):
    """
    just calls PrintSequenceRecords(list_of_SeqRecord, filename,**kwargs)
    """
    PrintSequenceRecords(list_of_SeqRecord, filename, **kwargs)


def PrintSequenceRecords(
    list_of_SeqRecord, filename, file_format="fasta", print_pickle=None
):
    """
    # Print sequence record to filename (don't give extension), give extension and format in file_format.
    # For a list of available file_format see http://biopython.org/wiki/SeqIO#File_Formats
    """
    if (
        file_format.lower() == "pir"
    ):  # now it is not supported in writing (it is in reading)
        SequenceRecordsToPir(filename, list_of_SeqRecord)
        return
    if file_format not in filename:
        filename += "." + file_format
    output_handle = open(filename, "w")
    fasta_out = FastaIO.FastaWriter(output_handle, wrap=None)
    fasta_out.write_file(list_of_SeqRecord)
    output_handle.close()
    if print_pickle is not None:
        if print_pickle == True:
            fname = filename + ".pkl"
        else:
            fname = print_pickle
        out = open(fname, "wb")
        pickle.dump(list_of_SeqRecord, out)
        out.close()


def SequenceRecordsToPir(alignment_filename, records, check_length=True):
    """
    # auxiliary function for PrintSequenceRecords(), it prints sequences in the pir format
    """
    al = open(alignment_filename, "w")
    for rec in records:
        id_line = "\n>P1;" + str(rec.id) + "\n"
        if (
            rec.description == "<unknown description>"
            or rec.description == ""
            or rec.description is None
        ):
            rec.description = (
                "sequence:%s:    : :+%-4d: :undefined:undefined:-1.00:-1.00"
                % (str(rec.id), len(str(rec.seq).replace("-", "").replace("/", "")))
            )
        elif check_length:
            spl = rec.description.split(":")
            declared_len = int(spl[4])
            actual_len = len(str(rec.seq).replace("-", "").replace("/", ""))
            if declared_len != actual_len:
                spl[4] = "%+-5d" % (actual_len)
                rec.description = ":".join(spl)  # save updated length
        desc_line = rec.description + "\n"
        seq_line = misc.insert_newlines(str(rec.seq), every=75) + "*\n"
        al.write(id_line + desc_line + seq_line)
    al.close()
    return


def print_Alignment_pap(
    seqs_aligned, outfilename, seqs_ids=None, nchar_id=10, nchar_seq=65
):
    """
    can be used to print aligned sequences in pap format (NB seqs must be already aligned and hence all of same length)
    """
    if type(seqs_aligned) is dict or isinstance(seqs_aligned, OrderedDict):
        if seqs_ids is None:
            seqs_aligned, seqs_ids = list(seqs_aligned.values()), list(
                seqs_aligned.keys()
            )
        else:
            seqs_aligned = list(seqs_aligned.values())
    if seqs_ids is None:
        if isinstance(seqs_aligned[0], SeqRecord):
            seqs_ids = [s.id for s in seqs_aligned]
        else:
            seqs_ids = list(map(str, list(range(1, len(seqs_aligned) + 1))))
    id_lines = []
    seq_bits = {}
    matching_aa = None
    for j, sname in enumerate(seqs_ids):
        id_lines += [("%-" + str(nchar_id) + "s ") % (sname[:nchar_id])]
        if type(seqs_aligned[j]) is not str:
            seq = str(seqs_aligned[j].seq)
        else:
            seq = seqs_aligned[j]
        seq_bits[j] = misc.split_every(seq, nchar_seq)
        if matching_aa is None:
            matching_aa = list(seq)
        else:
            matching_aa = [a if a == seq[j] else " " for j, a in enumerate(matching_aa)]
    out = open(outfilename, "w")
    nbits = len(
        seq_bits[j]
    )  # sequences were already aligned, so nbits is the same for all

    matching_aa = ["*" if a != " " else " " for a in matching_aa]  # replace with star
    matching_aa = misc.split_every("".join(matching_aa), nchar_seq)  # convert to string
    for i in range(nbits):
        out.write(
            (" " * nchar_id)
            + "".join(
                [
                    " %9d" % (i)
                    for i in range(
                        i * nchar_seq, i * nchar_seq + len(seqs_aligned[0][i]), 10
                    )
                ]
            )
            + "\n"
        )
        for j in range(len(id_lines)):
            out.write(id_lines[j] + seq_bits[j][i] + "\n")
        out.write(
            (("%-" + str(nchar_id) + "s ") % ("_consensus"[:nchar_id]))
            + matching_aa[i]
            + "\n\n"
        )
    out.close()
    return


def aa_frequency_by_sequence_index(
    pssm, aligned_key_seq=None, query_key_seq=None, pssm_rows=None, sort=True, return_retained_pssm_inds=False,
):
    """
    if pssm_rows is None it will use default ['-']+sorted(amino_list1), and ditch the gaps '-' in case the size is only 20 aa
    indices are positions (starting from 0) in the alignment unless aligned_key_seq is given in which case they are
    indices along the gapless aligned_key_seq
    aligned_key_seq must be alinged with gaps to the same length as pssm
    query_key_seq can be given in case not all of it aligned to generate a pssm (i.e. local alignment)
       in which case the returned aa_frequency_by_index will be a list of the same length as query_key_seq with None
       in those positions where no pssm info was available.
    return aa_frequency_by_index   a list of OreredDicts where index in the list are index in the alignment/key_seq
     keys of individual amino acids are residue type (as in  pssm_rows) while values are frequencies from the pssm
     if sort it sorts these OrderedDict from most abundant to less abundant residue (or gap)
    if return_retained_pssm_inds (works only wiht both query_key_seq and aligned_key_seq given)
    return aa_frequency_by_index,retained_pssm_inds  where retained_pssm_inds is a list as long as 
     query_key_seq (that is as long as aa_frequency_by_index) containing either None for positions with no PSSM 
     info or j where j is the index of the PSSM that correspond to that position.
     useful when there are other pssm-shaped array to process, such as antibody numbering or similar
    """
    # print("DEB: aligned_key_seq=",aligned_key_seq)
    # print("DEB: query_key_seq  =",query_key_seq,aligned_key_seq==query_key_seq, len(aligned_key_seq),pssm.shape)
    if pssm_rows is None:
        if pssm.shape[0] == 21:
            pssm_rows = ["-"] + sorted(amino_list1)
        elif pssm.shape[0] == 20:
            pssm_rows = sorted(amino_list1)
        else:
            raise Exception(
                "\n**ERROR** in aa_frequency_by_sequence_index() cannot infer pssm_rows based on pssm shape of %s\n"
                % (str(pssm.shape))
            )
    aa_frequency_by_index = []
    retained_pssm_inds=[]
    if query_key_seq is not None and aligned_key_seq is not None:
        warned = 0
        n, j = 0, 0
        for aa in query_key_seq:
            while n < len(aligned_key_seq) and aligned_key_seq[n] == "-":
                n += 1
                j += 1  # skip these pssm positions
            if n < len(aligned_key_seq) and aa == aligned_key_seq[n]:
                n += 1
                aa_frequency_by_index += [OrderedDict()]
                retained_pssm_inds += [j]
                for i, a in enumerate(pssm_rows):
                    aa_frequency_by_index[-1][a] = pssm[i, j]
                j += 1
            else:  # should never be here but if aligned_key_seq is trimmed version of query_key_seq it may end up here
                aa_frequency_by_index += [None]
                retained_pssm_inds += [None]
                if warned < 5:
                    if n < len(aligned_key_seq):
                        al_aa = aligned_key_seq[n]
                        sys.stderr.write(" *Warn* in aa_frequency_by_sequence_index found a position (j=%d n=%d) where query_key_seq=%s != aligned_key_seq=%s and the latter is not gap. May lead to downstream errors\n"
                                % (j, n, aa, al_aa))
                        warned += 1
                        if warned == 5:
                            sys.stderr.write("  suppressing further warnings\n")
                    else:
                        al_aa = "none"
                        sys.stderr.write(" *Possible Warning* in aa_frequency_by_sequence_index found a position (j=%d n=%d) where query_key_seq=%s != aligned_key_seq=%s and the latter is not gap. This is likely because only part of the sequence was aligned (e.g. only Fv domain in a longer antibody sequence)\n"
                                % (j, n, aa, al_aa))
                        warned=5
        if j != pssm.shape[1]:  # we have missed some of the pssm!
            sys.stderr.write(
                "***WARNING*** in aa_frequency_by_sequence_index given query_key_seq, aligned_key_seq, and pssm of respective lengths %d %d %s but assessed only %d pssm positions (expecting %d) for %d entries in aa_frequency_by_index\n"
                % (
                    len(query_key_seq),
                    len(aligned_key_seq),
                    str(pssm.shape),
                    j,
                    pssm.shape[1],
                    len(aa_frequency_by_index),
                )
            )
    else:
        for j in range(pssm.shape[1]):  # along the sequence
            if aligned_key_seq is not None and aligned_key_seq[j] == "-":
                continue  # this index will be skipped in aa_frequency_by_index
            aa_frequency_by_index += [OrderedDict()]
            for i, aa in enumerate(pssm_rows):
                aa_frequency_by_index[-1][aa] = pssm[i, j]
    if sort:
        for j, d in enumerate(aa_frequency_by_index):
            if aa_frequency_by_index[j] is not None:
                aa_frequency_by_index[j] = OrderedDict(
                    sorted(list(d.items()), key=lambda t: t[1], reverse=True)
                )  # sort by values
    sys.stderr.flush()
    if return_retained_pssm_inds :
        return aa_frequency_by_index,retained_pssm_inds
    return aa_frequency_by_index


def pssm_from_anarci_alignment(
    anarci_alignment,
    sequence=None,
    seq_name="WT",
    scheme="AHo",
    return_aa_frequency_by_sequence_index=True,
    upper_case=True,
    do_not_process_gaps_of_key_seq=False,
    exclude_incomplete_termini=False,
    try_to_fix_misalignedCys=False,
    unaligned_part_freq_by_ind_vals={},
    plot=False,
    plot_only_top=8,
    **plot_kwargs
):
    """
    sequence may be None or if given is added to the anarci_alignment and then used as key_seq in pssm
    if sequence is already processed it can be given as a tuple (sequence, seqind_to_schemnum, seqind_regions)
    if exclude_incomplete_termini is False it will be MUCH FASTER
    and will call get_pssm_fast
    else it calls process_aln_records on records extracted from anarci_alignment
    return aa_frequency_by_index, seqind_regions, pssm, aa_freq, aa_freq_ignoring_gaps, gap_freqs, consensus,key_seq, al
    OR
    return pssm, seqind_regions, aa_freq, aa_freq_ignoring_gaps, gap_freqs, consensus,key_seq, anarci_alignment
      (anarci_alignment may have extra sequence if given)
    seqind_regions is only if sequence is given, and it is particulary useful when not the whole sequence can be aligned to VH or VL domains
     it is a list of the same length of the sequence with ('','N') for unassigned ('fr',i) for framework where i is the number in the scheme (in most schemes i is a tuple where second is insertion code)
      and (CDR1/2/3,  i)
    aa_frequency_by_index is only returned if return_aa_frequency_by_sequence_index is True and sequence is given
    pssm has the first row for gaps ('-') and the other for the 20 amino acids in sorted(amino_list1).
    any non standard amino acid or non '-' is completely ingnored
    Can visualise pssm with plot=True or: (returned key_seq will have gaps as aligned)
       plotter.plot_pssm_of_seq(key_seq, pssm, plot_only_top=6)
    can call aa_frequency_by_sequence_index(pssm, key_seq=sequence) just after to get aa_frequency_by_index along gapless sequence
    unaligned_part_freq_by_ind_vals may be an emptly list or dictionary, or may be a list of amino acid types that may become allowed mutations at unaligned sites (if function used for this purpose)
      if dictionary it will be filled with WT residue as key and 0 as value
    do_not_process_gaps_of_key_seq should return gapless pssm in sequence (positions of key_seq that are gaps are not columns of pssm)
    try_to_fix_misalignedCys will have an effect when adding sequence if needed
    """
    seqind_regions = None
    if type(anarci_alignment) is str:
        al = Anarci_alignment(scheme=scheme)
        al.load(anarci_alignment)
    else:
        al = anarci_alignment
    if sequence is not None:  # could be string or processed sequence
        if type(sequence) is list or type(sequence) is tuple and len(sequence) == 3:
            sequence, seqind_to_schemnum, seqind_regions = sequence
            n, seqind_regions, seqind_to_schemnum = al.add_processed_sequence(
                sequence,
                seqind_to_schemnum,
                seqind_regions,
                name=seq_name,
                return_seqind_regions=True,
                return_seqind_to_schemnum=True,
                try_to_fix_misalignedCys=try_to_fix_misalignedCys,
                debug=try_to_fix_misalignedCys,
            )
        else:
            n, seqind_regions, seqind_to_schemnum = al.add_sequence(
                sequence,
                name=seq_name,
                overwrite=True,
                return_seqind_regions=True,
                return_seqind_to_schemnum=True,
                try_to_fix_misalignedCys=try_to_fix_misalignedCys,
                debug=try_to_fix_misalignedCys,
            )
        seqind_regions = list(zip(seqind_regions, list(seqind_to_schemnum.values())))
        print(
            "pssm_from_anarci_alignment added %d of %d residues of input sequence to alignment (%d sequences, ncolumns %d)"
            % (n, len(sequence), len(al), len(al.HD))
        )
    elif seq_name is not None and seq_name in al:
        sequence = "".join(al[seq_name])
    else:
        seq_name = None
        return_aa_frequency_by_sequence_index = False

    if exclude_incomplete_termini:
        recs = al.to_recs(gapless=False, upper_case=upper_case)
        (
            pssm,
            aa_freq,
            aa_freq_ignoring_gaps,
            gap_freqs,
            consensus,
            key_seq,
        ) = process_aln_records(
            recs,
            key_id=seq_name,
            do_not_process_gaps_of_key_seq=do_not_process_gaps_of_key_seq,
            exclude_incomplete_termini=exclude_incomplete_termini,
        )
    else:
        (
            pssm,
            aa_freq,
            aa_freq_ignoring_gaps,
            gap_freqs,
            consensus,
            key_seq,
        ) = get_pssm_fast(
            al,
            key_id=seq_name,
            do_not_process_gaps_of_key_seq=do_not_process_gaps_of_key_seq,
        )
    if plot:
        _ = plotter.plot_pssm_of_seq(
            key_seq,
            pssm,
            plot_only_top=plot_only_top,
            plot_conservation_index_num_seqs=pssm.shape[1],
            **plot_kwargs
        )
    if return_aa_frequency_by_sequence_index:
        aa_frequency_by_index = aa_frequency_by_sequence_index(
            pssm, aligned_key_seq=key_seq, query_key_seq=sequence
        )
        if (
            key_seq is not None
            and seqind_regions is not None
            and len(seqind_regions) != n
        ):
            n = 0
            actual_freq_by_ind = []
            for jj, a in enumerate(seqind_regions):
                if a[0] == "":
                    if hasattr(unaligned_part_freq_by_ind_vals, "copy"):
                        actual_freq_by_ind += [unaligned_part_freq_by_ind_vals.copy()]
                    else:
                        actual_freq_by_ind += [unaligned_part_freq_by_ind_vals]
                    if hasattr(unaligned_part_freq_by_ind_vals, "keys"):
                        actual_freq_by_ind[-1][
                            sequence[jj]
                        ] = 0  # add WT residue in dictioary but with 0 frequency (useful for automated design)
                else:
                    actual_freq_by_ind += [aa_frequency_by_index[n]]
                    n += 1
            aa_frequency_by_index = actual_freq_by_ind
        return (
            aa_frequency_by_index,
            seqind_regions,
            pssm,
            aa_freq,
            aa_freq_ignoring_gaps,
            gap_freqs,
            consensus,
            key_seq,
            al,
        )
    return (
        pssm,
        seqind_regions,
        aa_freq,
        aa_freq_ignoring_gaps,
        gap_freqs,
        consensus,
        key_seq,
        al,
    )



def conservation_index_variance_based(pssm, num_sequences, gapless=True):
    """
    return the conservation index variance based, if gapless the first row of the pssm is ignored.
    Formula 2.2 from:
    J. Pei and N. V. Grishin, AL2CO: calculation of positional conservation in a protein sequence alignment , Bioinformatics,  2001
    if you are interest in the same list you can do
    cons_index[numpy.where(numpy.array(list(key_seq))!='-')]
    If you only have pssm and not num_sequences you can estimate it like
    cons_index= numpy.sqrt( ((pssm[1:].T - pssm[1:].mean(axis=1) )**2).sum(axis=1))  # [1:] removes gap line
    albeit it is not identical in fact if you remove gaps then pssm2.mean(axis=1) does not sum to 1
    """
    if gapless and pssm.shape[0] == 21:
        counts = pssm[1:] * num_sequences
        d = counts.sum()
        freq_aa = (counts.sum(axis=1) / d).reshape(
            pssm.shape[0] - 1, 1
        )  # frequency of each amino acid in the alignment (may vary from protein family to protein family)
        cons_index = numpy.sqrt(numpy.sum((pssm[1:] - freq_aa) ** 2, axis=0))
    else:
        counts = pssm * num_sequences
        freq_aa = (counts.sum(axis=1) / counts.sum()).reshape(
            pssm.shape[0], 1
        )  # frequency of each amino acid in the alignment (may vary from protein family to protein family)
        cons_index = numpy.sqrt(numpy.sum((pssm - freq_aa) ** 2, axis=0))
    return cons_index



def print_pssm(
    pssm,
    out_file,
    sequence_numbering=None,
    amino_list=["-"] + sorted(amino_list1),
    sequence_numbering_name=None,
    key_sequence=None,
    key_seq_name=None,
    conservation_index=None,
    Nseqs=None,
    other_columns={},
    round_to=2,
):
    """
    prints a pssm to file, if key_sequence and key_seq_name are given (key_seq_name not used anymore, commented out)
     it prints an extra column with the input sequence the pssm corresponds to
    other_columns can be a dict with keys column names and values list/arrays of same length as pssm to add other columns to print out
     (e.g. antibody region, antibody numbering etc.)
    """
    closeit = False
    if sequence_numbering is None:
        sequence_numbering = list(range(1, len(pssm.T) + 1))
    if sequence_numbering_name is None:
        sequence_numbering_name = ""
    if Nseqs is not None :
        key_column_hd_name='Nseqs=%d|%s'%(Nseqs,sequence_numbering_name)
    else :
        key_column_hd_name=sequence_numbering_name
        if key_column_hd_name=='' :
            key_column_hd_name='Position'
    
    if type(out_file) is str:
        out_file = open(out_file, "w")
        closeit = True
    if key_sequence is not None and len(key_sequence) != pssm.shape[1]:
        sys.stderr.write(
            "**ERROR** in print_pssm given key_sequence of length %d but pssm has shape %s, not printing key_sequence column\n"
            % (len(key_sequence), str(pssm.shape))
        )
        key_sequence = None
    if conservation_index is not None and len(conservation_index) != pssm.shape[1]:
        sys.stderr.write(
            "**ERROR** in print_pssm given conservation_index of length %d but pssm has shape %s, not printing conservation_index column\n"
            % (len(conservation_index), str(pssm.shape))
        )
        conservation_index = None
        # if key_seq_name is None : key_seq_name='pssm_key_sequence'
        # out_file.write('> %s\n%s\n' % (key_seq_name,key_sequence))
    if (
        pssm.shape[0] == len(amino_list) - 1 and "-" in amino_list
    ):  # remove gap from header if not in pssm
        amino_list.remove("-")
    out_file.write("%s" % (key_column_hd_name))
    if key_sequence is not None:
        if key_seq_name is not None : out_file.write("\t%s"%(key_seq_name))
        else : out_file.write("\tSequence_aa")
    if conservation_index is not None:
        out_file.write("\tConservation_index")
    for k in other_columns :
        out_file.write("\t%s"%(k))
    out_file.write("\t" + "\t".join(amino_list) + "\n")
    for j, aaline in enumerate(pssm.T):
        out_file.write("%s" % (sequence_numbering[j]))
        if key_sequence is not None:
            out_file.write("\t%s" % (key_sequence[j]))
        if conservation_index is not None:
            out_file.write("\t%s" % (conservation_index[j]))
        for k in other_columns :
            out_file.write("\t%s" % (other_columns[k][j]))
        for score in aaline:
            if round_to is not None:
                out_file.write("\t%g" % (round(score, round_to)))
            else:
                out_file.write("\t%g" % (score))
        out_file.write("\n")
    if closeit:
        out_file.close()
    return







def process_aln_records(
    records,
    key_id=None,
    weights=None,
    amino_list=sorted(amino_list1),
    add_pseudocounts=True,
    exclude_incomplete_termini=False,
    do_not_process_gaps_of_key_seq=False,
    print_log=True,
):
    """
    the input may be the SeqRecords as read from a clustal alignment (fasta format for example) as downloaded from Uniprot
    from modeller one can save the alignment in fasta format and than read it using get_SeqRecords() and then pass it to this function
    return pssm, aa_freq, aa_freq_ignoring_gaps, gap_freqs, consensus,key_seq
    CONSIDER also: mybio.ppsm_freq_to_loglikelihood to get the pssm converted to log-likelihood instead of frequencies
       (key_seq is None if key_id is not given)
    pssm has the first row for gaps ('-') and the other for the 20 amino acids in amino_list (default sorted(amino_list1)).
    pssm is actually a matrix of frequencies (possibly adjusted with add_pseudocounts) - often known as PWM
     for Position Weight Matrix
    see ppsm_freq_to_loglikelihood() for conversion
    any non standard amino acid or non '-' is completely ingnored
    exclude_incomplete_termini is useful for sequences with trimmed N or C terminus because of sequencing regions (happens for some antibodies for isntance).
     if exclude_incomplete_termini True missing parts (i.e. gaps) at N or C terminus are not considered ONLY in the calculation of the pssm
      if exclude_incomplete_termini is 1 only the N-terminus is processed to exclude gaps, and if 2 only the C-terminus
    add_pseudocounts can be True, in which case it assumes (just for the pseudocounts) a uniform background frequency
       and it will add to the count matrix 1./len(amino_list). Thus the more sequences in the alignment the less important the pseudocounts become, but at least it's never 0 for log-likelihood calculations.
      Alternatively one can give a list or np.array of the same length as amino_list representing the counts to add to each amino acid type.
      Alternatively one can also give a number to use as pseudocounts, the same for all residues.
      NOTE: gaps line will not get any pseudocounts correction.
    Can visualise pssm with:
    plotter.plot_pssm_of_seq(key_seq, pssm, plot_only_top=6)

    """
    if type(records) is str:
        if ".aln" in records:
            file_format = "clustal"
        else:
            file_format = "fasta"
        records = get_SeqRecords(records, file_format=file_format)
    elif type(records[0]) is str:  # not SeqRecord format but simple string
        records = [SeqRecord(id=str(j), seq=Seq(r)) for j, r in enumerate(records)]
    start, end, key_seq = None, None, None
    if key_id is not None:
        if type(key_id) is str:
            for j, r in enumerate(records):
                if str(r.id) == key_id:
                    if print_log:
                        print(" Reference key %s found at %d" % (key_id, j), end=" ")
                    key_id = j
                    break
        if type(key_id) is not int:  # not found
            sys.stderr.write(
                "\n***WARNING*** requested key_id %s not found in alignment. Processing without!\n"
                % (key_id)
            )
        else:
            key_seq = str(records[key_id].seq).strip(
                "-"
            )  # not interested in the gap at termini of the key seq
            start = str(records[key_id].seq).find(key_seq)
            end = start + len(key_seq)
            if do_not_process_gaps_of_key_seq:
                inds_after_slice = numpy.array(list(key_seq))
                inds_after_slice = numpy.where(inds_after_slice != "-")[0]
                if print_log:
                    print(
                        "Considering %d NON-GAP positions in key seq, out of %d columns"
                        % (len(inds_after_slice), len(key_seq))
                    )
                key_seq = "".join(numpy.array(list(key_seq))[inds_after_slice])
            if print_log:
                print("start=", start, "end=", end)
    # using numpy we do a sort of big matrix
    Nseqs = float(len(records))
    mat, matstripped = None, None
    if weights is None:
        W = numpy.ones((len(records), 1))
    elif type(weights) is bool and weights == True:
        W = numpy.array([r.annotations["weight"] for r in records]).reshape(-1, 1)
    else:
        W = numpy.array(weights).reshape(-1, 1)
    for j, r in enumerate(records):
        if j > 0 and j % 10000 == 0:
            sys.stdout.write(
                "processed %d of %d (%.2lf %%)\n" % (j, Nseqs, 100.0 * j / Nseqs)
            )
            sys.stdout.flush()
        if key_seq is not None and do_not_process_gaps_of_key_seq:
            line = numpy.array(list(r.seq[start:end]))[inds_after_slice]
        else:
            line = numpy.array(list(r.seq[start:end]))
        if mat is None:
            mat = line
        else:
            mat = numpy.vstack((mat, line))
        if exclude_incomplete_termini:
            seq = "".join(line)  # str(r.seq[start:end])
            stri = seq.strip("-")
            i1 = seq.find(stri)
            i2 = i1 + len(stri)
            if type(exclude_incomplete_termini) is int:  # only one of the termini
                if exclude_incomplete_termini == 1:  # only N-terminus
                    i2 = len(seq)
                elif exclude_incomplete_termini == 2:  # only C-terminus
                    i1 = 0
            if matstripped is None:
                matstripped = numpy.array(
                    list(("*" * i1) + seq[i1:i2] + ("*" * (len(seq) - i2)))
                )
            else:
                matstripped = numpy.vstack(
                    (
                        matstripped,
                        numpy.array(
                            list(("*" * i1) + seq[i1:i2] + ("*" * (len(seq) - i2)))
                        ),
                    )
                )

    aa_freq = {}
    aa_freq_ignoring_gaps = {}
    gaps = (W * (mat == "-")).sum(axis=0)  # count of gap at each alignment position
    Nseqs = W.sum()  # if all weights are 1 it's the same as number of sequences.
    nongaps = Nseqs - gaps  # count of non gaps at each alignment position
    nongaps[
        nongaps == 0
    ] = -1  # trick for division, then we replace negatives with zeros
    if exclude_incomplete_termini:
        seqs_to_exclude_per_position = (W * (matstripped == "*")).sum(axis=0)
        Nseqs -= seqs_to_exclude_per_position  # now Nseqs is a numpy array, each position effectively has a different number of sequences

    if hasattr(add_pseudocounts, "__len__"):
        pseudocounts = add_pseudocounts
    elif type(add_pseudocounts) is bool and add_pseudocounts == True:
        pseudocounts = [1.0 / len(amino_list) for x in amino_list]
    elif type(add_pseudocounts) is int or type(add_pseudocounts) is float:
        pseudocounts = [add_pseudocounts for x in amino_list]
    else:
        pseudocounts = [0 for x in amino_list]
    # print 'mat.shape=',mat.shape,'gaps=',gaps
    add_den = sum(pseudocounts)
    for j, aa in enumerate(amino_list):
        aa_freq[aa] = ((((W * (mat == aa)).sum(axis=0)) + pseudocounts[j])) / (
            Nseqs + add_den
        )
        aa_freq_ignoring_gaps[aa] = ((W * (mat == aa)).sum(axis=0)) / (
            nongaps + add_den
        )
        aa_freq_ignoring_gaps[aa][
            aa_freq_ignoring_gaps[aa] < 0
        ] = 0  # these were those positions where nongaps was zero
    # create a PSSM like matrix, the first row is gaps
    if exclude_incomplete_termini:
        pssm = (gaps - seqs_to_exclude_per_position) / (
            Nseqs + add_den
        )  # in this ways gaps at the termini don't count towards increase pssm score.
    else:
        pssm = gaps / (Nseqs + add_den)
    # print('DEB 0mybio pssm.shape:',pssm.shape,gaps.shape,'mat.shape=',mat.shape,len(amino_list),'aa_freq[aa].shape=',aa,aa_freq[aa].shape,'pseudocounts=',pseudocounts,'Nseqs=',Nseqs,((mat==aa).sum(axis=0)).shape,(W*((mat==aa).sum(axis=0))).shape,(W*( ((mat==aa).sum(axis=0))+pseudocounts[j]) ).shape)
    for aa in amino_list:
        pssm = numpy.vstack((pssm, aa_freq[aa]))
    del mat
    # print('DEB mybio pssm.shape:',pssm.shape,pssm.argmax(axis=0).shape,pssm,pssm.sum(axis=0))
    consensus = "".join(numpy.array(["-"] + amino_list)[pssm.argmax(axis=0)])
    return pssm, aa_freq, aa_freq_ignoring_gaps, gaps / Nseqs, consensus, key_seq

def get_pssm_fast(
    records,
    key_id=None,
    weights=None,
    amino_list=sorted(amino_list1),
    add_pseudocounts=True,
    do_not_process_gaps_of_key_seq=False,
    print_log=True,
):
    """
    about 10x faster than process_aln_records but does not support exclude_incomplete_termini
    return pssm, aa_freq, aa_freq_ignoring_gaps, gaps_freq, consensus, key_seq
    gap '-' are added to the beginning of amino_list and will be the first row of the pssm
    other useful functions are
        conservation_index = mybio.conservation_index_variance_based(pssm,num_sequences=len(records),gapless=True)
        log2_pssm = mybio.ppsm_freq_to_loglikelihood(pssm, background='auto', value_for_zerofreqs=True)
    """
    key_ind = None
    key_seq = None
    if isinstance(records, numpy.ndarray):
        mat = records
        if key_id is int:
            key_ind = key_id
        elif key_id is not None:
            sys.stderr.write(
                "**ERROR** in mybio.get_pssm_fast only int key_id (or None) corresponding to index are supported when input is numpy.ndarray - ignoring key_id=%s\n"
                % (key_id)
            )
            key_id = None
    if isinstance(records, Anarci_alignment) or "Anarci_alignment" in str(
        type(records)
    ):
        if records.mat is None:
            records.numpymat()
        mat = records.mat
        if key_id is not None:
            if key_id in records:
                key_ind = list(records.keys()).index(key_id)
            elif key_id is int:
                key_ind = key_id
    else:
        if type(records) is str:
            if ".aln" in records:
                file_format = "clustal"
            else:
                file_format = "fasta"
            records = get_SeqRecords(records, file_format=file_format)
        mat = []
        for j, r in enumerate(records):

            if hasattr(r, "id"):
                mat += [numpy.array(list(r.seq))]
                if key_id is not None and key_id in [r.id, j]:
                    key_ind = j
            else:
                mat += [numpy.array(list(r))]
                if key_id is not None and key_id == j:
                    key_ind = j
    mat = numpy.array(mat)
    if key_id is not None and key_ind is None:  # not found
        sys.stderr.write(
            "\n***WARNING*** requested key_id %s not found in alignment. Processing without!\n"
            % (key_id)
        )

    if key_ind is not None:
        ks = "".join(mat[key_ind])
        key_seq = ks.strip("-")  # not interested in the gap at termini of the key seq
        start = ks.find(key_seq)
        end = start + len(key_seq)
        if do_not_process_gaps_of_key_seq:
            inds_before_slice = numpy.where(mat[key_ind] != "-")[0]
            key_seq = "".join(mat[key_ind][inds_before_slice])
            if print_log:
                print(
                    "Considering %d NON-GAP positions in key seq, out of %d columns"
                    % (len(inds_before_slice), len(key_seq))
                )
        if print_log:
            print("start=", start, "end=", end)

    # using numpy we do a sort of big matrix
    # Nseqs=float(len(records))

    if weights is None:
        W = numpy.ones((len(records), 1))
    elif type(weights) is bool and weights == True:
        W = numpy.array([r.annotations["weight"] for r in records]).reshape(-1, 1)
    else:
        W = numpy.array(weights).reshape(-1, 1)

    # peseudocount don't have gaps at the moment
    if hasattr(add_pseudocounts, "__len__"):
        pseudocounts = add_pseudocounts
    elif type(add_pseudocounts) is bool and add_pseudocounts == True:
        pseudocounts = [1.0 / len(amino_list) for x in amino_list]
    elif type(add_pseudocounts) is int or type(add_pseudocounts) is float:
        pseudocounts = [add_pseudocounts for x in amino_list]
    else:
        pseudocounts = [0 for x in amino_list]
    add_den = sum(pseudocounts)
    Nseqs = W.sum()  # if all weights are 1 it's the same as number of sequences.

    aa_count = {}
    aa_freq = {}
    aa_freq_ignoring_gaps = {}
    if do_not_process_gaps_of_key_seq:
        gaps_count = (W * (mat[:, inds_before_slice] == "-")).sum(axis=0)
    elif key_ind is not None:
        gaps_count = (W * (mat[:, start:end] == "-")).sum(axis=0)
    else:
        gaps_count = (W * (mat == "-")).sum(
            axis=0
        )  # count of gap at each alignment position
    nongaps = Nseqs - gaps_count  # count of non gaps at each alignment position
    nongaps[
        nongaps == 0
    ] = -100  # trick for division, then we replace negatives with zeros
    pssm = gaps_count / (Nseqs + add_den)  # frequency matrix, first row is gaps
    # print('DEB:',pssm.shape,gaps_count.shape,mat.shape)
    # print("DEB: pseudocounts",pseudocounts,numpy.array(pseudocounts).shape, '\nnongaps',nongaps,'add_den',add_den)
    for j, aa in enumerate(amino_list):
        if do_not_process_gaps_of_key_seq:
            aa_count[aa] = (
                (W * (mat[:, inds_before_slice] == aa)).sum(axis=0)
            ) + pseudocounts[j]
        elif key_ind is not None:
            aa_count[aa] = ((W * (mat[:, start:end] == aa)).sum(axis=0)) + pseudocounts[
                j
            ]
        else:
            aa_count[aa] = ((W * (mat == aa)).sum(axis=0)) + pseudocounts[j]
        aa_freq[aa] = aa_count[aa] / (Nseqs + add_den)
        aa_freq_ignoring_gaps[aa] = aa_count[aa] / (nongaps + add_den)
        aa_freq_ignoring_gaps[aa][
            aa_freq_ignoring_gaps[aa] < 0
        ] = 0  # these were those positions where nongaps was zero
        pssm = numpy.vstack((pssm, aa_freq[aa]))
    consensus = "".join(numpy.array(["-"] + amino_list)[pssm.argmax(axis=0)])
    return pssm, aa_freq, aa_freq_ignoring_gaps, gaps_count / Nseqs, consensus, key_seq

def ppsm_freq_to_loglikelihood(pssm, background="auto", value_for_zerofreqs=True):
    """
    return log2likelihood_pssm
    pssm is a matrix of frequencies, also called PWM
    pssm sholud not have a row for gaps and ideally should not contain any 0 (made with pseudocounts)
    background=='auto' gets background frequency as overall frequency in alignment background=pssm.mean(axis=1).
    background=='uniform' assign uniform frequency
    background== np.array of length of pssm (usually 21) for custom background frequencies (must be in same order as pssm)
    note that consensus sequence may differ if calculated from frequencies or log-likelihood
    value_for_zerofreqs=True will set the log-likelihood of items with zero frequency to 
      2 int units below the minimum value from non-zero (e.g. if min o observed is -11.68 -> then non observed get -13)
    """
    if background is None:
        background = "auto"
    if type(background) is str:
        if background == "auto":
            background = pssm.mean(axis=1)  # mean frequency in alignment
        elif background == "uniform":
            background = numpy.ones(len(pssm)) / len(pssm)
        else:
            raise Exception(
                "in ppsm_freq_to_loglikelihood background=%s but should be 'auto','uniform' or array\n"
                % (str(background))
            )
    elif hasattr(background, "__len__"):
        background = numpy.array(background).astype("float")
        if len(background) != len(pssm):
            raise Exception(
                "in ppsm_freq_to_loglikelihood background given as array but length not compatible with pssm (respectively %d and %d)"
                % (len(background), len(pssm))
            )
    else:
        raise Exception(
            "in ppsm_freq_to_loglikelihood background=%s but should be 'auto','uniform' or array\n"
            % (str(background))
        )
    background[background <= 0] = min(background[background > 0]) / 4.0
    pssm[pssm <= 0] = (background).min() / 10.0
    if background.shape == pssm.shape:
        mat = numpy.log2((pssm / background))
    else:  # background is a vector of ground frequencies (not per position)
        mat = numpy.log2((pssm.T / background).T)
    if value_for_zerofreqs is not None:
        Min = numpy.nanmin(mat[numpy.isfinite(mat)])
        if Min < value_for_zerofreqs:
            value_for_zerofreqs = int(Min - 2)  # e.g. -11.68 -> -13 like this
        # print ('DEB: pssm Min=',Min,'==> value_for_zerofreqs=',value_for_zerofreqs)
        mat[numpy.isinf(mat)] = value_for_zerofreqs
    return mat




def is_protein(seq, aa_list=amino_list1):
    """
    return True,None
    or return False, non_prot_res
    """
    for aa in seq:
        if aa not in aa_list:
            return False, aa
    return True, None

# given a complete path to a file (or just a file name) it returns (path,name,extension)
def get_file_path_and_extension(complete_filename):
    if "/" in complete_filename:
        path = complete_filename[: complete_filename.rfind("/") + 1]
        rest = complete_filename[complete_filename.rfind("/") + 1 :]
        if "." in rest:
            name = rest[: rest.find(".")]
            extension = rest[rest.find(".") :]
        else:
            name = rest
            extension = ""
    else:
        path = ""
        if "." in complete_filename:
            name = complete_filename[: complete_filename.find(".")]
            extension = complete_filename[complete_filename.find(".") :]
        else:
            name = complete_filename
            extension = ""
    return (path, name, extension)

def distance_matrix_from_aligned_sequences(
    aligned_seqs,
    matrix_for_distance=distance_blosum_normalised,
    alphabet_dict=blosum_alph,
    RegionOfInterest_indices=None,
    only_first_N_sequences=False,
    return_number_of_mutations=False,
    quiet=False,
):
    """
    return distance_similarity
    if return_number_of_mutations return a 'distance' that is the total number of mutations
        return distance_similarity,distance_n_mutations
    this function calculate a distance matrix in a way rather optimised for speed
     returned only as flattend triangle of symmetrix matrix, if needed one can use squareform
            (from scipy.spatial.distance import squareform) to make it square
    if only_first_N_sequences is given it will return a non-square matrix of shape
        (only_first_N_sequences,len(aligned_seqs))
    matrix_for_distance: the calculation is based on the inter-residue distances given in matrix_for_distance
    note that the expectation is that diagonal elements are 0 (0 distance among identical residues).
    alphabet_dict is like {'A':0,'C':1,...} with the indices of each amino acid in the given matrix_for_distance
    RegionOfInterest_indices can be given as those indices corresponding to the alignment column to be used for distance calculations
     (this is useful to calculate the distance excluding specific regions, e.g. CDR3 of antibodies that are anyway highly diverse)
    """
    sta = time.time()
    mat_inds = (
        []
    )  # first covert aligned sequence into indeces of matrix_for_distance, which will be much faster in double loop.
    if RegionOfInterest_indices is not None:
        RegionOfInterest_indices = numpy.array(RegionOfInterest_indices)
        for r in aligned_seqs:
            mat_inds += [
                [
                    alphabet_dict[aa.upper()]
                    for aa in numpy.array(list(r))[RegionOfInterest_indices]
                ]
            ]  # NOTE .upper() for q and other possible guesses
    else:
        for r in aligned_seqs:
            mat_inds += [
                [alphabet_dict[aa.upper()] for aa in r]
            ]  # NOTE .upper() for q and other possible guesses
    N = len(mat_inds)
    distance_similarity = numpy.empty((N * (N - 1)) // 2)  # allocate your memory
    n = 0
    if return_number_of_mutations:
        distance_n_mutations = numpy.empty((N * (N - 1)) // 2)
        number_of_mutations = numpy.ones(matrix_for_distance.shape)
        numpy.fill_diagonal(number_of_mutations, 0)
        if "X" in alphabet_dict:
            number_of_mutations[alphabet_dict["X"], alphabet_dict["X"]] = 1
        for j in range(len(mat_inds) - 1):
            distance_similarity[n : n + N - 1 - j] = (
                matrix_for_distance[mat_inds[j + 1 :], mat_inds[j]]
            ).sum(
                axis=1
            )  # mat.shape[1] is the length of the aligned sequences
            distance_n_mutations[n : n + N - 1 - j] = (
                number_of_mutations[mat_inds[j + 1 :], mat_inds[j]]
            ).sum(axis=1)
            if only_first_N_sequences and j + 1 >= only_first_N_sequences:
                distance_similarity = distance_similarity[: n + N - 1 - j]
                distance_n_mutations = distance_n_mutations[: n + N - 1 - j]
                Md = numpy.zeros((only_first_N_sequences, N))
                Mn = numpy.zeros((only_first_N_sequences, N))
                ns = 0
                for js in range(only_first_N_sequences):
                    Md[js][js + 1 :] = distance_similarity[ns : ns + N - 1 - js]
                    Mn[js][js + 1 :] = distance_n_mutations[ns : ns + N - 1 - js]
                    ns += N - 1 - js
                Md[:, :only_first_N_sequences] += Md[:, :only_first_N_sequences].T
                Mn[:, :only_first_N_sequences] += Mn[:, :only_first_N_sequences].T
                distance_similarity = Md
                distance_n_mutations = Mn
                break
            n += N - 1 - j
        if not quiet:
            print(
                "distance matrix and distance_n_mutations computed for %d sequences [only_first_N_sequences=%s], took %.1lf s"
                % (
                    len(aligned_seqs),
                    str(only_first_N_sequences),
                    (time.time() - sta),
                )
            )  # for 4-mer 1.6*10^5 sequences took 388.76327562332153 on circe
        return distance_similarity, distance_n_mutations
    else:
        for j in range(len(mat_inds) - 1):
            distance_similarity[n : n + N - 1 - j] = (
                matrix_for_distance[mat_inds[j + 1 :], mat_inds[j]]
            ).sum(
                axis=1
            )  # mat.shape[1] is the length of the aligned sequences
            if only_first_N_sequences and j + 1 >= only_first_N_sequences:
                distance_similarity = distance_similarity[: n + N - 1 - j]
                Md = numpy.zeros((only_first_N_sequences, N))
                ns = 0
                for js in range(only_first_N_sequences):
                    Md[js][js + 1 :] = distance_similarity[ns : ns + N - 1 - js]
                    ns += N - 1 - js
                Md[:, :only_first_N_sequences] += Md[:, :only_first_N_sequences].T
                distance_similarity = Md
                break
            n += N - 1 - j
        if not quiet:
            print(
                "distance matrix computed for %d sequences [only_first_N_sequences=%s], took %s s"
                % (
                    len(aligned_seqs),
                    str(only_first_N_sequences),
                    str(time.time() - sta),
                )
            )  # for 4-mer 1.6*10^5 sequences took 388.76327562332153 on circe
        return distance_similarity



def renumber_Fv_pdb_file(pdbf, H, L ,scheme='IMGT',outfilename=None) :
    '''
    If L=None, will only consider the Heavy chain (suitable for VHHs)
    return outfilename,pdb_respos_to_IMGT,IMGT_to_pdb_respos,Chains_Fv_res,original_polymer
    H and L are ids of heavy and light (only one heavy and one light currently supported 
        (can run multiple time by giving output file as input and new set of H and L))
    uses anarci
    create scheme (e.g. IMGT) numbered file printed to outfilename
    when it's None outfilename = pdb_id+scheme+'.pdb'
    Note IMGT is a bit weird (check on structure downloaded from SabDab) for 
     example it has 111, 111A, 112A, 112 (in this order, see 7so5)
    '''
    if outfilename is None :
        outfilename=pdb_id+scheme+'.pdb'
    patht,pdb_id,ext=misc.get_file_path_and_extension(pdbf)

    polymer,sruct, ovmap = structs.pdb_to_polymer(pdbf)
    Chains_Fv_res = {}
    pdb_respos_to_IMGT={}
    IMGT_to_pdb_respos={}
    ch=H
    Fv_res = get_antibodyVD_numbers(polymer[ch].sequenceOK, scheme=scheme, full_return=True, seqname='H', print_warns=False, auto_detect_chain_type=True)
    chty = list(Fv_res.keys())[0] # ["H", "K", "L"]
    Chains_Fv_res[ch]=Fv_res[chty]
    pdb_respos_to_IMGT[ch]={}
    IMGT_to_pdb_respos[ch]={}
    seqind_to_schemnum,schemnum_to_seqind,seqind_regions,warnings, info_dict, eval_table = Fv_res[chty]
    for j,res in enumerate(polymer[ch].seq) :
        pdb_respos_to_IMGT[ch][res.pdb_id] = seqind_to_schemnum[j]
        IMGT_to_pdb_respos[ch][seqind_to_schemnum[j]] = res.pdb_id
    if L!=None:
        ch=L
        Fv_res = get_antibodyVD_numbers(polymer[ch].sequenceOK,scheme=scheme,auto_cdr_scheme=True,full_return=True,seqname='L')
        chtyL = list(Fv_res.keys())[0]
        Chains_Fv_res[ch]=Fv_res[chtyL]
        pdb_respos_to_IMGT[ch]={}
        IMGT_to_pdb_respos[ch]={}
        seqind_to_schemnum,schemnum_to_seqind,seqind_regions,warnings, info_dict, eval_table = Fv_res[chtyL]
        for j,res in enumerate(polymer[ch].seq) :
            pdb_respos_to_IMGT[ch][res.pdb_id] = seqind_to_schemnum[j]
            IMGT_to_pdb_respos[ch][seqind_to_schemnum[j]] = res.pdb_id
    out = open(outfilename,'w')
    remark_added=False
    max_respos={}
    with open(pdbf) as inp :
        for line in inp :
            if line.startswith("ATOM") :
                if not remark_added :
                    if L!=None:
                        out.write('REMARK   6 Fv residues renumbered with %s numbering scheme [%s:%s and %s:%s]\n'%(scheme,H,chty,L,chtyL))
                    else:
                        out.write('REMARK   6 Fv residues renumbered with %s numbering scheme [%s:%s]\n'%(scheme,H,chty))
                    remark_added=True
                ch=line[21]
                if ch in Chains_Fv_res : # chain ID
                    respos = line[22:27].strip()
                    try : 
                        respos = int(respos)
                        if ch not in max_respos :
                            max_respos[ch]=respos
                            delta=None
                    except Exception : 
                        pass
                    if respos in pdb_respos_to_IMGT[ch] :
                        pdb_respos_to_IMGT[ch][respos]
                        new_respos = '%5s'%(str(pdb_respos_to_IMGT[ch][respos][0])+str(pdb_respos_to_IMGT[ch][respos][1])) # second most often ' '
                        line = line[:22]+new_respos+line[27:]
                        if type(respos) is int and respos > max_respos[ch] :
                            max_respos[ch]=respos
                    elif respos<= max_respos[ch] : # renumber after Fv to avoid duplicated positions
                        if delta is None : delta=max_respos[ch]-respos
                        new_respos = '%4d '%(respos+delta+1)
            out.write(line)

    return outfilename,pdb_respos_to_IMGT,IMGT_to_pdb_respos,Chains_Fv_res,polymer

def run_paragraph_on_Fv_pdbfiles(pdbf_list, H_list, L_list ,threshold=0.734,outfile='Paragraph_output.csv',imgtfolder='IMGT_pdbs'):
    '''
    to predict antibody paratope
    requires anarci  so Paragraph SHOULD BE ISNTALLED in your x86 environment to run
    first renumbers pdb file to IMGT (required)
    then run paragraphs, then adds column to input corresponding to initial pdb numbering
    # very complicated to run on pdb file importing Paragraph, so just runs from command line
    return output_csv_data,paratope_residues_pdbrespos,paratope_residues_IMGT
    '''
    if type(pdbf_list) is str :
        pdbf_list=[pdbf_list]
        H_list=[H_list]
        L_list=[L_list]
    tmp_HLcsv = open('Tmp_pdb_H_L.csv','w')
    if imgtfolder=='.' : imgtfolder=''
    else :
        if imgtfolder[-1]!='/' : imgtfolder+='/'
        os.system('mkdir '+imgtfolder)
    pdbf_IMGT_to_pdb_respos={}
    polymers={}
    for j,pdbf in enumerate(pdbf_list) :
        H= H_list[j]
        L= L_list[j]
        patht,pdb_id,ext= misc.get_file_path_and_extension(pdbf)
        outfilename,pdb_respos_to_IMGT,IMGT_to_pdb_respos,Chains_Fv_res,original_polymer = renumber_Fv_pdb_file(pdbf, H, L ,scheme='IMGT',outfilename=imgtfolder+pdb_id+'IMGT.pdb')
        pdbf_IMGT_to_pdb_respos[pdb_id] = IMGT_to_pdb_respos.copy()
        polymers[pdb_id] = original_polymer
        tmp_HLcsv.write('%s,%s,%s\n'%(pdb_id+'IMGT',H,L))
    tmp_HLcsv.close()
    if L!=[None]:
        command = 'Paragraph --pdb_H_L_csv %s --pdb_folder_path %s --out_path %s'%('Tmp_pdb_H_L.csv',imgtfolder,outfile)
    else: command = 'Paragraph --pdb_H_L_csv %s --pdb_folder_path %s --out_path %s --heavy'%('Tmp_pdb_H_L.csv',imgtfolder,outfile)
    print(command)
    os.system(command)
    output=csv_dict.Data(fname=outfile,key_column=None)
    paratope_residues_IMGT={}
    paratope_residues_pdbrespos={}
    for k in output :
        pdb=output[k][output.hd['pdb']].replace('IMGT','')
        ch = output[k][output.hd['chain_id']]
        imgt = output[k][output.hd['IMGT']]
        if type(imgt) is int or imgt[-1].isdigit() : imgt=(imgt,' ')
        else : imgt=(int(imgt[:-1]),imgt[-1])
        respos=pdbf_IMGT_to_pdb_respos[pdb][ch][imgt]
        aa = ThreeToOne[output[k][output.hd['AA']]]
        # check mapping is correct
        if polymers[pdb][ch].seq[polymers[pdb][ch].aa_pdb_id_map[respos]].res1letter != aa :
            sys.stderr.write('**ERROR** mapping issue in run_paragraph_on_Fv_pdbfiles (likely in going back from IMGT numbering)! Expecting residue %s from Paragraph but found %s instead [pdb %s ch %s IMGT %s mapped pdb respos %s]\n'%(aa,polymers[pdb][ch].seq[polymers[pdb][ch].aa_pdb_id_map[respos]],pdb,ch,imgt,respos ))
        parat =False
        if output[k][output.hd['pred']] >= threshold :
            parat=True
            if pdb not in paratope_residues_IMGT :
                paratope_residues_IMGT[pdb]=[]
                paratope_residues_pdbrespos[pdb]=[]
            paratope_residues_IMGT[pdb]+=[ '%s%s.%s'%(aa,output[k][output.hd['IMGT']],ch) ]
            paratope_residues_pdbrespos[pdb]+=[ '%s%s.%s'%(aa,respos,ch) ]
        output[k]+=[ respos,aa , pdb, parat ]
    output._update_hd('respos_pdb')
    output._update_hd('res1letter')
    output._update_hd('pdb_id')
    output._update_hd('is_paratope')
    output.Print(outfile)
    os.system('rm Tmp_pdb_H_L.csv')
    return output,paratope_residues_pdbrespos,paratope_residues_IMGT

