import argparse
import numpy as np
from itertools import groupby

GLOBAL = 'global'
LOCAL = 'local'
OVERLAP = 'overlap'

GAP = '-'
LEFT = 0
UP = 1
DIAGONAL = 2
STOP = 3

# Dictionary of the nucleotides and their integer representation at the program
nuc_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}


def fastaread(fasta_name):
    """
    Read a fasta file. For each sequence in the file, yield the header and the actual sequence.
    In Ex1 you may assume the fasta files contain only one sequence.
    You may keep this function, edit it, or delete it and implement your own reader.
    """
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq


def getScore(seq_a, seq_b, score_matrix, row, column):
    """
    Returns the penalty score of two nucleotides
    """
    return score_matrix[nuc_dict[seq_b[row]]][nuc_dict[seq_a[column]]]


def getGapScore(seq, index, score_matrix):
    """
    Returns the penalty score of a nucleotide and a gap
    """
    return score_matrix[nuc_dict[seq[index]]][nuc_dict[GAP]]


def sequencesPrinter(seq_a, seq_b):
    """
    A sequences printer function will get to sequences and print them to the
    screen according to the exercise instruction.
    :param seq_a: The first sequence
    :param seq_b: The second sequence
    """
    maxSeqLength = max(len(seq_a), len(seq_b))
    for i in range(0, maxSeqLength, 50):
        print(seq_a[i:i + 50])
        print(seq_b[i:i + 50] + "\n")


def fillMatrices(align_matrix, path_matrix, score_matrix, seq_a, seq_b,
                 align_type):
    """
    The filling matrices function fill the alignment and path matrices with
    integer scores according to the two given sequences, the align type and the
    score matrix.
    :param align_matrix: A matrix represents the scores of each alignment using
                         dynamic programming
    :param path_matrix: A matrix represents the path to get the best alignment
                        according to the scores at the align_matrix
    :param score_matrix: The user given score matrix
    :param seq_a: The first sequence
    :param seq_b: The second sequence
    :param align_type: Specifier of the alignment type-global, local or overlap
    :return: The align_matrix and the path_matrix after filling them
    """
    for column in range(align_matrix.shape[1]):
        for row in range(align_matrix.shape[0]):
            if not column and not row:
                align_matrix[row][column] = 0
                path_matrix[row][column] = -1
            elif not column:
                if align_type == LOCAL:
                    align_matrix[row][column] = 0
                else:
                    align_matrix[row][column] = align_matrix[row - 1][column] + getScore(seq_a, seq_b, score_matrix,
                                                                                         row, column)
                path_matrix[row][column] = 1

            elif not row:
                if align_type != GLOBAL:
                    align_matrix[row][column] = 0
                else:
                    align_matrix[row][column] = align_matrix[row][column - 1] + getScore(seq_a, seq_b, score_matrix,
                                                                                         row, column)
                path_matrix[row][column] = 0

            else:
                scores = {}
                scores[LEFT] = align_matrix[row][column - 1] + getGapScore(seq_a, column, score_matrix)
                scores[UP] = align_matrix[row - 1][column] + getGapScore(seq_b, row, score_matrix)
                scores[DIAGONAL] = align_matrix[row - 1][column - 1] + getScore(seq_a, seq_b, score_matrix, row, column)

                if align_type == LOCAL:
                    scores[STOP] = 0

                # Insert the best score to the alignment matrix and nuc
                # align_type to the path matrix
                nuc_type = max(scores, key=scores.get)
                path_matrix[row][column] = nuc_type
                align_matrix[row][column] = scores[nuc_type]

    return align_matrix, path_matrix


def generateSeq(first_seq, second_seq, path_matrix, row_index, col_index):
    """
    The generating sequences function using the path matrix to create two new
    sequences, based on the original two, after the best score alignment
    found.
    :param first_seq: The first sequence
    :param second_seq: The second sequence
    :param path_matrix: A matrix represents the path to the sequence founded as
                        the best alignment
    :param row_index: The current row index
    :param col_index: The current column index
    :return: The two sequences after alignment
    """
    seq_a = ""
    seq_b = ""

    while col_index >= 0 or row_index >= 0:
        path = path_matrix[row_index, col_index]

        if path == LEFT:
            seq_a = first_seq[col_index - 1] + seq_a
            seq_b = GAP + seq_b
            col_index -= 1
        elif path == UP:
            seq_a = GAP + seq_a
            seq_b = second_seq[row_index - 1] + seq_b
            row_index -= 1
        elif path == DIAGONAL:
            seq_a = first_seq[col_index - 1] + seq_a
            seq_b = second_seq[row_index - 1] + seq_b
            row_index -= 1
            col_index -= 1
        elif path == STOP:
            break
        elif path == -1:
            break

    return seq_a, seq_b


def alignment(seq_a, seq_b, score_matrix, align_type):
    """
    The alignment function will manage the alignment according to its type
    and given score matrix.
    :param seq_a: The first sequence
    :param seq_b: The second sequence
    :param score_matrix: The user given score matrix
    :param align_type: Specifier of the alignment type-global, local or overlap
    :return: The score of the best alignment found by the program
    """
    score = 0

    seq_a = GAP + seq_a
    seq_b = GAP + seq_b
    len_a = len(seq_a)
    len_b = len(seq_b)

    align_matrix = np.zeros((len_b, len_a))
    path_matrix = np.full((len_b, len_a), fill_value=-1)

    align_matrix, path_matrix = fillMatrices(align_matrix, path_matrix,
                                             score_matrix, seq_a, seq_b,
                                             align_type)

    col_index = len_a - 1
    row_index = len_b - 1

    if align_type == LOCAL:
        score = np.amax(align_matrix)
        found_it = np.where(align_matrix == score)
        col_index = found_it[1][0]
        row_index = found_it[0][0]

    elif align_type == OVERLAP:

        current_col = align_matrix[:, len_a - 1]
        score = np.max(current_col)
        row_index = np.where(current_col == score)[0][0]
        overlap = len_b - (row_index + 1)

    seq_b_cpy = seq_b

    seq_a, seq_b = generateSeq(seq_a[1:], seq_b[1:], path_matrix, row_index,
                               col_index)

    if align_type == OVERLAP:
        seq_a = seq_a + GAP * overlap
        seq_b = seq_b + seq_b_cpy[row_index + 1:]

    sequencesPrinter(seq_a, seq_b)

    if align_type == GLOBAL:
        score = align_matrix[len_b - 1][len_a - 1]

    return score


def main():
    """
    The main function of the program. Will parse all given parameters from
    the user and preform sequences alignment according to the user request.
    Will print to the screen the alignment type and the best score.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_a',
                        help='Path to first FASTA file (e.g. fastas/HomoSapiens-SHH.fasta)')
    parser.add_argument('seq_b', help='Path to second FASTA file')
    parser.add_argument('--align_type',
                        help='Alignment align_type (e.g. local)',
                        required=True)
    parser.add_argument('--score',
                        help='Score matrix in.tsv format (default is score_matrix.tsv) ',
                        default='score_matrix.tsv')
    command_args = parser.parse_args()

    seq_a = next(fastaread(command_args.seq_a))[1]
    seq_b = next(fastaread(command_args.seq_b))[1]
    score_matrix = np.loadtxt(fname=command_args.score, delimiter='\t',
                              skiprows=1, usecols=(1, 2, 3, 4, 5))

    if command_args.align_type == GLOBAL:
        score = alignment(seq_a, seq_b, score_matrix, GLOBAL)
    elif command_args.align_type == LOCAL:
        score = alignment(seq_a, seq_b, score_matrix, LOCAL)
    elif command_args.align_type == OVERLAP:
        score = alignment(seq_a, seq_b, score_matrix, OVERLAP)

    print(command_args.align_type + ":" + str(int(score)))


if __name__ == '__main__':
    main()
