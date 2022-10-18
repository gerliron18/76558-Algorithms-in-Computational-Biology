import argparse
import itertools
from scipy.special import logsumexp
from functools import reduce
import numpy as np
import motif_find as mf

nucleotideDict = {"^": 0, "A": 1, "C": 2, "G": 3, "T": 4, "$": 5}
nucleotideArr = ['A', 'C', 'G', 'T', '^', '$']


def parse_args():
    """
    Parse the command line arguments.
    :return: The parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta', help='File path with list of sequences (e.g. seqs_ATTA.fasta)')
    parser.add_argument('seed', help='Guess for the motif (e.g. ATTA)')
    parser.add_argument('p', type=float, help='Initial guess for the p transition probability (e.g. 0.01)')
    parser.add_argument('q', type=float, help='Initial guess for the q transition probability (e.g. 0.9)')
    parser.add_argument('alpha', type=float, help='Softening parameter for the initial profile (e.g. 0.1)')
    parser.add_argument('convergenceThr', type=float, help='ll improvement threshold for the stopping condition'
                                                           ' (e.g. 0.1)')
    return parser.parse_args()


def read_fasta_file(fileName):
    seqArr = []
    f = open(fileName)

    lines = (x[1] for x in itertools.groupby(f, lambda line: line.startswith(">")))
    for line in lines:
        seq = "".join(s.strip() for s in next(lines))
        seq = "^" + seq + "$"
        seqArr.append(seq)

    return seqArr


def generate_history(likelihoodArr):
    f = open("ll_history.txt", "w+")

    for likelihood in range(len(likelihoodArr)):
        f.write(str(likelihoodArr[likelihood]) + "\n")

    f.close()


def reshape_vec(row_vec):
    vec = np.asarray(row_vec)

    return vec.reshape(vec.size, 1)


def calculate_p_q(N_kl):
    q = N_kl[0, 1] - logsumexp(N_kl[0, :])

    temp = np.logaddexp(N_kl[1, 2], N_kl[-2, -1])
    p = temp - reduce(np.logaddexp, [N_kl[1, 2], N_kl[1, 1], N_kl[-2, -1], N_kl[-2, -2]])

    return p, q


def generate_motif_profile(transitionTable, emissionTable, motif_len):
    f = open("motif_profile.txt", "w+")
    emissionTable = np.exp(emissionTable)
    profiles = emissionTable[2: motif_len + 2, :-2]

    for i in range(profiles.shape[1]):
        for j in range(profiles.shape[0]):
            f.write(str(round(profiles[j][i], 4)) + "\t")

        f.write("\n")

    p = transitionTable[1, 2]
    q = transitionTable[0, 1]

    f.write(str(round(np.exp(p), 4)) + "\n")
    f.write(str(round(np.exp(q), 4)))


    f.close()


def generate_motif_positions(transitionTable, emissionTable, seqArr, motif_len):
    viterbiSeqArr = []
    [viterbiSeqArr.append(mf.viterbiAlg(transitionTable, emissionTable, seq, motif_len)) for seq in seqArr]

    f = open("motif_positions.txt", "w+")

    for seq in range(len(viterbiSeqArr)):
        f.write(str(viterbiSeqArr[seq].find("M")) + "\n")

    f.close()


def updateTables(forwardTable, backwardTable, states_count):
    with np.errstate(divide='ignore'):
        forwardTable = np.concatenate((np.log(np.zeros((states_count, 1))), forwardTable), axis=1)
        backwardTable = np.concatenate((backwardTable, np.log(np.zeros((states_count, 1)))), axis=1).T

    return forwardTable, backwardTable


def calculate_log(first, second):
    firstStack = np.stack([first] * first.shape[0]).transpose(2, 1, 0)
    secondStack = np.stack([second] * second.shape[1]).transpose(1, 0, 2)

    return logsumexp(firstStack + secondStack, axis=0)


def updateNkl(N_kl, nucleotide_matrix, emissionTable, transitionTable, current_ll, seqArr,
              states_count):
    for seq in seqArr:
        forwardTable = mf.forwardAlg(emissionTable, transitionTable, states_count, seq)
        backwardTable = mf.backwardAlg(emissionTable, transitionTable, states_count, seq)

        likelihood = forwardTable[-1][-1]
        backwardTable -= likelihood
        current_ll += likelihood
        matrix = forwardTable + backwardTable

        permutation = [1, 2, 3, 4, 0, 5]
        index = np.empty_like(permutation)
        index[permutation] = np.arange(len(permutation))
        emission = emissionTable[:, index]

        for nuc in range(len(nucleotideArr)):
            currentIndex = np.transpose(np.nonzero(np.array(list(seq)) == nucleotideArr[nuc]))
            matrixLocation = np.apply_along_axis(lambda x: x[currentIndex], 1, matrix).reshape(states_count, -1)
            nucleotide_matrix[nuc] = np.hstack((nucleotide_matrix[nuc], matrixLocation))

            num = reshape_vec(emission[:, nucleotideDict[nucleotideArr[nuc]]])
            backwardTable[:, np.reshape(currentIndex, -1, )] += num

        forwardTable, backwardTable = updateTables(forwardTable, backwardTable, states_count)
        N_kl = np.logaddexp(N_kl, calculate_log(forwardTable, backwardTable))

    return N_kl, current_ll


def main():
    np.seterr(all='warn', invalid='ignore', divide='ignore')
    args = parse_args()
    seqArr = read_fasta_file(args.fasta)

    transitionTable = mf.generateTransition(args.seed, args.p, args.q)
    emissionTable = mf.generateEmission(args.seed, args.alpha)
    best_ll = np.NINF
    likelihoodArr = []

    while True:
        current_ll = 0
        nucleotide_matrix = []

        # Initialize N_kl and N_kx
        N_kl = np.log(np.zeros((len(args.seed) + 4, len(args.seed) + 4)))
        N_kx = np.zeros((len(args.seed) + 4, 0))

        [nucleotide_matrix.append(np.empty((len(args.seed) + 4, 0))) for nuc in range(len(nucleotideArr))]

        # Update N_kl and calculate current log-likelihood
        N_kl, current_ll = updateNkl(N_kl, nucleotide_matrix, emissionTable, transitionTable, current_ll, seqArr, len(args.seed) + 4)
        N_kl = N_kl + transitionTable

        # Update N_kx
        for nuc in range(len(nucleotideArr)):
            N_kx = np.hstack((N_kx, reshape_vec(logsumexp(nucleotide_matrix[nuc], axis=1))))


        # Check if need to break from loop
        if abs(current_ll - best_ll) <= args.convergenceThr:
            break

        # Add the current likelihood to the array
        likelihoodArr.append(current_ll)

        # Calculate the new p, q values
        p, q = calculate_p_q(N_kl)

        # Update transition & emission matrices
        transitionTable = mf.generateTransition(args.seed, np.exp(p),np.exp(q))
        emissionTable[2:2 + len(args.seed), 0:-2] = (N_kx[:, 0:-2] - reshape_vec(logsumexp(N_kx[:, 0:-2], axis=1)))[2:2 + len(args.seed),:]


        best_ll = current_ll

    # Create output files
    generate_history(likelihoodArr)
    generate_motif_profile(transitionTable, emissionTable, len(args.seed))
    generate_motif_positions(transitionTable, emissionTable, seqArr, len(args.seed))


if __name__ == "__main__":
    main()
