import argparse
import numpy as np
from scipy.special import logsumexp

nucleotideDict = {"A": 0, "C": 1, "G": 2, "T": 3, "^": 4, "$": 5}


def generateEmission(initial_emission):
    """
    Generates a Numpy Matrix from the emission table and calculate how many states in the motif.
    The emission table will also contain the states of start, end and non-motif state.
    :param initial_emission: the motif emissions as .tsv file.
    :return: How many states there are in the motif and the emission table.
    """
    # Read the initial_emission table to np matrix and calculate k
    f = open(initial_emission)
    initial_matrix = np.genfromtxt(fname=f, delimiter="\t", skip_header=1,
                                   filling_values=1)
    k = np.size(initial_matrix, 0)

    # Generating new emission matrix
    finalEmission = np.zeros((k + 4, 6))
    finalEmission[0][4] = 1  # start state
    finalEmission[1][0:4] = 0.25  # non-motif

    row, column = 2, 0
    finalEmission[row: row + initial_matrix.shape[0],
    column: column + initial_matrix.shape[1]] += initial_matrix

    finalEmission[k + 2][0:4] = 0.25  # non-motif
    finalEmission[k + 3][5] = 1  # end state

    return finalEmission, k


def generateTransition(k, p, q):
    """
    Generates Transition Table, in which the values represents the probability to move from one
    state type to another.
    :param k: The size of the motif.
    :param p: The probability to move from non-motif to motif, and from non-motif to end.
    :param q: The probability to move from start to non-motif in seq where motif exists.
    :return: The Transition table with all the odds to move from one state to another.
    """
    # Generating new transition matrix
    finalTransition = np.zeros((k + 4, k + 4))

    # Fill Bstart transitions
    finalTransition[0][1] = q
    finalTransition[0][k + 2] = 1 - q

    # Fill B1 transitions
    finalTransition[1][1] = 1 - p
    finalTransition[1][2] = p

    # Fill motif transitions
    for i in range(k):
        finalTransition[2 + i][2 + i + 1] = 1

    # Fill B2 transitions
    finalTransition[k + 2][k + 2] = 1 - p
    finalTransition[k + 2][k + 3] = p

    return finalTransition


def generateViterbiSeq(scoreMatrix, traceMatrix, k, seq):
    """
    Draws the best alignment of sequence and motif.
    """
    currentVec = scoreMatrix[:, len(seq) - 1]
    maxVal = np.amax(currentVec)
    maxIndex = np.where(currentVec == maxVal)[0][0]

    motifSeq = ""

    for i in range(len(seq) - 1, 1, -1):
        if (traceMatrix[maxIndex][i] < 2) or (
                traceMatrix[maxIndex][i] >= (k + 2)):
            motifSeq += "B"
        else:
            motifSeq += "M"

        maxIndex = int(traceMatrix[maxIndex][i])

    motifSeq = motifSeq[::-1]
    return motifSeq


def viterbiAlg(emissionTable, transitionTable, k, seq):
    """
    Will calculate the best possible option to the sequence and it's motif position.
    Will generate a possibilities matrix and use an helper function to draw the best one.
    :return: the best alignment of sequence and motif.
    """
    # Generating new score and trace matrices
    scoreMatrix = np.zeros((k + 4, len(seq)))
    traceMatrix = np.zeros((k + 4, len(
        seq)))  # the Trace matrix will used to find the best trail.
    scoreMatrix[0][0] = 1

    with np.errstate(divide='ignore'):
        scoreMatrix = np.log(scoreMatrix)

    for nucleotide in range(1, len(seq)):
        scoreVec = np.reshape(scoreMatrix[:, nucleotide - 1], k + 4)
        emissionVec = np.transpose(emissionTable)[
            nucleotideDict[seq[nucleotide]]]
        scoreMatrix[:, nucleotide] = np.max(
            np.add(scoreVec, np.transpose(transitionTable)),
            axis=1) + emissionVec
        traceMatrix[:, nucleotide] = np.argmax(
            np.add(scoreVec, np.transpose(transitionTable)), axis=1)

    motifSeq = generateViterbiSeq(scoreMatrix, traceMatrix, k, seq)

    sequencesPrinter(seq, motifSeq)
    return motifSeq


def forwardAlg(emissionTable, transitionTable, k, seq):
    """
    Finds the best motif alignment by forwarding each time one letter on the sequence.
    :param k: size of motif.
    """
    # Generating new score matrix of forward algorithm with initialize value
    scoreMatrix = np.zeros((k + 4, len(seq)))
    scoreMatrix[0][0] = 1

    # Deal with log
    with np.errstate(divide='ignore'):
        scoreMatrix = np.log(scoreMatrix)

    # Fill the scoreMatrix
    for nucleotide in range(1, len(seq)):
        scoreVec = np.reshape(scoreMatrix[:, nucleotide - 1], k + 4)
        emissionVec = np.transpose(emissionTable)[
            nucleotideDict[seq[nucleotide]]]
        finalScore = logsumexp(np.add(scoreVec, np.transpose(transitionTable)),
                               axis=1) + emissionVec  # will use adding insted of multiplication cause using logaritmic rules.
        scoreMatrix[:, nucleotide] = finalScore

    return scoreMatrix


def backwardAlg(emissionTable, transitionTable, k, seq):
    """
    Finds the best motif alignment by backing each time one letter on the sequence, start from the end.
    :param k: size of motif.
    """
    # Generating new score matrix of backward algorithm with initialize value
    scoreMatrix = np.zeros((k + 4, len(seq)))
    scoreMatrix[k + 3][len(seq) - 1] = 1

    # Deal with log
    with np.errstate(divide='ignore'):
        scoreMatrix = np.log(scoreMatrix)

    # Fill the scoreMatrix
    for nucleotide in range(len(seq) - 1, 0, -1):
        scoreVec = np.reshape(scoreMatrix[:, nucleotide], k + 4)
        emissionVec = np.transpose(emissionTable)[
            nucleotideDict[seq[nucleotide - 1]]]
        finalScore = logsumexp(np.add(scoreVec, transitionTable),
                               axis=1) + emissionVec
        scoreMatrix[:, nucleotide - 1] = finalScore

    return scoreMatrix


def sequencesPrinter(seq, motifSeq):
    """
    Prints the sequence as lines of 50 bp each
    """
    # Remove redundant characters
    seq = seq[1: -1]

    while len(seq) > 0 and len(motifSeq) > 0:
        if len(motifSeq) > 0:
            print(motifSeq[: min(50, len(seq))])
            motifSeq = motifSeq[50:]

        if len(seq) > 0:
            print(seq[: min(50, len(seq))])
            seq = seq[50:]

        print()


def posteriorAlg(emissionTable, transitionTable, k, seq):
    """
    Finds the best fit posterior to the given sequence.
    """
    forwardMatrix = forwardAlg(emissionTable, transitionTable, k, seq)
    backwardMatrix = backwardAlg(emissionTable, transitionTable, k, seq)

    likelihood = forwardMatrix[-1][-1]
    motifSeq = ""

    for i in range(1, len(seq) - 1):
        maxVal = 0
        maxK = 0

        for j in range(k + 4):
            forwardVal = forwardMatrix[j][i]
            backwardVal = backwardMatrix[j][i]

            with np.errstate(invalid='ignore'):
                value = forwardVal + backwardVal - likelihood

            if (value > maxVal) or (maxVal == 0):
                maxVal = value
                maxK = j

        if (maxK < 2) or (maxK >= k + 2):
            motifSeq += "B"
        else:
            motifSeq += "M"

    sequencesPrinter(seq, motifSeq)
    return motifSeq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)',
                        required=True)
    parser.add_argument('seq',
                        help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
    parser.add_argument('initial_emission',
                        help='Path to emission table (e.g. initial_emission.tsv)')
    parser.add_argument('p', help='transition probability p (e.g. 0.01)',
                        type=float)
    parser.add_argument('q', help='transition probability q (e.g. 0.5)',
                        type=float)
    args = parser.parse_args()

    emissionTable, k = generateEmission(args.initial_emission)
    transitionTable = generateTransition(k, args.p, args.q)

    # We will change all our calculation to log to avoid underflow.
    with np.errstate(divide='ignore'):
        emissionTable = np.log(emissionTable)
        transitionTable = np.log(transitionTable)

    newSeq = "^" + args.seq + "$"  # '^' and '$' will represent the start and end positions.

    if args.alg == 'viterbi':
        return viterbiAlg(emissionTable, transitionTable, k, newSeq)

    elif args.alg == 'forward':
        scoreMatrix = forwardAlg(emissionTable, transitionTable, k, newSeq)
        print("%.2f" % scoreMatrix[-1][-1])

    elif args.alg == 'backward':
        scoreMatrix = backwardAlg(emissionTable, transitionTable, k, newSeq)
        print("%.2f" % scoreMatrix[0][0])

    elif args.alg == 'posterior':
        return posteriorAlg(emissionTable, transitionTable, k, newSeq)


if __name__ == '__main__':
    main()
