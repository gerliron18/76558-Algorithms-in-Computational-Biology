import argparse
import numpy as np
from scipy.special import logsumexp
import pandas as pd

nucleotideDict = {"A": 0, "C": 1, "G": 2, "T": 3, "^": 4, "$": 5}
emissionDict = {"^": 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, "$": 5}


def applySoftenParam(emissionTable, seed, alpha):
    for nuc in range(len(seed)):
        new = []
        for i in range(4):
            if nucleotideDict[seed[nuc]] == i:
                new.append(1 - 3*alpha)
            else:
                new.append(alpha)

        new.append(0)
        new.append(0)

        emissionTable.append(new)


def generateEmission(seed, alpha):
    """
    Generates a Numpy Matrix from the emission table and calculate how many states in the motif.
    The emission table will also contain the states of start, end and non-motif state.
    :param initial_emission: the motif emissions as .tsv file.
    :return: How many states there are in the motif and the emission table.
    """

    emissionTable = []
    emissionTable.append([0, 0, 0, 0, 1, 0])
    emissionTable.append([0.25, 0.25, 0.25, 0.25, 0, 0])

    applySoftenParam(emissionTable, seed, alpha)

    emissionTable.append([0.25, 0.25, 0.25, 0.25, 0, 0])
    emissionTable.append([0, 0, 0, 0, 0, 1])

    emissionTable = np.log(np.asarray(emissionTable))
    return emissionTable


def generateTransition(seed, p, q):
    """
    Generates Transition Table, in which the values represents the probability to move from one
    state type to another.
    :param motif: The motif.
    :param p: The probability to move from non-motif to motif, and from non-motif to end.
    :param q: The probability to move from start to non-motif in seq where motif exists.
    :return: The Transition table with all the odds to move from one state to another.
    """
    transitionTable = []

    for i in range(len(seed) + 4):
        row = []
        for j in range(len(seed) + 4):
            if (i >= 2 and i < len(seed) + 1 and j == i + 1) or (i == len(seed) + 1 and j == len(seed) + 2):
                row.append(1)
            elif i == 1:
                if j == 2:
                    row.append(p)
                elif j == 1:
                    row.append(1 - p)
                else:
                    row.append(0)
            elif i == len(seed) + 2:
                if j == len(seed) + 2:
                    row.append(1 - p)
                elif j == len(seed) + 3:
                    row.append(p)
                else:
                    row.append(0)
            elif i == 0:
                if j == 1:
                    row.append(q)
                elif j == len(seed) + 2:
                    row.append(1 - q)
                else:
                    row.append(0)
            else:
                row.append(0)
        transitionTable.append(row)

    transitionTable = np.log(np.asarray(transitionTable))

    return transitionTable


def forwardAlg(emissionTable, transitionTable, motif_len, seq):
    """
    Finds the best motif alignment by forwarding each time one letter on the sequence.
    :param k: size of motif.
    """
    # Generating new score matrix of forward algorithm with initialize value
    N = len(seq)
    V = np.zeros(shape=(motif_len, N))

    for k in range(motif_len):
        V[k][0] = emissionTable[k][nucleotideDict[seq[0]]]

    for i in range(1, N):
        logvCol = np.reshape(V[:, i-1], motif_len)
        emission = np.transpose(emissionTable)[nucleotideDict[seq[i]]]
        transition = np.transpose(transitionTable)
        sum = logsumexp(np.add(logvCol, transition), axis=1) + emission
        V[:, i] = sum

    return V


def backwardAlg(emissionTable, transitionTable, motif_len, seq):
    """
    Finds the best motif alignment by backing each time one letter on the sequence, start from the end.
    :param k: size of motif.
    """
    # Generating new score matrix of backward algorithm with initialize value
    N = len(seq)
    V = np.log(np.zeros([motif_len, N]))
    V[-1, -1] = np.log(1)

    permutation = [1, 2, 3, 4, 0, 5]
    index = np.empty_like(permutation)
    index[permutation] = np.arange(len(permutation))
    emission = emissionTable[:, index]

    for j in range(N - 2, -1, -1):
        logvCol = V[:, j + 1].reshape(-1, 1)
        x = emission[:, emissionDict[seq[j + 1]]].reshape((-1, 1))
        V[:, j] = logsumexp(logvCol + transitionTable.T + x, axis=0)

    return V


def classifyBM(k, motif_len):
    if k < 2 or k > (motif_len - 4) + 1:
        out = "B"
    else:
        out = "M"
    return out


def generateViterbiSeq(T, motif_len, N):
    """
    Draws the best alignment of sequence and motif.
    """
    seq = ""
    k = motif_len - 1

    for i in range(N-1, 0, -1):
        seq += classifyBM(T[k][i], motif_len)
        k = int(T[k][i])

    return seq[len(seq)::-1][1:]



def viterbiAlg(transitionTable, emissionTable, seq, motif_len):
    """
    Will calculate the best possible option to the sequence and it's motif position.
    Will generate a possibilities matrix and use an helper function to draw the best one.
    :return: the best alignment of sequence and motif.
    """
    # Generating new score and trace matrices
    N = len(seq)
    T = np.zeros(shape=(motif_len + 4, N))
    V = np.zeros(shape=(motif_len + 4, N))

    for k in range(motif_len + 4):
        V[k][0] = emissionTable[k][nucleotideDict[seq[0]]]

    for i in range(1, N):
        logvCol = np.reshape(V[:, i - 1], motif_len + 4)
        transition = np.transpose(transitionTable)
        emission = np.transpose(emissionTable)[nucleotideDict[seq[i]]]
        sum = np.add(logvCol, transition)

        T[:, i] = np.argmax(sum, axis=1)
        V[:, i] = np.max(sum, axis=1) + emission

    res = generateViterbiSeq(T, motif_len + 4, N)
    return res


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
