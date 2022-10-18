import numpy as  np
import matplotlib.pyplot as pt
from scipy.linalg import expm

nucleotideArr = ['A', 'C', 'G', 'T']
N = [10, 100, 1000, 10000]
TIMES = [0.04, 0.1, 0.3]


def frequencyComp(samplesArr, repeats):
    distanceArr = []

    for i in range(len(TIMES)):
        iDistanceArr = []

        for j in range(len(nucleotideArr)):
            frequency = samplesArr[i].count(nucleotideArr[j]) / repeats

            if nucleotideArr[j] == 'A':
                transitionProb = 0.25 * (1 + (3 * np.exp(-(TIMES[i]))))
            else:
                transitionProb = 0.25 * (1 - (3 * np.exp(-(TIMES[i]))))

            iDistanceArr.append(transitionProb - frequency)

        distanceArr.append(iDistanceArr)

    plotGraph(distanceArr, repeats)


def plotGraph(distanceArr, repeats):
    pt.title(
        "The actual and predicted frequency of b comparison for N = " + str(
            repeats))
    pt.xlim(-1, 5)
    pt.ylim(-1, 0.6)
    pt.xticks([0, 1, 2, 3], nucleotideArr)
    pt.plot([0, 1, 2, 3], distanceArr[0], 'ro', color='r', label="time = 0.04")
    pt.plot([0, 1, 2, 3], distanceArr[1], 'ro', color='g', label="time = 0.1")
    pt.plot([0, 1, 2, 3], distanceArr[2], 'ro', color='b', label="time = 0.3")
    pt.legend()
    pt.show()


def question1_2():
    for repeats in N:
        samplesArr = []

        for time in TIMES:
            sampleLetterArr = []

            for iteration in range(repeats):
                transitionProb = 0.25 * (1 + (3 * np.exp(-time)))
                
                if np.random.uniform(0, 1) > transitionProb:
                    sampleLetterArr.append(np.random.choice(['C', 'G', 'T']))
                else:
                    sampleLetterArr.append('A')
            samplesArr.append(sampleLetterArr)
        frequencyComp(samplesArr, repeats)


def plotBoxplot(mleArr, M):
    pt.title("MLE estimation for M = " + str(M))
    pt.boxplot(mleArr)
    pt.xticks([1, 2, 3], TIMES)
    pt.xlabel("time")
    pt.show()


def generateSeqs(length, time):
    firstSeq, secondSeq = "", ""

    for n in range(length):
        a = np.random.choice(nucleotideArr)
        transitionProb = 0.25 * (1 + (3 * np.exp(-time)))

        if np.random.uniform(0, 1) > transitionProb:
            b = np.random.choice(['C', 'G', 'T'])
        else:
            b = a

        firstSeq += a
        secondSeq += b

    return firstSeq, secondSeq


def MLEcalculator(equalityCount, differenceCount):
    MLE = -(np.log((3 * equalityCount - differenceCount) / (3 * equalityCount + 3 * differenceCount)))

    return MLE


def question1_3():
    mleArr = []
    length = 500
    M = 100

    for time in TIMES:
        mleSeqArr = []

        for m in range(M):
            equalityCount = 0
            differenceCount = 0
            firstSeq, secondSeq = generateSeqs(length, time)

            for n in range(length):
                if firstSeq[n] == secondSeq[n]:
                    equalityCount += 1
                else:
                    differenceCount += 1

            mleSeqArr.append(MLEcalculator(equalityCount, differenceCount))

        mleArr.append(mleSeqArr)

    plotBoxplot(mleArr, M)


def question2():
    matrix1 = np.matrix([[-5, 1, 3, 1],
                         [3, -5, 1, 1],
                         [1, 1, -5, 3],
                         [1, 3, 1, -5]])
    result1 = expm(matrix1 * 100000)
    print(result1)


    matrix2 = np.matrix([[1, 0, 0, -1],
                         [0, 1, -1, 0],
                         [0, -1, 1, 0],
                         [-1, 0, 0, 1]])
    result2 = expm(matrix2)
    print(result2)


def main():
    question1_2()
    question1_3()
    question2()


if __name__ == "__main__":
    main()
