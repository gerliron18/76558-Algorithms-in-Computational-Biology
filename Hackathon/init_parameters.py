
import numpy as np
import matplotlib.pyplot as plt


def eliminate_zeros(arr):
    return arr[arr[:, 1] != 0]


def eliminate_zeros_add_one(arr):
    arr[:, 0] += 1
    arr[:, 1] += 2
    return arr


def array_to_prob(arr):
    return arr[:, 0] / arr[:, 1]


def smooth(arr):
    ret = np.zeros(arr.shape[0] + 7)
    for i in range(7):
        ret[i:-(7-i)] += arr
    return (ret / 7)[3:-4]


HIGH = 'High'
LOW = 'Low'


def get_num_low(arr):
    for i in range(0, arr.shape[0], 200):
        x = np.arange(1, 201)
        plt.plot(x, arr[200*i: 200*(i+1)], 'b-')
        plt.ylim(0, 1)
        plt.show()
        input('cont?')
    low_threshold = 0.3
    high_threshold = 0.7
    state = 'Low'
    transitions = 0
    sum_high, sum_low = 0, 0
    for i in range(arr.shape[0]):
        if arr[i] < low_threshold and state == HIGH:
            transitions += 1
            state = LOW
        if arr[i] > high_threshold:
            state = HIGH
        if state == HIGH:
            sum_high += 1
        if state == LOW:
            sum_low += 1

    print('num of transitions: ' + str(transitions))
    print('prob for transition from low to high: ' + str(1 / (sum_low / transitions)))
    print('prob for transition from high to low: ' + str(1 / (sum_high / transitions)))


filename = 'Prostate-Epithelial-Z000000S3.beta'
arr = np.fromfile(filename, dtype=np.uint8).reshape((-1, 2))
arr = eliminate_zeros_add_one(arr)
arr = array_to_prob(arr)
arr = smooth(arr)
get_num_low(arr)


