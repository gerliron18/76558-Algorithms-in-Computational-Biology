#!/usr/bin/python -u

import os
import argparse
import subprocess
import numpy as np
import pandas as pd

"""
This is a sanity-test script for your motif_find.py.
No need to read, edit or submit it this file!

Better make sure you pass all 4 tests before submitting.
Usage:
    python3 sanity_test.py PATH_TO_MOTIF_FIND
"""

seq = 'CCAAAATT'
states = 'BBMMMMBB'
ll = -10.47


def is_output_correct(ret, alg):
    """ validate user output against school solution """

    r = [l.strip() for l in ret.split('\n') if l.strip()]
    if alg in ('forward', 'backward'):
        return len(r) == 1 and np.round(float(r[0]), 2) == ll
    else:
        return len(r) == 2 and r[0] == states and r[1] == seq


def test_single_alg(mf, epath, alg):
    """ run motif_find.py on a single algorithm
        (forward/ backward/ posterior/ viterbi) """
    print(f'testing {alg}...')

    # run test as subprocess
    try:
        cmd = f'python {mf} --alg {alg} {seq} {epath} .1 .99'
        ret = subprocess.check_output(cmd, shell=True).decode()
    except Exception as e:
        print('Failed to run motif_find.py as a subprocess! ', e)
        return False

    # validate return value and print SUCCESS/FAIL
    if is_output_correct(ret, alg):
        print('\033[32m{}\033[00m'.format('SUCCESS'))
    else:
        print('\033[31m{}\033[00m'.format('FAIL'))


def main(args):

    # make sure input motif_find.py exists
    mf = args.motif_find_path
    if not os.path.isfile(mf):
        print(f'Invalid file: {mf}')
        return 1

    # generate and dump a trivial emission table
    epath = './emissions_AAAA.tsv'
    pd.DataFrame(data=np.array([[.97, .01, .01, .01]] * 4),
                 columns=list('ACGT')).to_csv(epath, sep='\t',
                                              index=None)

    # test all 4 algorithms
    for alg in ('forward', 'backward', 'posterior', 'viterbi'):
        test_single_alg(mf, epath, alg)

    # cleanup
    os.remove(epath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('motif_find_path', help='Path to your motif_find.py script (e.g. ./motif_find.py)')
    main(parser.parse_args())
