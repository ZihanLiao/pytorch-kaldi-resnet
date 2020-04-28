import os
import argparse
import pickle
import numpy as np

import kaldi_io

def main():
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--enroll", type=str, help="enroll topk mean and std file")
    parser.add_argument("--test", type=str, help="test topk mean and std file")
    parser.add_argument("--score-in", type=str, help="score in file")
    parser.add_argument("--score-out", type=str, help="score out file")
    args = parser.parse_args()

    spkr2mean, spkr2std = {}, {}
    for line in open(args.enroll, 'r'):
        spkr, mean, std = line.strip().split()
        spkr2mean[spkr] = float(mean)
        spkr2std[spkr] = float(std)

    utt2mean, utt2std = {}, {}
    for line in open(args.test, 'r'):
        utt, mean, std = line.strip().split()
        utt2mean[utt] = float(mean)
        utt2std[utt] = float(std)

    scores_to_file = []
    f = open(args.score_in, 'r')
    for line in f:
        spkr, utt, score = line.strip().split()
        score = float(score)
        adapt_snorm = (score - spkr2mean[spkr])/max(spkr2std[spkr], 1e-8)/2 \
                      + (score - utt2mean[utt])/max(utt2std[utt], 1e-8)/2
        scores_to_file.append('{} {} {}'.format(spkr, utt, adapt_snorm))
    f.close()

    np.savetxt(args.score_out, scores_to_file, '%s')
    print("saved adaptive S-norm scores in {}".format(args.score_out))

    #std, mean = compute_std_mean(args.test, uttlist)
    #print("std: {}, mean: {}".format(std, mean))


if __name__ == '__main__':
    main()
