import os
import sys
import argparse
import numpy as np

import torch
import kaldi_io

def compute_mean(spk2utt_file, ark_file):
    utt2vec = {}
    mean = {}
    for key, vec in kaldi_io.read_vec_flt_ark(ark_file):
        utt2vec[key] = vec
    for line in open(spk2utt_file, 'r'):
        arr = line.strip().split()
        spk = arr[0]
        utts = arr[1:]
        mat = []
        for key in utts:
            mat.extend(utt2vec[key])
        #print("{} shape: {},".format(key, len(vec)))
        #print("mat shape: {},".format(len(mat)))
        mat = torch.FloatTensor(mat)
        mat = mat.view(-1, len(utt2vec[key]))
        mean[spk] = torch.mean(mat, dim=0).data.numpy()

    print("speakers: {}, feat-dim: {}".format(len(mean), len(mean[spk])))

    return mean

def main():
    spk2utt_file = sys.argv[1]
    ark_file = sys.argv[2]
    mean_file = sys.argv[3]

    mean = compute_mean(spk2utt_file, ark_file)
    f = open(mean_file, 'w')
    for spk in mean:
        f.write(spk+' [ '+' '.join(map(str, mean[spk]))+' ]\n')
    f.close()
    print("saved mean of {} by {} in {}".format(ark_file, spk2utt_file, mean_file))

if __name__ == '__main__':
    main()

