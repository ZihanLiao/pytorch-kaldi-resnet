import os
import sys
import argparse
import pickle
import numpy as np

import torch
import torch.nn.functional as F
import kaldi_io

def compute_mean(ark_file):
    mat = []
    for key, vec in kaldi_io.read_vec_flt_ark(ark_file):
        mat.extend(vec)
        #print("{} shape: {},".format(key, len(vec)))
        #print("mat shape: {},".format(len(mat)))
    mat = torch.FloatTensor(mat)
    mat = mat.view(-1, len(vec))
    print("speakers: {}, feat-dim: {}".format(mat.shape[0], mat.shape[1]))

    mean = torch.mean(mat, dim=0).data.numpy()
    return mean

def main():
    ark_file = sys.argv[0]
    mean_file = sys.argv[1]

    mean = compute_mean(ark_file)
    f = open(mean_file, 'w')
    f.write(' [ '+' '.join(map(str, mean))+' ]\n')
    f.close()
    print("saved mean of {} in {}".format(ark_file, mean_file)

if __name__ == '__main__':
    main()
