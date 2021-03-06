import os
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

def compute_std_mean(ark_file, mean, uttlist):
    mat = []
    for key, vec in kaldi_io.read_vec_flt_ark(ark_file):
        if key not in uttlist:
            continue
        vec = vec - mean
        mat.extend(vec)
    mat = torch.FloatTensor(mat)
    mat = mat.view(-1, len(vec))
    print("speakers: {}, feat-dim: {}".format(mat.shape[0], mat.shape[1]))

    std, mean = torch.std_mean(mat, dim=0).data.numpy()
    return std, mean

def main():
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--mean", type=str, help="mean vec file")
    parser.add_argument("--enroll", type=str, help="enroll embeddings file")
    parser.add_argument("--test", type=str, help="test embeddings file")
    parser.add_argument("--trials", type=str, help="trials file")
    parser.add_argument("--score-file", type=str, help="score file")
    args = parser.parse_args()
    if args.mean and os.path.exists(args.mean):
        mean = kaldi_io.read_vec_flt(args.mean)
        print("loaded mean from {}".format(args.mean))
    else:
        print("mean file missing")
        return

    spkr2vec, utt2vec = {}, {}
    for utt, vec in kaldi_io.read_vec_flt_ark(args.enroll):
        spkr2vec[utt] = vec - mean
    for utt, vec in kaldi_io.read_vec_flt_ark(args.test):
        utt2vec[utt] = vec - mean

    scores_to_file = []
    f = open(args.trials, 'r')
    for line in f:
        spkr, utt, target = line.strip().split()
        spkr_vec = torch.FloatTensor(spkr2vec[spkr])
        utt_vec = torch.FloatTensor(utt2vec[utt])
        cos = F.cosine_similarity(spkr_vec, utt_vec, dim=0).data.numpy() 
        scores_to_file.append('{} {} {}'.format(spkr, utt, cos))
    f.close()

    np.savetxt(args.score_file, scores_to_file, '%s')
    print("saved scores of {} in {}".format(args.trials, args.score_file))

    #std, mean = compute_std_mean(args.test, uttlist)
    #print("std: {}, mean: {}".format(std, mean))


if __name__ == '__main__':
    main()
