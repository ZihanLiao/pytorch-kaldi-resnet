import os
import sys
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import kaldi_io

def compute_topk_mean_std(utt2vec, cohort_mat, topk=300):
    print('cohort_mat.shape: {}, utt2vec: {}'.format(cohort_mat.shape, len(utt2vec)))
    norm_mat = F.normalize(cohort_mat, p=2, dim=1)
    std, mean = {}, {}
    for key in utt2vec:
        vec = torch.FloatTensor(utt2vec[key])
        vec = F.normalize(vec, p=2, dim=0)
        scores = torch.matmul(norm_mat, vec)
        scores_topk, indics = scores.topk(topk)
        s, m = torch.std_mean(scores_topk)
        std[key] = s.data.numpy()
        mean[key] = m.data.numpy()
     
    return mean, std

def main():
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--mean", type=str, help="mean vec file")
    parser.add_argument("--ark-file", type=str, help="test embeddings file")
    parser.add_argument("--cohort-file", type=str, help="cohort embeddings file")
    parser.add_argument("--mean-std-file", type=str, help="file to save mean and std")
    args = parser.parse_args()

    if args.mean and os.path.exists(args.mean):
        mean = kaldi_io.read_vec_flt(args.mean)
        print("loaded mean from {}".format(args.mean))
    else:
        print("mean file missing")
        return

    utt2vec = {}
    for utt, vec in kaldi_io.read_vec_flt_ark(args.ark_file):
        utt2vec[utt] = torch.FloatTensor(vec - mean)

    cohort_mat = []
    for spk, vec in kaldi_io.read_vec_flt_ark(args.cohort_file):
        cohort_mat.extend(vec - mean)
    cohort_mat = torch.FloatTensor(cohort_mat)
    cohort_mat = cohort_mat.view(-1, len(vec)) 
    
    #print('cohort_mat.shape: {}, utt2vec: {}, vec: {}'.format(cohort_mat.shape, len(utt2vec), len(vec)))
    
    mean, std = compute_topk_mean_std(utt2vec, cohort_mat)

    f = open(args.mean_std_file, 'w')
    for spk in mean:
        f.write('{} {} {}\n'.format(spk, mean[spk], std[spk]))
    f.close()
    print("saved speaker mean in {}".format(args.mean_std_file))

if __name__ == '__main__':
    main()

