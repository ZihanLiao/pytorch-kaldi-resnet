import os
import sys
import argparse
import numpy as np

import torch
import kaldi_io

def compute_speaker_mean(ark_file, utt2spk_file):
    utt2spk = {}
    for line in open(utt2spk_file, 'r'):
        utt, spk = line.strip().split()
        utt2spk[utt] = spk
        
    spk2num = {}
    speaker_mean = {}
    for utt, vec in kaldi_io.read_vec_flt_ark(ark_file):
        if utt not in utt2spk:
            raise Exception('{} not specified to any speaker'.format(utt))
        spk = utt2spk[utt]
        if spk not in speaker_mean:
            speaker_mean[spk] = np.zeros([len(vec)], dtype=np.float32)
            spk2num[spk] = 0
        speaker_mean[spk] += vec
        spk2num[spk] += 1
    for spk in speaker_mean:
        speaker_mean[spk] /= spk2num[spk]
        
    print("speakers: {}, feat-dim: {}".format(len(speaker_mean), len(vec)))
    return speaker_mean

def main():
    ark_file = sys.argv[1]
    utt2spk_file = sys.argv[2]
    mean_file = sys.argv[3]

    speaker_mean = compute_speaker_mean(ark_file, utt2spk_file)
    f = open(mean_file, 'w')
    for spk in speaker_mean:
        f.write(spk+' [ '+' '.join(map(str, speaker_mean[spk]))+' ]\n')
    f.close()
    print("saved speaker mean in {}".format(mean_file))

if __name__ == '__main__':
    main()

