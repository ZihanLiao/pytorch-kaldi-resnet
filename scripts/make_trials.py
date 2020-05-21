import sys
import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--strategy", type=str, help="strategy to make trials: rude, balance or part")
    parser.add_argument("--enroll", type=str, help="enroll dir")
    parser.add_argument("--test", type=str, help="test dir")
    parser.add_argument("--music", type=str, help="music dir")
    parser.add_argument("--trials", type=str, help="trials out file")
    args = parser.parse_args()

    enroll_spkr = []
    for line in open(os.path.join(args.enroll,'spk2utt'), 'r'):
        spkr = line.strip().split()[0]
        enroll_spkr.append(spkr)

    test_utt2spkr = {}
    for line in open(os.path.join(args.test,'utt2spk'), 'r'):
        utt, spkr = line.strip().split()
        test_utt2spkr[utt] = spkr

    trials = []
    if args.strategy == "rude":
        for utt in test_utt2spkr:
            for spkr in enroll_spkr:
                if test_utt2spkr[utt] == spkr:
                    trials.append('{} {} target'.format(spkr, utt))
                else:
                    trials.append('{} {} nontarget'.format(spkr, utt))
        
    if args.strategy == "balance":
        for utt in test_utt2spkr:
            spkr = test_utt2spkr[utt]
            if spkr in enroll_spkr:
                trials.append('{} {} target'.format(spkr, utt))
                #utts = [test_utt2spkr[utt] for utt in test_utt2spkr.keys() ]
                utt = np.random.choice([key for key in test_utt2spkr])
                if test_utt2spkr[utt] == spkr:
                    istarget = "target"
                else:
                    istarget = "nontarget"
                trials.append('{} {} {}'.format(spkr, utt, istarget))
        
    if args.strategy == "part":
        music = []
        for line in open(os.path.join(args.music, 'utt2spk'), 'r'):
            utt, spkr = line.strip().split()
            music.append(utt)
        for utt in test_utt2spkr:
            spkr = test_utt2spkr[utt]
            if spkr in enroll_spkr:
                trials.append('{} {} target'.format(spkr, utt))
                for n in range(len(enroll_spkr)):
                    trials.append('{} {} nontarget'.format(spkr, np.random.choice(music)))

    np.savetxt(args.trials, trials, '%s')
    print("saved {} trials in {}".format(len(trials), args.trials))


if __name__ == '__main__':
    main()
