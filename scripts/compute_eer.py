#!/usr/bin/env python3
# Copyright 2018  David Snyder
# Apache 2.0

# This script computes the minimum detection cost function, which is a common
# error metric used in speaker recognition.  Compared to equal error-rate,
# which assigns equal weight to false negatives and false positives, this
# error-rate is usually used to assess performance in settings where achieving
# a low false positive rate is more important than achieving a low false
# negative rate.  See the NIST 2016 Speaker Recognition Evaluation Plan at
# https://www.nist.gov/sites/default/files/documents/2016/10/07/sre16_eval_plan_v1.3.pdf
# for more details about the metric.
from __future__ import print_function
from operator import itemgetter
import sys, argparse, os
import numpy as np

def GetArgs():
    parser = argparse.ArgumentParser(description="Compute equal error rate "
        "Usage: scripts/compute_eer.py <scores-file> <trials-file> "
        "E.g., scripts/compute_eer.py exp/scores/trials data/test/trials",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("scores_filename",
        help="Input scores file, with columns of the form "
        "<utt1> <utt2> <score>")
    parser.add_argument("trials_filename",
        help="Input trials file, with columns of the form "
        "<utt1> <utt2> <target/nontarget>")
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    return args

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

def main():
    args = GetArgs()
    scores_file = open(args.scores_filename, 'r').readlines()
    trials_file = open(args.trials_filename, 'r').readlines()

    scores = []
    labels = []

    trials = {}
    for line in trials_file:
        utt1, utt2, target = line.rstrip().split()
        trial = utt1 + " " + utt2
        trials[trial] = target

    for line in scores_file:
        utt1, utt2, score = line.rstrip().split()
        trial = utt1 + " " + utt2
        if trial in trials:
            scores.append(float(score))
            if trials[trial] == "target":
                labels.append(1)
            else:
                labels.append(0)
        else:
            raise Exception("Missing entry for " + utt1 + " and " + utt2
                + " " + args.scores_filename)

    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)

    idxE = np.nanargmin(np.absolute((np.array(fnrs) - np.array(fprs))))
    eer  = max(fprs[idxE],fnrs[idxE])

    sys.stdout.write("{0:.2%}\n".format(eer))
    sys.stderr.write("eer is {0:.2%}\n".format(eer))

if __name__ == "__main__":
  main()
