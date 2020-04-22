#!/bin/bash

# Author: Nanxin Chen, Cheng-I Lai
# Pipeline for preprocessing + training + postprocessing neural speaker embeddings. This includes:
# step 0:  create VoxCeleb1+2 data directories
# step 1:  make FBanks + VADs (based on MFCCs) for clean data
# step 2:  data augmentation
# step 3:  make FBanks for noisy data
# step 4:  applies CM and removes silence (for training data)
# step 5:  filter by length, split to train/cv, and (optional) save as pytorch tensors
# step 6:  nn training
# step 7:  applies CM and removes silence (for decoding data)
# step 8:  decode with the trained nn
# step 9:  get train and test embeddings
# step 10: compute mean, LDA and PLDA on decode embeddings
# step 11: scoring
# step 12: EER & minDCF results
# (This script is modified from Kaldi egs/)

. ./cmd.sh
. ./path.sh
set -e

#voxceleb1_trials=data/test/trials_o
voxceleb1_trials=data/test/trials_e
#voxceleb1_trials=data/test/trials_h

modeldir=$1
dir=$2
backend=$3
stage=$4


if [ $stage -le 12 ]; then
  if [ $backend == "plda" ];then
    echo "plda scoring..."
    $train_cmd $dir/log/test_scoring.log \
        ivector-plda-scoring --normalize-length=true \
        "ivector-copy-plda --smoothing=0.0 $modeldir/plda - |" \
        "ark:ivector-subtract-global-mean $modeldir/mean.vec ark:$modeldir/test.iv ark:- | transform-vec $modeldir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $modeldir/mean.vec ark:$modeldir/test.iv ark:- | transform-vec $modeldir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $dir/scores_$backend || exit 1;
  else
    echo "cosine scoring..."
    $train_cmd $dir/log/test_scoring.log \
        ivector-compute-dot-products \
        "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" \
        "ark:ivector-subtract-global-mean $modeldir/mean.vec ark:$modeldir/test.iv ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $modeldir/mean.vec ark:$modeldir/test.iv ark:- | ivector-normalize-length ark:- ark:- |" \
        $dir/scores_$backend || exit 1;
  fi
fi


if [ $stage -le 13 ]; then
    eer=`compute-eer <(python local/prepare_for_eer.py $voxceleb1_trials $dir/scores_$backend) 2> /dev/null`
    mindcf1=`python local/compute_min_dcf.py --p-target 0.01 $dir/scores_$backend $voxceleb1_trials 2> /dev/null`
    mindcf2=`python local/compute_min_dcf.py --p-target 0.001 $dir/scores_$backend $voxceleb1_trials 2> /dev/null`
    echo "EER: $eer%" > $dir/eer_$backend
    echo "minDCF(p-target=0.01): $mindcf1" >> $dir/eer_$backend
    echo "minDCF(p-target=0.001): $mindcf2" >> $dir/eer_$backend
    echo "backend: $backend"
    cat $dir/eer_$backend
fi
