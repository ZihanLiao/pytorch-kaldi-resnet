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
#. ./path.sh
set -e


datadir=data
trials=$datadir/test/trials.spkid

. utils/parse_options.sh

modeldir=$1
dir=$2
backend=$3
stage=$4
score_file=scores_$backend
eer_file=eer_$backend


if [ $stage -le 12 ]; then
  if [ $backend == "plda" ];then
    echo "plda scoring..."
    $train_cmd $dir/log/test_scoring.log \
        ivector-plda-scoring --normalize-length=true \
        "ivector-copy-plda --smoothing=0.0 $modeldir/plda - |" \
        "ark:ivector-subtract-global-mean $modeldir/mean.vec ark:$modeldir/test.iv ark:- | transform-vec $modeldir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $modeldir/mean.vec ark:$modeldir/test.iv ark:- | transform-vec $modeldir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat '$trials' | cut -d\  --fields=1,2 |" $dir/$score_file || exit 1;
  elif [ $backend == 'cosine' ];then
    echo "cosine scoring..."
    $train_cmd $dir/log/test_scoring.log \
        python scripts/cosine_score.py \
        --mean $modeldir/mean.vec \
        --enroll $modeldir/test.iv \
        --test $modeldir/test.iv \
        --trials $trials \
        --score-file $dir/$score_file

    $train_cmd $dir/log/adaptive_snorm.log \
        python scripts/adaptive_snorm.py \
        --enroll $dir/topk_mean_std \
        --test $dir/topk_mean_std \
        --score-in $dir/$score_file \
        --score-out $dir/${score_file}_adapt_snorm
    score_file=${score_file}_adapt_snorm
    eer_file=${eer_file}_adapt_snorm
:<<!
        ivector-compute-dot-products \
        "cat '$trials' | cut -d\  --fields=1,2 |" \
        "ark:ivector-subtract-global-mean $modeldir/mean.vec ark:$modeldir/test.iv ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $modeldir/mean.vec ark:$modeldir/test.iv ark:- | ivector-normalize-length ark:- ark:- |" \
        $dir/$score_file || exit 1;
!
  elif [ $backend == "pool" ];then
    # Get results using the trainset PLDA model.
    $train_cmd $dir/log/compute_mean_byspk.log \
        python scripts/compute_mean_byspk.py \
        $datadir/enroll/spk2utt $modeldir/enroll.iv $modeldir/spk.iv
        
    $train_cmd $dir/log/test_scoring.log \
        python scripts/cosine_score.py \
        --mean $modeldir/mean.vec \
        --enroll $modeldir/spk.iv \
        --test $modeldir/test.iv \
        --trials $trials \
        --score-file $dir/$score_file
:<<!
    $train_cmd $dir/log/test_scoring.log \
        ivector-compute-dot-products \
        "cat '$trials' | cut -d\  --fields=1,2 |" \
        "ark:ivector-mean ark:$datadir/enroll/spk2utt ark:$modeldir/enroll.iv ark:- | ivector-subtract-global-mean $modeldir/mean.vec ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $modeldir/mean.vec ark:$modeldir/test.iv ark:- | ivector-normalize-length ark:- ark:- |" \
        $dir/$score_file || exit 1;
!
    
  fi
fi


if [ $stage -le 13 ]; then
    #eer=`compute-eer <(python local/prepare_for_eer.py $trials $dir/$score_file) 2> /dev/null`
    eer=`python scripts/compute_eer.py $dir/$score_file $trials 2> /dev/null`
    mindcf1=`python local/compute_min_dcf.py --p-target 0.01 $dir/$score_file $trials 2> /dev/null`
    mindcf2=`python local/compute_min_dcf.py --p-target 0.001 $dir/$score_file $trials 2> /dev/null`
    echo "EER: $eer" > $dir/$eer_file
    echo "minDCF(p-target=0.01): $mindcf1" >> $dir/$eer_file
    echo "minDCF(p-target=0.001): $mindcf2" >> $dir/$eer_file
    echo "backend: $backend"
    cat $dir/$eer_file
fi
