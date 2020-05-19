#!/bin/bash

# Author: Hongmei Liu
# step 12: scoring
# step 13: EER & minDCF results

. ./cmd.sh
#. ./path.sh
set -e

#voxceleb1_trials=data/test/trials_o
voxceleb1_trials=data/test/trials_e
#voxceleb1_trials=data/test/trials_h

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
        "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" $dir/$score_file || exit 1;
  else
    echo "cosine scoring..."
    $train_cmd $dir/log/test_scoring.log \
        python scripts/cosine_score.py \
        --mean $modeldir/mean.vec \
        --enroll $modeldir/test.iv \
        --test $modeldir/test.iv \
        --trials $voxceleb1_trials \
        --score-file $dir/$score_file
:<<!
        ivector-compute-dot-products \
        "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" \
        "ark:ivector-subtract-global-mean $modeldir/mean.vec ark:$modeldir/test.iv ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean $modeldir/mean.vec ark:$modeldir/test.iv ark:- | ivector-normalize-length ark:- ark:- |" \
        $dir/$score_file || exit 1;
!
    
  fi
  if [ $backend == 'snorm' ];then
    echo "adptive S-norm..."
    $train_cmd $dir/log/adaptive_snorm.log \
        python scripts/adaptive_snorm.py \
        --enroll $dir/topk_mean_std \
        --test $dir/topk_mean_std \
        --score-in $dir/$score_file \
        --score-out $dir/${score_file}_adapt_snorm
  fi
fi

if [ $backend == 'snorm' ];then
  score_file=${score_file}_adapt_snorm
  eer_file=${eer_file}_adapt_snorm
fi

if [ $stage -le 13 ]; then
    #eer=`compute-eer <(python local/prepare_for_eer.py $voxceleb1_trials $dir/$score_file) 2> /dev/null`
    eer=`python scripts/compute_eer.py $dir/$score_file $voxceleb1_trials 2> /dev/null`
    mindcf1=`python local/compute_min_dcf.py --p-target 0.01 $dir/$score_file $voxceleb1_trials 2> /dev/null`
    mindcf2=`python local/compute_min_dcf.py --p-target 0.001 $dir/$score_file $voxceleb1_trials 2> /dev/null`
    echo "EER: $eer%" > $dir/$eer_file
    echo "minDCF(p-target=0.01): $mindcf1" >> $dir/$eer_file
    echo "minDCF(p-target=0.001): $mindcf2" >> $dir/$eer_file
    echo "backend: $backend"
    cat $dir/$eer_file
fi
