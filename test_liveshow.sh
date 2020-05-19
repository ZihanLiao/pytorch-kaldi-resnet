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

#. ./path.sh
#set -e

stage=$1

mdl_dir=exp/resnet34_aamsoftmax_epoch30_dataset2_wd0.0005_fine_tuned
model=$mdl_dir/model_best.pth.tar
num_enroll=5
num_test=5

#datadir=data/live_sing
#dir=results/live_sing
src_datadir=/data_lfrz612/liu.hongmei/upload-liveshow/live-sing-20200514/dataset/v1
datadir=data/live_sing_test/v1
dir=results/live_sing_3s_v1

mkdir -p $datadir $dir

host=`hostname`
if [ $host == "bigdata-ci-gpu-mozart-101.rw.momo.com" ];then
  . ./cmd.sh
else
  . ./cmd.sh
fi

if [ $stage -le 0 ]; then
  echo "prepare enroll and test"
  sh make_enroll_test.sh --num-enroll $num_enroll --num-test $num_test \
    --part-list $src_datadir/music.list \
    $src_datadir $datadir  3 
fi

#exit
if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train enroll test music; do
    local/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank.conf \
      --nj $train_nj --cmd "$train_cmd" $datadir/${name}
    local/fix_data_dir.sh $datadir/${name}
    #local/compute_vad_decision.sh --nj $train_nj --cmd "$train_cmd" \
    #local/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
    #  $datadir/${name}
    #local/fix_data_dir.sh $datadir/${name}
  done

  # Make MFCCs and compute the energy-based VAD for each dataset
  # NOTE: Kaldi VAD is based on MFCCs, so we need to additionally extract MFCCs
  # (https://groups.google.com/forum/#!msg/kaldi-help/-REizujqa5k/u_FJnGokBQAJ)
  for name in train enroll test music; do
    local/copy_data_dir.sh $datadir/${name} $datadir/${name}_mfcc
    local/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf \
      --nj $train_nj --cmd "$train_cmd" $datadir/${name}_mfcc
    local/fix_data_dir.sh $datadir/${name}_mfcc
    #local/compute_vad_decision.sh --nj $train_nj --cmd "$train_cmd" \
    local/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
      $datadir/${name}_mfcc
    local/fix_data_dir.sh $datadir/${name}_mfcc
  done

  # correct the right vad.scp
  for name in train enroll test music; do
    cp $datadir/${name}_mfcc/vad.scp $datadir/${name}/vad.scp
    local/fix_data_dir.sh $datadir/$name
  done
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  for name in train enroll test music; do
    #local/nnet3/xvector/prepare_feats_for_egs.sh --nj $train_nj --cmd "$train_cmd" --compress false \
    local/nnet3/xvector/prepare_feats_for_egs.sh --nj 1 --cmd "$train_cmd" --compress false \
      $datadir/$name $datadir/${name}_no_sil $datadir/${name}_no_sil/data
    local/fix_data_dir.sh $datadir/${name}_no_sil
  done
fi

# !!!note that we also need to apply the same pre-processing to decode data!!!
if [ $stage -le 6 ]; then
  mkdir $datadir/processed
  utils/filter_scp.pl $datadir/utt2spk $datadir/train_no_sil/feats.scp | sed 's/data_lfrz613/data/' \
    > $datadir/processed/decode_train.scp 
  utils/filter_scp.pl $datadir/test/utt2spk $datadir/test_no_sil/feats.scp | sed 's/data_lfrz613/data/' \
    > $datadir/processed/decode_test.scp 
  utils/filter_scp.pl $datadir/enroll/utt2spk $datadir/enroll_no_sil/feats.scp | sed 's/data_lfrz613/data/' \
    > $datadir/processed/decode_enroll.scp 
  utils/filter_scp.pl $datadir/music/utt2spk $datadir/music_no_sil/feats.scp | sed 's/data_lfrz613/data/' \
    > $datadir/processed/decode_music.scp 
echo "switch 613 to continue testing"
exit
fi

[[ ! $host == "lfrz-platform-bigdata-ci-gpu-abaci-613.dev.lfrz.momo.com" ]] && exit;

# Network Decoding; do this for all your data
if [ $stage -le 9 ]; then
  echo "produce embeddings"
  [[ ! -f $model ]] && echo "$model not exists" && exit
  num_spk=`cat exp/processed_cv0.03/num_spk`
  echo "There are "$num_spk" number of speakers."
  for x in train enroll test music;do
  $cuda_cmd $dir/log/decode_${x}.log \
      python scripts/decode.py \
      --multiprocessing-distributed \
      --dist-url "tcp://127.0.0.1:29841" \
      --world-size 1 --rank 0 \
      --gpu-num 8 --workers 16 \
      --batch-size 8 --chunk-size -1 \
      --spk_num $num_spk --arch 'resnet34' \
      --input-dim 40 --pooling 'mean+std' \
      --model-path $model --decode-scp $datadir/processed/decode_${x}.scp \
      --out-path $dir/embeddings_$x
  done
fi
#exit 0

mkdir -p $dir/backend
if [ $stage -le 10 ]; then
    echo "get train and test embeddings"
    # remove duplications
    cat $dir/embeddings_train/* |  awk '{if(a[$1]!=1){print $0;a[$1]=1}}' > $dir/backend/train.iv
    cat $dir/embeddings_enroll/* |  awk '{if(a[$1]!=1){print $0;a[$1]=1}}' > $dir/backend/enroll.iv
    cat $dir/embeddings_test/*  |  awk '{if(a[$1]!=1){print $0;a[$1]=1}}' >  $dir/backend/test.iv
    cat $dir/embeddings_music/* |  awk '{if(a[$1]!=1){print $0;a[$1]=1}}' >> $dir/backend/test.iv
fi


if [ $stage -le 11 ]; then
    #cat $mdl_dir/backend/mean.vec > $dir/backend/mean.vec

    echo "compute mean vector..."
    # Compute the mean vector for centering the evaluation ivectors.
    $train_cmd $dir/log/compute_mean.log \
        python scripts/compute_mean.py $dir/backend/train.iv \
        $dir/backend/mean.vec
:<<!

    # This script uses LDA to decrease the dimensionality prior to PLDA.
    lda_dim=200
    $train_cmd $dir/log/lda.log \
        ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
        "ark:ivector-subtract-global-mean ark:$dir/backend/train.iv ark:- |" \
        ark:data/train/utt2spk $dir/backend/transform.mat || exit 1;

    # Train the PLDA model.
    $train_cmd $dir/log/plda.log \
        ivector-compute-plda ark:data/train/spk2utt \
        "ark:ivector-subtract-global-mean ark:$dir/backend/train.iv ark:- | transform-vec $dir/backend/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
        $dir/backend/plda || exit 1;
!
fi

if [ $stage -le 12 ]; then
    echo "test ..."
    for x in rude balance part;do
      trials=$datadir/test/trials.$x
      echo "use trials $trials"
 #     if [ ! -f $trials ];then
        echo "make trials $trials"
        python scripts/make_trials.py \
            --strategy "$x" \
            --enroll $datadir/enroll \
	    --test $datadir/test \
            --music $datadir/music \
            --trials $trials
  #    fi
        ./test2.sh --datadir $datadir --trials $trials $dir/backend $dir/backend "pool" $stage
    done

    #./test.sh --datadir $datadir --trials $datadir/test/trials.part $dir/backend $dir/backend "pool" $stage
    #./test.sh --datadir $datadir --trials $datadir/test/trials.balance $dir/backend $dir/backend "pool" $stage
    #./test.sh --datadir $datadir --trials $datadir/test/trials.spkid $dir/backend $dir/backend "pool" $stage
    #./test.sh --datadir $datadir $dir/backend $dir/backend "pool" $stage
    #./test.sh $dir/backend $dir/backend "cosine" $stage
    #./test.sh $dir/backend $dir/backend "plda" $stage
fi


