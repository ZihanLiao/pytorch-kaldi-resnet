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

stage=$1

epoch=30

lr=0.1
lr_final=0
weight_decay=0.0005

margin=0.2
scale=30
#datadir=exp/processed
datadir=exp/processed_cv0.03

#expname=resnet34_aamsoftmax_epoch30_v2
#expname=resnet34_aamsoftmax_epoch${epoch}_wd${weight_decay}_fine_tuned
expname=resnet34_aamsoftmax_epoch${epoch}_wd${weight_decay}
dir=exp/$expname

num_spk=`cat exp/processed/num_spk`
echo "There are "$num_spk" number of speakers."

# feature prepare in kaldi
if [ $stage -le 6 ]; then
  #./feature_pre.sh $datadir $stage 0.1
  ./feature_pre.sh $datadir $stage 0.03
fi

num_spk=`cat $datadir/num_spk`
echo "There are "$num_spk" number of speakers."

#:<<!
# Network Training
if [ $stage -le 7 ]; then
mkdir -p $dir/pretrained
  echo "pretrain model..."
  $cuda_cmd $dir/log/pretrain.log \
      python scripts/train_resnet.py \
      --multiprocessing-distributed \
      --world-size 1 --rank 0 \
      --gpu-num 8 --workers 16 \
      --batch-size 1024 --print-freq 500 \
      --dist-url "tcp://127.0.0.1:27544" \
      --arch 'resnet34' --input-dim 40 \
      --loss-type "softmax" --pooling 'mean+std' \
      --dataset 'v1' --epochs $epoch \
      --lr $lr --lr-final $lr_final \
      --wd $weight_decay \
      --min-chunk-size 200 --max-chunk-size 200 \
      --train-list $datadir/train_orig.scp --cv-list $datadir/cv_orig.scp \
      --spk-num $num_spk --utt2spkid $datadir/utt2spkid \
      --log-dir $dir/pretrained
      #--resume $dir/model_best.pth.tar \
fi
#!

mkdir -p $dir
# pretrained model is softmax loss
#pretrained_model=exp/resnet34_softmax_epoch30/model_best.pth.tar
pretrained_model=$dir/pretrained/model_best.pth.tar
if [ $stage -le 8 ]; then
  echo "train model in $dir ..."
  $cuda_cmd $dir/log/train.log \
      python scripts/train_resnet.py \
      --multiprocessing-distributed \
      --world-size 1 --rank 0 \
      --gpu-num 8 --workers 16 \
      --batch-size 1024 --print-freq 500 \
      --dist-url "tcp://127.0.0.1:27544" \
      --arch 'resnet34' --input-dim 40 \
      --loss-type "AAM" --pooling 'mean+std' \
      --margin $margin --scale $scale \
      --dataset 'v1' --epochs $epoch \
      --lr $lr --lr-final $lr_final \
      --wd $weight_decay \
      --min-chunk-size 200 --max-chunk-size 200 \
      --train-list $datadir/train_orig.scp --cv-list $datadir/cv_orig.scp \
      --spk-num $num_spk --utt2spkid $datadir/utt2spkid \
      --pretrained $pretrained_model \
      --log-dir $dir
      #--resume $dir/model_best.pth.tar \
  #exit 0
fi

#exit

# Network Decoding; do this for all your data
if [ $stage -le 9 ]; then
  echo "produce embeddings"
  model=$dir/model_best.pth.tar # get best model
  [[ ! -f $model ]] && echo "$model not exists" && exit
  for x in train test;do
  $cuda_cmd $dir/log/decode_${x}.log \
      python scripts/decode.py \
      --multiprocessing-distributed \
      --dist-url "tcp://127.0.0.1:27544" \
      --world-size 1 --rank 0 \
      --gpu-num 8 --workers 16 \
      --batch-size 8 --chunk-size -1 \
      --spk_num $num_spk --arch 'resnet34' \
      --input-dim 40 --pooling 'mean+std' \
      --model-path $model --decode-scp $datadir/decode_${x}.scp \
      --out-path $dir/embeddings_$x
  done
fi
#exit 0

mkdir -p $dir/backend
if [ $stage -le 10 ]; then
    echo "get train and test embeddings"
    # remove duplications
    cat $dir/embeddings_train/* | awk '{if(a[$1]!=1){print $0;a[$1]=1}}' > $dir/backend/train.iv
    cat $dir/embeddings_test/* |  awk '{if(a[$1]!=1){print $0;a[$1]=1}}' > $dir/backend/test.iv
fi


if [ $stage -le 11 ]; then
    echo "compute mean vector..."
    # Compute the mean vector for centering the evaluation ivectors.
    $train_cmd $dir/log/compute_mean.log \
        python scripts/compute_mean.py $dir/backend/train.iv \
        $dir/backend/mean.vec

    $train_cmd $dir/log/compute_speaker_mean.log \
        python scripts/compute_speaker_mean.py $dir/backend/train.iv \
        $datadir/utt2spk $dir/backend/spk_mean.vec

    $train_cmd $dir/log/compute_topk_mean_std.py \
        python scripts/compute_topk_mean_std.py \
        --mean $dir/backend/mean.vec \
        --cohort-file $dir/backend/spk_mean.vec \
        --ark-file $dir/backend/test.iv \
        --mean-std-file $dir/backend/topk_mean_std

:<<!
		ivector-mean ark:$dir/backend/train.iv\
		$dir/backend/mean.vec || exit 1;

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

if [ $stage -le 13 ]; then
    echo "test ..."
    #./test.sh $dir/backend $dir/backend "plda" $stage
    ./test.sh $dir/backend $dir/backend "cosine" $stage
    ./test.sh $dir/backend $dir/backend "snorm" $stage
fi

