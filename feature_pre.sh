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

# Change this to your Kaldi voxceleb egs directory
kaldi_voxceleb=/data_102/common_resource/kaldi/egs/voxceleb

# The trials file is downloaded by local/make_voxceleb1.pl.
trials_root=/data_lfrz612/train_data/voxceleb1/list_of_trial_pairs
voxceleb1_root=/data_lfrz612/train_data/voxceleb1
voxceleb2_root=/data_lfrz612/train_data/voxceleb2
musan_root=/data_lfrz612/train_data/musan

dir=$1
stage=$2
cv_percent=$3

if [ $stage -le -1 ]; then
    # link necessary Kaldi directories
    rm -fr utils steps sid
    ln -s $kaldi_voxceleb/v2/utils ./
    ln -s $kaldi_voxceleb/v2/steps ./
    ln -s $kaldi_voxceleb/v2/sid ./
fi


if [ $stage -le 0 ]; then
  log=exp/make_voxceleb
:<<!
  $train_cmd $log/make_voxceleb2_dev.log local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train
  $train_cmd $log/make_voxceleb2_test.log local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test
  # This script creates data/voxceleb1_test and data/voxceleb1_train.
  # Our evaluation set is the test portion of VoxCeleb1.
  #$train_cmd $log/make_voxceleb1.log local/make_voxceleb1.pl $voxceleb1_root data
  $train_cmd $log/make_voxceleb1.log local/make_voxceleb1_v2.pl $voxceleb1_root dev data/voxceleb1_train
  $train_cmd $log/make_voxceleb1.log local/make_voxceleb1_v2.pl $voxceleb1_root test data/voxceleb1_test
  # We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  # This should give 7325 speakers and 1277344 utterances.
  #$train_cmd $log/combine_voxceleb1+2.log local/combine_data.sh data/train data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train
  ln -s voxceleb2_train data/train
!
  $train_cmd $log/combine_voxceleb1_dev+test.log local/combine_data.sh data/voxceleb1 data/voxceleb1_train data/voxceleb1_test
  cp -r data/voxceleb1 data/test
  
  local/make_voxceleb1_trials.pl $trials_root/voxceleb1_clean.txt data/test/trials_o
  local/make_voxceleb1_trials.pl $trials_root/voxceleb1_E_clean.txt data/test/trials_e
  local/make_voxceleb1_trials.pl $trials_root/voxceleb1_H_clean.txt data/test/trials_h
  for trials in trials_o trials_e trials_h ;do
    cat data/test/$trials |  awk '{print $1;print $2}'
  done | sort -u > data/test/wavlist
  mv data/test/utt2spk data/test/utt2spk.bak
  utils/filter_scp.pl data/test/wavlist data/test/utt2spk.bak > data/test/utt2spk
  utils/fix_data_dir.sh data/test
  ln -s test data/val
fi

#exit
if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in train val; do
    local/make_fbank.sh --write-utt2num-frames true --fbank-config conf/fbank.conf \
      --nj $train_nj --cmd "$train_cmd" data/${name} exp/make_fbank fbank
    local/fix_data_dir.sh data/${name}
    local/compute_vad_decision.sh --nj $train_nj --cmd "$train_cmd" \
      data/${name} exp/make_vad fbank
    local/fix_data_dir.sh data/${name}
  done

  # Make MFCCs and compute the energy-based VAD for each dataset
  # NOTE: Kaldi VAD is based on MFCCs, so we need to additionally extract MFCCs
  # (https://groups.google.com/forum/#!msg/kaldi-help/-REizujqa5k/u_FJnGokBQAJ)
  for name in train val; do
    local/copy_data_dir.sh data/${name} data/${name}_mfcc
    local/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf \
      --nj $train_nj --cmd "$train_cmd" data/${name}_mfcc exp/make_mfcc mfcc
    local/fix_data_dir.sh data/${name}_mfcc
    local/compute_vad_decision.sh --nj $train_nj --cmd "$train_cmd" \
      data/${name}_mfcc exp/make_vad mfcc
    local/fix_data_dir.sh data/${name}_mfcc
  done

  # correct the right vad.scp
  for name in train val; do
    cp data/${name}_mfcc/vad.scp data/${name}/vad.scp
    local/fix_data_dir.sh data/$name
  done
fi


# In this section, we augment the VoxCeleb2 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -le 2 ]; then
  log=exp/augmentation
:<<!
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
  # additive noise here.
  $train_cmd $log/reverberate_data_dir.log \
    python2 steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/train data/train_reverb
  cp data/train/vad.scp data/train_reverb/
  local/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
  rm -rf data/train_reverb
  mv data/train_reverb.new data/train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  $train_cmd $log/make_musan.log local/make_musan.sh $musan_root data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done
!
  # Augment with musan_noise
  $train_cmd $log/augment_musan_noise.log \
    python2 steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
  # Augment with musan_music
  $train_cmd $log/augment_musan_music.log \
    python2 steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
  # Augment with musan_speech
  $train_cmd $log/augment_musan_speech.log \
    python2 steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  local/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble
fi


if [ $stage -le 3 ]; then
  # Take a random subset of the augmentations
  local/subset_data_dir.sh data/train_aug 1000000 data/train_aug_1m
  local/fix_data_dir.sh data/train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  local/make_fbank.sh --fbank-config conf/fbank.conf --nj $train_nj --cmd "$train_cmd" \
    data/train_aug_1m exp/make_fbank fbank

  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  local/combine_data.sh data/train_combined data/train_aug_1m data/train
fi


# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  for x in train_combined val;do
    local/nnet3/xvector/prepare_feats_for_egs.sh --nj $train_nj --cmd "$train_cmd" --compress false \
      data/$x data/${x}_no_sil exp/${x}_no_sil
    local/fix_data_dir.sh data/${x}_no_sil
  done
fi

mkdir -p $dir
# Now we split all data into two parts: training and cv
if [ $stage -le 5 ]; then

  # filter out utterances w/ < 800 frames
  min_length=200
  awk -v min_length=$min_length 'NR==FNR{a[$1]=$2;next}{if(a[$1]>=min_length)print}' data/train_combined_no_sil/utt2num_frames data/train_combined_no_sil/utt2spk > $dir/utt2spk
  # create spk2num_frames, this will be useful for balancing training
  awk '{if(!($2 in a))a[$2]=0;a[$2]+=1;}END{for(i in a)print i,a[i]}' $dir/utt2spk > $dir/spk2num

  # create train (100-cv_percent%) and cv (cv_percent%) utterance list
  rm -f $dir/cv.list $dir/train.list 2> /dev/null
  awk -v seed=$RANDOM -v dir=$dir -v cv=$cv_percent 'BEGIN{srand(seed);}NR==FNR{a[$1]=$2;next}{if(a[$2]<10)print $1>>dir"/train.list";else{if(rand()<=cv)print $1>>dir"/cv.list";else print $1>>dir"/train.list"}}' $dir/spk2num $dir/utt2spk

  # get the feats.scp for train and cv based on train.list and cv.list
  awk 'NR==FNR{a[$1]=1;next}{if($1 in a)print}' $dir/train.list data/train_combined_no_sil/feats.scp | sed 's/data_lfrz613/data/' | shuf > $dir/train_orig.scp
  awk 'NR==FNR{a[$1]=1;next}{if($1 in a)print}' $dir/cv.list data/train_combined_no_sil/feats.scp | sed 's/data_lfrz613/data/' | shuf > $dir/cv_orig.scp

  # maps speakers to labels (spkid)
  awk 'BEGIN{s=0;}{if(!($2 in a)){a[$2]=s;s+=1;}print $1,a[$2]}' $dir/utt2spk > $dir/utt2spkid
  awk 'BEGIN{s=0;}{if($2>s)s=$2;}END{print s+1}' $dir/utt2spkid > $dir/num_spk
  num_spk=`cat $dir/num_spk`
  echo "There are "$num_spk" number of speakers."

:<<!
  # save the uncompressed, preprocessed pytorch tensors
  # Note: this is optional!
  mkdir -p $dir/py_tensors
  python scripts/prepare_data.py --feat_scp $dir/train_orig.scp --save_dir $dir/py_tensors
  python scripts/prepare_data.py --feat_scp $dir/cv_orig.scp --save_dir $dir/py_tensors
!
fi

# !!!note that we also need to apply the same pre-processing to decode data!!!
if [ $stage -le 6 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
:<<!
  for x in train test;do
    local/nnet3/xvector/prepare_feats_for_egs.sh --nj $train_nj --cmd "$train_cmd" --compress false \
      data/$x data/${x}_no_sil exp/${x}_no_sil
    local/fix_data_dir.sh data/${x}_no_sil
    cat data/${x}_no_sil/feats.scp > $dir/decode_${x}.scp
  done
!
  # select 500k train samples randomly for compute mean vec
  utils/filter_scp.pl data/train/utt2spk data/train_combined_no_sil/feats.scp | sed 's/data_lfrz613/data/' |\
    shuf |head -500000 > $dir/decode_train.scp 
  utils/filter_scp.pl data/test/utt2spk data/test_no_sil/feats.scp | sed 's/data_lfrz613/data/' \
    > $dir/decode_test.scp 

  #utils/filter_scp.pl data/train/utt2spk data/train_combined_no_sil/feats.scp | sed 's/data_lfrz613/data/' > $dir/decode_train.scp 
  #utils/filter_scp.pl data/test/utt2spk data/test_no_sil/feats.scp | sed 's/data_lfrz613/data/' > $dir/decode_test.scp 

fi
