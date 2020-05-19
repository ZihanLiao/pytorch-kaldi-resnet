#!/bin/bash

#. ./path.sh
num_enroll=10
num_test=10
part_list=""

. utils/parse_options.sh

src_dir=$1
dir=$2
stage=3

mkdir -p $dir
if [ $stage -le 3 ];then
  python scripts/format_utt.py --src-type "live_sing" \
    --wav2label $src_dir/wav2label --outpath $dir
  utils/utt2spk_to_spk2utt.pl < $dir/utt2spk > $dir/spk2utt
  local/fix_data_dir.sh $dir
fi

if [ $stage -le 4 ];then
  utils/filter_scp.pl -f 2 "$part_list" $dir/wav.scp | local/subset_data_dir.sh --utt-list - $dir $dir/music 
  utils/filter_scp.pl --exclude $dir/music/utt2spk $dir/utt2spk | local/subset_data_dir.sh --utt-list - $dir $dir/notmusic

  local/subset_data_dir.sh --per-spk  $dir/notmusic $num_enroll $dir/enroll
  local/fix_data_dir.sh $dir/enroll

  local/filter_scp.pl --exclude $dir/enroll/utt2spk $dir/utt2spk | local/subset_data_dir.sh --utt-list - $dir/notmusic $dir/train
  local/subset_data_dir.sh --per-spk  $dir/train $num_test $dir/test
  local/fix_data_dir.sh $dir/test

  local/filter_scp.pl --exclude $dir/test/utt2spk $dir/train/utt2spk >  $dir/train/utt2spk.new
  cat $dir/train/utt2spk.new > $dir/train/utt2spk
  local/fix_data_dir.sh $dir/train

  rm -rf $dir/notmusic 
fi
