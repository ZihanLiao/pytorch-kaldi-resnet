#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#           2018  David Snyder
#           2019  Soonshin Seo
#
# Usage: make_voxceleb1_v2.pl /export/voxceleb1 dev data/dev
#
# The VoxCeleb1 corpus underwent several updates that changed the directory and speaker ID format.
# The script 'make_voxceleb1.pl' works for the oldest version of the corpus. 
# This script should be used if you've downloaded the corpus recently.

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-voxceleb1> <dataset> <path-to-data-dir>\n";
  print STDERR "e.g. $0 /export/voxceleb1 dev data/dev\n";
  exit(1);
}

($trials_in, $trials_out) = @ARGV;

  open(TRIAL_IN, "<", "$trials_in") or die "could not open the verification trials file $trials_in";
  open(TRIAL_OUT, ">", "$trials_out") or die "Could not open the output file $trials_out";

  my $test_spkrs = ();
  while (<TRIAL_IN>) {
    chomp;
    my ($tar_or_non, $path1, $path2) = split;
    # Create entry for left-hand side of trial
    my ($spkr_id, $rec_id, $name) = split('/', $path1);
    $name =~ s/.wav$//g;
    my $utt_id1 = "$spkr_id-$rec_id-$name";
    $test_spkrs{$spkr_id} = ();

    # Create entry for right-hand side of trial
    my ($spkr_id, $rec_id, $name) = split('/', $path2);
    $name =~ s/.wav$//g;
    my $utt_id2 = "$spkr_id-$rec_id-$name";
    $test_spkrs{$spkr_id} = ();

    my $target = "nontarget";
    if ($tar_or_non eq "1") {
      $target = "target";
    }
    print TRIAL_OUT "$utt_id1 $utt_id2 $target\n";
  }

  close(TRIAL_OUT) or die;
  close(TRIAL_IN) or die;
