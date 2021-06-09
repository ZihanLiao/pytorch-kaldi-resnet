import os
import sys
import numpy as np
os.system(f'{sys.argv[0]} {sys.argv}')

if len(sys.argv) <= 1 or len(sys.argv) >= 3:
    os.system("echo 'Usage: {} [options] <data-dir> [<log-dir> [<fbank-dir>] ]'".format(sys.argv[0]))
    os.system("echo 'e.g.: {} data/train exp/make_fbank/train mfcc'".format(sys.argv[0]))
    os.system("echo 'Note: <log-dir> defaults to <data-dir>/log, and <fbank-dir> defaults to <data-dir>/data'")
    os.system("echo 'Options: '")
    os.system("echo '  --fbank-config <config-file>                     # config passed to compute-fbank-feats '")
    os.system("echo '  --nj <nj>                                        # number of parallel jobs'")
    os.system("echo '  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.'")
    os.system("echo '  --write-utt2num-frames <true|false>     # If true, write utt2num_frames file.'")
    exit(1)

data = sys.argv[1]


if len(sys.argv) >= 2:
    logdir = sys.argv[2]
else:
    logdir = os.path.join(data, 'log')

if len(sys.argv) >= 3:
    fbankdir = sys.argv[3]
else:
    fbankdir = os.path.join(data, 'data')

# make $fbankdir an absolute pathname.


# use "name" as part of name of the archive.
name = os.path.basename(data)

os.mkdir(fbankdir)
os.mkdir(logdir)

fbank_meta_params = []
fbank_coeff = {
                'blackman_coeff': 0.42,
                'channel': -1,
                'dither': 1.0,
                'energy_floor': 0.0,
                'frame_length': 25.0,
                'frame_shift': 10.0,
                'high_freq': 0.0,
                'htk_compat': False,
                'low_freq': 20.0,
                'min_duration': 0.0,
                'num_mel_bins': 23,
                'preemphasis_coefficient': 0.97,
                'raw_energy': True,
                'remove_dc_offset': True,
                'round_to_power_of_two': True,
                'sample_frequency': 16000.0,
                'snip_edges': True,
                'subtract_mean': False,
                'use_energy': False,
                'use_log_fbank': True,
                'use_power': True,
                'vtln_high': -500.0,
                'vtln_low': 100.0,
                'vtln_warp': 1.0,
                'window_type': 'POVEY'
}
