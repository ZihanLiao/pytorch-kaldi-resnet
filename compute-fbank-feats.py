import os
import sys
import numpy as np
import torch
import torchaudio
from kaldi import fbank
from make_fbank import fbank_coeff
from IPython.display import  Audio, display

PATH = os.path.basename(os.getcwd())
WAV_PATH = os.path.join(PATH, 'wav_file')

#
# feat_fbank = fbank(waveform,
#               fbank_coeff['blackman_coeff'],
#               fbank_coeff['channel'],
#               fbank_coeff['dither'],
#               fbank_coeff['energy_floor'],
#               fbank_coeff['frame_length'],
#               fbank_coeff['frame_shift'],
#               fbank_coeff['high_freq'],
#               fbank_coeff['htk_compat'],
#               fbank_coeff['low_freq'],
#               fbank_coeff['min_duration'],
#               fbank_coeff['num_mel_bins'],
#               fbank_coeff['preemphasis_coefficient'],
#               fbank_coeff['raw_energy'],
#               fbank_coeff['remove_dc_offset'],
#               fbank_coeff['round_to_power_of_two'],
#               fbank_coeff['sample_frequency'],
#               fbank_coeff['snip_edges'],
#               fbank_coeff['subtract_mean'],
#               fbank_coeff['use_energy'],
#               fbank_coeff['use_log_fbank'],
#               fbank_coeff['use_power'],
#               fbank_coeff['vtln_high'],
#               fbank_coeff['vtln_low'],
#               fbank_coeff['vtln_warp'],
#               fbank_coeff['window_type']
#                    )

feat_fbank = fbank(waveform, **fbank_coeff)

