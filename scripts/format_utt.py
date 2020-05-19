import argparse
import sys
import os

parser = argparse.ArgumentParser(description='Usage:')
parser.add_argument('--src-type', type=str, default='live', help='live_sing or live')
parser.add_argument('--wav2label', type=str, help='')
parser.add_argument('--outpath', type=str, help='output dir')
args = parser.parse_args()

wav2label = args.wav2label
outpath = args.outpath
src_type = args.src_type

#/data_lfrz612/dataset/live_sing/wav/14529381301_AXFkXt9-xgqy2A4pgiES.mp4 14529381301 live_sing_yuanchang

fscp = open(os.path.join(outpath, "wav.scp"), 'w')
futt2spk = open(os.path.join(outpath, "utt2spk"), 'w')
labs = {}
ind = 0
for line in open(wav2label, 'r'):
    arr = line.strip().split()
    if len(arr) == 2:
        wav, lab = line.strip().split()
    elif len(arr) == 3:
        wav, lab1, lab2 = line.strip().split()
        lab = lab1+lab2
    else:
        raise Exception('format error: {}'.format(line))
    if lab not in labs:
        labs[lab] = "id{:03d}".format(ind)
        ind = ind + 1
    spk = labs[lab]

    utt = wav.split('/')[-1]
    keys = utt[:-4].split('_')
    #for i in range(len(keys)-5,len(keys):
    if src_type=='live':
        if len(keys) < 5:
            print("Error key: {}".format(wav))
            continue
        if len(keys)>=5: 
            keys = keys[-5:]
        total = [32, 8, 8, 5, 5]
    elif src_type=='live_sing':
        keys = [keys[0], '_'.join(keys[1:])]
        total = [15, 20]
    else:
        raise Exception("Wrong type: {}".format(src_type))
        
    for i in range(len(keys)):
        keys[i] = '0'*(total[i]-len(keys[i]))+keys[i]
    utt = "{}_{}".format(spk, "_".join(keys))
    fscp.write("{} {}\n".format(utt,wav))

    futt2spk.write("{} {}\n".format(utt,spk))

print("speakers: {}".format(ind))

fscp.close()
futt2spk.close()

