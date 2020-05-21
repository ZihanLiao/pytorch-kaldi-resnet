import numpy as np
from torch.utils.data import Dataset
import kaldi_io

# Author: Nanxin Chen, Cheng-I Lai

class SequenceDataset(Dataset):
    """PyTorch datalaoder for processing 'uncompressed' Kaldi feats.scp
    """
    def __init__(self, scp_file, utt2spkid_file, chunk_size):
        """Preprocess Kaldi feats.scp here and balance the training set
        """
        self.rxfiles, self.labels, self.seq_len, self.utt2spkid = [], [], [], {}

        # balanced training
        id_count = {}
        for line in open(utt2spkid_file):
            utt, label = line.rstrip().split()
            self.utt2spkid[utt] = int(label)
            if not int(label) in id_count:
                id_count[int(label)] = 0
            id_count[int(label)] += 1
        max_id_count = int((max(id_count.values())+1)/2)
        max_id_count = min(500, max_id_count)

        for line in open(scp_file):
            utt, rxfile = line.rstrip().split()
            label = self.utt2spkid[utt]
            repetition = max(1, max_id_count // id_count[label])
            self.rxfiles.extend([rxfile] * repetition)
            self.labels.extend([label] * repetition)

        self.rxfiles = np.array(self.rxfiles)
        self.labels  = np.array(self.labels, dtype=np.int)
        if isinstance(chunk_size, int):
            self.seq_len = np.array([chunk_size]*len(self.labels), dtype=np.int)
        if isinstance(chunk_size, list):
            if len(chunk_size)==1:
                self.seq_len = np.array(chunk_size*len(self.labels), dtype=np.int)
            elif len(chunk_size)>1:
                min_chunk_size = min(chunk_size)
                max_chunk_size = max(chunk_size)
                self.seq_len = np.random.randint(min_chunk_size, max_chunk_size+1, size=len(self.labels))

        print("Totally "+str(len(self.rxfiles))+" samples with at most "+
            str(max_id_count)+" samples for one class")

    def __len__(self):
        """Return number of samples
        """
        return len(self.labels)

    def set_chunk_size(self, seq_len):
        """Update the self.seq_len. We call this in the main training loop
        once per training iteration.
        """
        self.seq_len = seq_len

    def __getitem__(self, index):
        """Generate samples
        """
        rxfile  = self.rxfiles[index]
        full_mat = kaldi_io.read_mat(rxfile)
        seq_len = self.seq_len[index]
        assert len(full_mat) >= seq_len
        pin = np.random.randint(0, len(full_mat) - seq_len + 1)
        chunk_mat = full_mat[pin:pin+seq_len, :]
        chunk_mat = chunk_mat.T
        lab = np.array(self.labels[index])
        #print('feature size {}, {}'.format(chunk_mat.shape, chunk_mat))

        return chunk_mat, lab

class SequenceDataset2(Dataset):
    """PyTorch datalaoder for processing 'uncompressed' Kaldi feats.scp
    """
    def __init__(self, scp_file, utt2spkid_file, chunk_size):
        """Preprocess Kaldi feats.scp here and balance the training set
        """
        self.rxfiles = {}
        self.labels = []

        # balanced training
        utt2spkid, spkid2utt = {}, {}
        for line in open(utt2spkid_file):
            utt, spkid = line.rstrip().split()
            spkid = int(spkid)
            utt2spkid[utt] = spkid

        id_count = {}
        for line in open(scp_file):
            utt, rxfile = line.rstrip().split()
            spkid = utt2spkid[utt]
            if not spkid in id_count:
                id_count[spkid] = 0
            id_count[spkid] += 1
            if not spkid in self.rxfiles:
                self.labels.extend([spkid])
                self.rxfiles[spkid] = []
            self.rxfiles[spkid].append(rxfile) 

        self.repetition = int((max(id_count.values())+1)/2)
        # average duration is 7s
        #self.repetition = max(id_count.values()) * 700 // chunk_size
        print("id_count: {}".format(max(id_count.values())))

        #self.rxfiles = np.array(self.rxfiles)
        self.labels.sort()
        self.labels = np.array(self.labels)
        self.seq_len = chunk_size
        self.num_spk = len(self.rxfiles)

        print("Totally "+str(self.num_spk)+" speakers with at most "+
            str(self.repetition)+" samples for one class")

    def __len__(self):
        """Return number of samples
        """
        return len(self.labels)*self.repetition

    def set_chunk_size(self, seq_len):
        """Update the self.seq_len. We call this in the main training loop
        once per training iteration.
        """
        self.seq_len = seq_len

    def __getitem__(self, index):
        """Generate samples
        """
        i = index % (self.num_spk)
        spkid = self.labels[i]
        #rxfiles  = self.rxfiles[spkid]
        uttid = np.random.randint(0, len(self.rxfiles[spkid]))
        rxfile = self.rxfiles[spkid][uttid]
        #print('num_spk: {}, index: {}, i: {}, spkid: {}, uttid: {}, rxfile: {}'.format(self.num_spk, index, i, spkid, uttid, rxfile))

        full_mat = kaldi_io.read_mat(rxfile)
        seq_len = self.seq_len
        assert len(full_mat) >= seq_len
        pin = np.random.randint(0, len(full_mat) - seq_len + 1)
        chunk_mat = full_mat[pin:pin+seq_len, :]
        chunk_mat = chunk_mat.T
        lab = np.array(spkid)
        #print('feature size {}, {}'.format(chunk_mat.shape, chunk_mat))

        return chunk_mat, lab

class EmbeddingDataset(Dataset):
    """PyTorch datalaoder for processing 'uncompressed' Kaldi feats.scp
    """
    def __init__(self, scp_file, chunk_size=-1):
        """Preprocess Kaldi feats.scp here and balance the training set
        """
        self.rxfiles, self.utts = [], []

        for line in open(scp_file):
            utt, rxfile = line.rstrip().split()
            repetition = 1
            self.rxfiles.extend([rxfile] * repetition)
            self.utts.extend([utt] * repetition)

        self.rxfiles = np.array(self.rxfiles)
        self.seq_len = chunk_size

        print("Totally "+str(len(self.rxfiles))+" samples")

    def __len__(self):
        """Return number of samples
        """
        return len(self.rxfiles)

    def set_chunk_size(self, seq_len):
        """Update the self.seq_len. We call this in the main training loop
        once per training iteration.
        """
        self.seq_len = seq_len

    def __getitem__(self, index):
        """Generate samples
        """
        utt = [self.utts[index]] #batch must contain tensors, numpy arrays, numbers, dicts or lists;
        rxfile  = self.rxfiles[index]
        full_mat = kaldi_io.read_mat(rxfile)
        assert len(full_mat) >= self.seq_len
        if self.seq_len > -1:
            pin = np.random.randint(0, len(full_mat) - self.seq_len + 1)
            chunk_mat = full_mat[pin:pin+self.seq_len, :]
        else:
            chunk_mat = full_mat
        chunk_mat = chunk_mat.T
        #print('feature size {}, {}'.format(chunk_mat.shape, chunk_mat))

        return chunk_mat, utt
