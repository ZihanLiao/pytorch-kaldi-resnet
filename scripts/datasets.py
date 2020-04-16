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

class EmbeddingDataset2(Dataset):
    """PyTorch datalaoder for processing 'uncompressed' Kaldi feats.scp
    """
    def __init__(self, scp_file, chunk_size):
        """Preprocess Kaldi feats.scp here and balance the training set
        """
        self.rxfiles, self.labels = [], []

        for line in open(scp_file):
            utt, rxfile = line.rstrip().split()
            label = utt
            repetition = 1
            self.rxfiles.extend([rxfile] * repetition)
            self.labels.extend([label] * repetition)

        self.rxfiles = np.array(self.rxfiles)
        self.seq_len = chunk_size

        print("Totally "+str(len(self.rxfiles))+" samples")

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
        seq_len = self.seq_len
        assert len(full_mat) >= seq_len
        pin = np.random.randint(0, len(full_mat) - seq_len + 1)
        chunk_mat = full_mat[pin:pin+seq_len, :]
        chunk_mat = chunk_mat.T
        lab = [self.labels[index]]
        #print('feature size {}, {}'.format(chunk_mat.shape, chunk_mat))

        return chunk_mat, lab
