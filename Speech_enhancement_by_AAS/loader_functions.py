import random
import os
import numpy as np

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class FeatDataset(Dataset):
    def __init__(self, manifest, labels):
        with open(manifest) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(FeatDataset, self).__init__()

    def __getitem__(self, index):
        sample = self.ids[index]
        if(len(sample) == 2):
            feat_path, transcript_path = sample[0], sample[1]
        else:
            feat_path, transcript_path, feat_paired_path = sample[0], sample[1], sample[2]

        feat = torch.load(feat_path)
        transcript = self.parse_transcript(transcript_path)

        if(len(sample) == 2):
            return feat, transcript
        else:
            feat_paired = torch.load(feat_paired_path)
            return feat, transcript, feat_paired

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    mask = torch.ByteTensor(minibatch_size, 1, max_seqlength).zero_() # dimension size 1 is for further expand()
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
        if(seq_length < max_seqlength):
            mask[x][:, seq_length:].fill_(1)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes, mask

def _collate_fn_paired(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, freq_size, max_seqlength)
    outputs = torch.zeros(minibatch_size, freq_size, max_seqlength) # outputs has same size with inputs
    mask = torch.ByteTensor(minibatch_size, 1, max_seqlength).zero_() # dimension size 1 is for further expand()
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        txt = sample[1]
        target = sample[2]
        seq_length = tensor.size(1)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
        outputs[x].narrow(1, 0, seq_length).copy_(target)
        if(seq_length < max_seqlength):
            mask[x][:, seq_length:].fill_(1)

        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(txt)
        targets.extend(txt)
    targets = torch.IntTensor(targets)
    return inputs, outputs, mask, targets, input_percentages, target_sizes

class FeatLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(FeatLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

class FeatLoader_paired(DataLoader):
    def __init__(self, *args, **kwargs):
        super(FeatLoader_paired, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_paired


class FeatSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(FeatSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self):
        np.random.shuffle(self.bins)