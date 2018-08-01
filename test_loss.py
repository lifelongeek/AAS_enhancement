import argparse

import numpy as np
import torch
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

from tqdm import tqdm

from decoder import GreedyDecoder

from data.data_loader import SpectrogramDataset, AudioDataLoader, FeatDataset, FeatLoader
#from model import DeepSpeech
from model_ken import DeepSpeech_ken, supported_rnns, DeepSpeech
from utils import *

import random


import pdb

# Test script to be compatible with train_simplified.py

def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')

#parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--gpu', default=-1, type=int)

parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam", "none"], type=str, help="Decoder to use")
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")


parser.add_argument('--preprocess', default='file', type=str, help = 'file | code(detail, file: logmel + (0,1) // code : log(1+S) + CMVN)')
#parser.add_argument('--process_mel', default=False, type=str2bool)
#parser.add_argument('--n_mels', default=40, type=int)
#parser.add_argument('--normalize', default=False, type=str2bool)

parser.add_argument('--arch_ver', default='ken', type=str, help = 'ken (1D CNN, lReLU)|orig(2D CNN, hardtanh)')

parser.add_argument('--transcript_prob', type=float, default=0.002)

args = parser.parse_args()

if __name__ == '__main__':
    if(args.arch_ver == 'ken'):
        myModel = DeepSpeech_ken
    elif(args.arch_ver == 'orig'):
        myModel = DeepSpeech

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    model = myModel.load_model(args.model_path, gpu=args.gpu)
    model.eval() # FOR DEBUG
    #model.train() # FOR DEBUG
    #print('#################################### DEBUG : model.train() ######################################')

    get_weight_statistic(model)

    labels = myModel.get_labels(model)
    """
    audio_conf = myModel.get_audio_conf(model)

    if(not audio_conf):
        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          n_mels=args.n_mels,
                          process_mel=args.process_mel)
    """

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)

    elif args.decoder == "greedy":
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))

    else:
        decoder = None
    target_decoder = GreedyDecoder(labels, blank_index=labels.index('_'))  # 왜 decoder와 따로 만드는거지?

    if(args.preprocess == 'file'):
        test_dataset = FeatDataset(manifest_filepath=args.test_manifest, labels=labels)
        test_loader = FeatLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
#    else:
 #       test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest, labels=labels, normalize=args.normalize)
 #       test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    total_cer, total_wer = 0, 0
    output_data = []

    losses = AverageMeter()
    criterion = CTCLoss()

    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data

        #pdb.set_trace()
        inputs = Variable(inputs, volatile=True)

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        if args.gpu>=0:
            inputs = inputs.cuda()

        out = model(inputs)
        out = out.transpose(0, 1)  # TxNxH

        # Decoding first & and then measuring loss
        seq_length = out.size(0)
        sizes = input_percentages.mul_(int(seq_length)).int()

        if decoder is None:
            # add output to data array, and continue
            output_data.append((out.data.cpu().numpy(), sizes.numpy()))
            continue

        decoded_output, _, = decoder.decode(out.data, sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)
        wer, cer = 0, 0

        for x in range(len(target_strings)):
            decoding, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(decoding, reference) / float(len(reference.split()))
            cer_inst = decoder.cer(decoding, reference) / float(len(reference))
            wer += wer_inst
            cer += cer_inst
            if (random.uniform(0, 1) < args.transcript_prob):
                print('reference = ' + reference)
                print('decoding = ' + decoding)
                print('wer = ' + str(wer_inst) + ', cer = ' + str(cer_inst))
        total_cer += cer
        total_wer += wer


        # Part2. Measuring loss
        #targets = _get_variable_nograd(targets, cuda=False)
        #target_sizes = _get_variable_nograd(target_sizes, cuda=False)
        #sizes = _get_variable_nograd(sizes, cuda=False)

        targets = Variable(targets, requires_grad=False)
        target_sizes = Variable(target_sizes, requires_grad=False)
        sizes = Variable(sizes, requires_grad=False)

        l_CTC = criterion(out , targets, sizes, target_sizes)
        N = target_sizes.size(0) # minibatch size
        #print('N = ' + str(N))  # correct minibatch size
        l_CTC = l_CTC/N
        losses.update(l_CTC.data[0], N)

    wer = total_wer / len(test_loader.dataset)
    cer = total_cer / len(test_loader.dataset)

    print('Test Summary \t'
            'Average WER {wer:.3f}\t'
            'Average CER {cer:.3f}\t'
            'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(wer=wer * 100, cer=cer * 100, loss=losses))
