import argparse

import numpy as np
from torch.autograd import Variable
from tqdm import tqdm

from decoder import GreedyDecoder

from data.data_loader import SpectrogramDataset, AudioDataLoader, FeatDataset, FeatLoader
#from model import DeepSpeech
from model_ken import DeepSpeech_ken, supported_rnns
from utils import *

import random
import torch

import logging

import pdb

# Test script to be compatible with train_simplified.py

def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--result_path', default='')
parser.add_argument('--detail_log_path', default='')


#parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--gpu', default=-1, type=int)

parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch_size', default=40, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam", "none"], type=str, help="Decoder to use")
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")


parser.add_argument('--preprocess', default='file', type=str, help = 'file | code(detail, file: logmel + (0,1) // code : log(1+S) + CMVN)')
parser.add_argument('--process_mel', default=False, type=str2bool)
parser.add_argument('--n_mels', default=40, type=int)
parser.add_argument('--normalize', default=False, type=str2bool)

parser.add_argument('--arch_ver', default='ken', type=str, help = 'ken (1D CNN, lReLU)|orig(2D CNN, hardtanh)')

parser.add_argument('--transcript_prob', type=float, default=0.002)

# for audio_conf (only used when audio_conf is not stored in advance)
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')


no_decoder_args = parser.add_argument_group("No Decoder Options", "Configuration options for when no decoder is "
                                                                  "specified")
no_decoder_args.add_argument('--output_path', default=None, type=str, help="Where to save raw acoustic output")

beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--top_paths', default=1, type=int, help='number of beams to return')
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
beam_args.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--cutoff_top_n', default=40, type=int,
                       help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                            'vocabulary will be used in beam search, default 40.')
beam_args.add_argument('--cutoff_prob', default=1.0, type=float,
                       help='Cutoff probability in pruning,default 1.0, no pruning.')
beam_args.add_argument('--lm_workers', default=1, type=int, help='Number of LM processes to use')
args = parser.parse_args()

if __name__ == '__main__':
    if(args.arch_ver == 'ken'):
        myModel = DeepSpeech_ken
    elif(args.arch_ver == 'orig'):
        myModel = DeepSpeech

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    FORMAT = '%(message)s'
    logging.basicConfig(format=FORMAT)
    log_file_handler = logging.FileHandler(args.detail_log_path)
    log_file_handler.setFormatter(logging.Formatter(FORMAT))
    logger = logging.getLogger()
    logger.addHandler(log_file_handler)

    model = myModel.load_model(args.model_path, gpu=args.gpu)
    model.eval() # FOR DEBUG
    #model.train() # FOR DEBUG
    #print('#################################### DEBUG : model.train() ######################################')

    get_weight_statistic(model)
    if args.decoder == 'greedy' or args.decoder == 'beam':
        result_file = open(args.result_path, 'w')

    labels = myModel.get_labels(model)
    audio_conf = myModel.get_audio_conf(model)

    if(not audio_conf):
        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          n_mels=args.n_mels,
                          process_mel=args.process_mel)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    else:
        decoder = None
    target_decoder = GreedyDecoder(labels, blank_index=labels.index('_'))

    if(args.preprocess == 'file'):
        test_dataset = FeatDataset(manifest_filepath=args.test_manifest, labels=labels)
        test_loader = FeatLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest, labels=labels, normalize=args.normalize)
        test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    total_cer, total_wer = 0, 0
    output_data = []

    #pdb.set_trace()
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
        #pdb.set_trace()
        out = out.transpose(0, 1)  # TxNxH
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
            #if(random.uniform(0, 1) < 1): # debug
                print('reference = ' + reference)
                print('decoding = ' + decoding)
                print('wer = ' + str(wer_inst) + ', cer = ' + str(cer_inst))

                logger.error('decoding : ' + decoding)
                logger.error('reference : ' + reference)
                logger.error('WER = ' + str(wer_inst) + ', CER = ' + str(cer_inst))
                logger.error(' ')

        total_cer += cer
        total_wer += wer

    if decoder is not None:
        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)

        print('Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer * 100, cer=cer * 100))

        result_file.write('Test Summary')
        result_file.write('Average WER : ' + str(wer*100))
        result_file.write('Average CER : ' + str(cer*100))
        result_file.close()
    else:
        np.save(args.output_path, output_data)
