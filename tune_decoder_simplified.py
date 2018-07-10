import argparse
import json
import sys
from multiprocessing import Pool
import logging

import numpy as np
import torch

from data.data_loader import FeatDataset, FeatLoader
from decoder import GreedyDecoder, BeamCTCDecoder
from model_ken import DeepSpeech_ken, supported_rnns
from utils import *

import random

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--logits', default="", type=str, help='Path to logits from test.py')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--expnum', default=0, type=int)

parser.add_argument('--num_workers', default=8, type=int, help='Number of parallel decodes to run')
parser.add_argument('--log_path', default="decoding/log/log.json", help="Where to save tuning results")
parser.add_argument('--result_path', default="decoding/result/result.txt", help="write final result")


parser.add_argument('--detail_log_path', default="decoding/log_detail/log.txt", help="write final result")
parser.add_argument('--detail_log_print_prob', default=0.005)





beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--lm_alpha_from', default=1, type=float, help='Language model weight start tuning')
beam_args.add_argument('--lm_alpha_to', default=3.2, type=float, help='Language model weight end tuning')
beam_args.add_argument('--lm_beta_from', default=0.0, type=float,
                       help='Language model word bonus (all words) start tuning')
beam_args.add_argument('--lm_beta_to', default=0.45, type=float,
                       help='Language model word bonus (all words) end tuning')
beam_args.add_argument('--lm_num_alphas', default=45, type=float, help='Number of alpha candidates for tuning')
beam_args.add_argument('--lm_num_betas', default=8, type=float, help='Number of beta candidates for tuning')
beam_args.add_argument('--cutoff_top_n', default=40, type=int,
                       help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                            'vocabulary will be used in beam search, default 40.')
beam_args.add_argument('--cutoff_prob', default=1.0, type=float,
                       help='Cutoff probability in pruning,default 1.0, no pruning.')

args = parser.parse_args()


# Ver2
FORMAT = '%(message)s'
logging.basicConfig(format=FORMAT)

log_file_handler = logging.FileHandler(args.detail_log_path)
log_file_handler.setFormatter(logging.Formatter(FORMAT))
logger = logging.getLogger()
logger.addHandler(log_file_handler)


#def decode_dataset(logits, test_dataset, batch_size, lm_alpha, lm_beta, mesh_x, mesh_y, labels, logFile):
def decode_dataset(logits, test_dataset, batch_size, lm_alpha, lm_beta, mesh_x, mesh_y, labels):
    print("Beginning decode for {}, {}".format(lm_alpha, lm_beta))
    test_loader = FeatLoader(test_dataset, batch_size=batch_size, num_workers=0)
    target_decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    decoder = BeamCTCDecoder(labels, beam_width=args.beam_width, cutoff_top_n=args.cutoff_top_n,
                             blank_index=labels.index('_'), lm_path=args.lm_path,
                             alpha=lm_alpha, beta=lm_beta, num_processes=1)
    total_cer, total_wer = 0, 0
    #decoding_log = []
    for i, (data) in enumerate(test_loader):
        inputs, targets, input_percentages, target_sizes = data

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out = torch.from_numpy(logits[i][0])
        sizes = torch.from_numpy(logits[i][1])

        decoded_output, _= decoder.decode(out, sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)
        wer, cer = 0, 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference) / float(len(reference.split()))
            cer_inst = decoder.cer(transcript, reference) / float(len(reference))
            wer += wer_inst
            cer += cer_inst

            # ver1
            # write result to logFile # can't do this because multi processing code cannot do this
            #logFile.write('decoding : ' + transcript)
            #logFIle.write('reference : ' + reference)
            #logFile.write('WER = ' + str(wer_inst) + ', CER = ' + str(cer_inst))

            if(random.uniform(0, 1) < float(args.detail_log_print_prob)):
                print('decoding : ' + transcript)
                print('reference : ' + reference)
                print('WER = ' + str(wer_inst) + ', CER = ' + str(cer_inst))
                print(' ')

                #ver1
                #decoding_log_sample = []
                #decoding_log_sample.append(transcript)
                #decoding_log_sample.append(reference)
                #decoding_log.append(decoding_log_sample)

                #ver2. thread safe but does not write anything to file
                #logging.info('decoding : ' + transcript)
                #logging.info('reference : ' + reference)
                #logging.info('WER = ' + str(wer_inst) + ', CER = ' + str(cer_inst))
                #logging.info(' ')


                #ver3
                logger.error('decoding : ' + transcript)
                logger.error('reference : ' + reference)
                logger.error('WER = ' + str(wer_inst) + ', CER = ' + str(cer_inst))
                logger.error(' ')

        total_cer += cer
        total_wer += wer

    wer = total_wer / len(test_loader.dataset)
    cer = total_cer / len(test_loader.dataset)

    return [mesh_x, mesh_y, lm_alpha, lm_beta, wer, cer]

    # Ver1
    #return [mesh_x, mesh_y, lm_alpha, lm_beta, wer, cer], decoding_log


def getWER(item):
    return item[5]

if __name__ == '__main__':
    if args.lm_path is None:
        print("error: LM must be provided for tuning")
        sys.exit(1)

    model = DeepSpeech_ken.load_model(args.model_path, gpu=-1) # run
    model.eval()

    labels = DeepSpeech_ken.get_labels(model)
    test_dataset = FeatDataset(manifest_filepath=args.test_manifest, labels=labels)

    logits = np.load(args.logits)
    batch_size = logits[0][0].shape[1]

    results = []

    # Ver1. print is not thread safe
    #logFile = open(args.log_path, 'w')

    def result_callback(result):
        results.append(result)


    p = Pool(args.num_workers)

    cand_alphas = np.linspace(args.lm_alpha_from, args.lm_alpha_to, args.lm_num_alphas)
    cand_betas = np.linspace(args.lm_beta_from, args.lm_beta_to, args.lm_num_betas)
    params_grid = []
    for x, alpha in enumerate(cand_alphas):
        for y, beta in enumerate(cand_betas):
            params_grid.append((alpha, beta, x, y))

    futures = []
    for index, (alpha, beta, x, y) in enumerate(params_grid):
        print("Scheduling decode for a={}, b={} ({},{}).".format(alpha, beta, x, y))
        f = p.apply_async(decode_dataset, (logits, test_dataset, batch_size, alpha, beta, x, y, labels),callback=result_callback)

        # Ver1
        #f = p.apply_async(decode_dataset, (logits, test_dataset, batch_size, alpha, beta, x, y, labels, logFile), callback=result_callback) # can't write logFile in multiprocessing mode


        futures.append(f)

    for f in futures:
        f.wait()
        print("Result calculated:", f.get())


    print("Saving tuning log to: {}".format(args.log_path))
    with open(args.log_path, "w") as fh:
        json.dump(results, fh)

    print("Saving final best result to: {}".format(args.result_path))
    sorted_results = sorted(results, key=getWER)

    with open(args.result_path, "w") as fh:
        json.dump(sorted_results[0], fh)

    print('BEST_RESULTS = ')
    print(sorted_results[0])