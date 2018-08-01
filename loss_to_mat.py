import torch
import argparse
import scipy.io as sio
import os
import pdb
import numpy

parser = argparse.ArgumentParser()
#parser.add_argument('--checkpoint', default='', help='Checkpoint path')
parser.add_argument('--expnum', default=0)
parser.add_argument('--DB_name', default='librispeech')
args = parser.parse_args()

args.checkpoint = 'models/' + args.DB_name + '_' + str(args.expnum) + '.pth.tar'



print('!!!!!!!!!!!!!!! CHECKPOINT = ' + args.checkpoint + ' !!!!!!!!!!!!!!!!!!!!')



checkpoint = torch.load(args.checkpoint, map_location = lambda storage, loc: storage)
epoch = int(checkpoint.get('epoch',1))
loss_results, cer_results, wer_results = checkpoint['loss_results'][:epoch], checkpoint['cer_results'][:epoch], checkpoint['wer_results'][:epoch]

save_dir = 'logs/' + args.DB_name + '/' + str(args.expnum)
if not os.path.exists(save_dir):
    print('make directory ' + save_dir )
    os.makedirs(save_dir)

#pdb.set_trace()


sio.savemat(save_dir + '/loss.mat', {'loss':loss_results.numpy()})
sio.savemat(save_dir + '/wer.mat', {'wer':wer_results.numpy()})
sio.savemat(save_dir + '/cer.mat', {'cer':cer_results.numpy()})

print('save results to ' + save_dir)
