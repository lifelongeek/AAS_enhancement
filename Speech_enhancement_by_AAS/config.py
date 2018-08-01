#-*- coding: utf-8 -*-
import argparse
import pdb

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Basic options
parser.add_argument('--trainer', type=str, default = 'AAS')
parser.add_argument('--mode', type=str, default = 'train', help = 'train | test | visualize')
parser.add_argument('--simul_real', type=str, default = 'real', help = 'simul | real | simulreal')
parser.add_argument('--DB_name', type=str, default='librispeech', help = 'librispeech | chime')
parser.add_argument('--expnum', type=int, default=0)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--print_every', type = int, default=100)
parser.add_argument('--load_path', type = str, default='')
parser.add_argument('--ASR_path', type = str, default='')



# Manifest list
parser.add_argument('--tr_cl_manifest', default='')
parser.add_argument('--tr_ny_manifest', default='')
parser.add_argument('--trsub_manifest', default='')
parser.add_argument('--val_manifest', default='')
parser.add_argument('--val2_manifest', default='')

parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for transcription')

# Speech enhancement architecture
parser.add_argument('--nFeat', default=40, type=int)
parser.add_argument('--rnn_size', default=500, type=int, help='Hidden size of RNNs')
parser.add_argument('--rnn_layers', default=4, type=int, help='Number of RNN layers')
parser.add_argument('--rnn_type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')

# Optimization
parser.add_argument('--epochs', default=300, type=int, help='Number of training epochs')
parser.add_argument('--start_iter', default=0, type=int)
parser.add_argument('--max_iter', default = 30000000, type=int)
parser.add_argument('--log_iter', default = 100, type=int)
parser.add_argument('--save_iter', default = 1000, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--w_acoustic', default=1, type=float)
parser.add_argument('--w_adversarial', default=1, type=float)
parser.add_argument('--allow_ASR_update_iter', type = int, default=0)

parser.add_argument('--gamma', type = float, default = 0.5, help = 'began parameter')
parser.add_argument('--lambda_k', type = float, default = 0.001, help = 'began parameter')


parser.add_argument('--optimizer', default='adam', help='adam|sgd')
parser.add_argument('--random_seed', type=int, default=123)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)

def get_config():
    config, unparsed = parser.parse_known_args()
    if(len(unparsed) > 0):
        print(unparsed)
        assert(len(unparsed) == 0), 'length of unparsed option should be 0'
    return config, unparsed
