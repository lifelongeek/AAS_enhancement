import argparse
import errno
import json
import os
import time

import torch
from tqdm import tqdm
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
from data.data_loader import FeatLoader, FeatDataset, AudioDataLoader, SpectrogramDataset, BucketingSampler
from decoder import GreedyDecoder
#from model import DeepSpeech, supported_rnns
from model_ken import DeepSpeech_ken, supported_rnns, ResidualDeepSpeech, ResidualCNN4block #DeepSpeech
from utils import *
import random
import decimal
import pdb


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--DB_name', type=str, default='librispeech')
parser.add_argument('--expnum', type=int, default=0)


parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/librispeech_logMel_train_manifest.csv')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/librispeech_logMel_val_manifest.csv')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for transcription')

# Used only preprocess = code
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')

parser.add_argument('--rnn_size', default=500, type=int, help='Hidden size of RNNs')
parser.add_argument('--rnn_layers', default=2, type=int, help='Number of RNN layers')
parser.add_argument('--rnn_type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')

parser.add_argument('--conv_layers', default=2, type=int)
parser.add_argument('--cnn_residual_blocks', default=1, type=int) # used for ResidualDeepSpeech

parser.add_argument('--conv_map', default=256, type=int)
parser.add_argument('--conv_kernel', default=11, type=int)
parser.add_argument('--conv_stride', default=2, type=int)

parser.add_argument('--nFreq', default=40, type=int) # for mel spectrogram : nFreq = 40
parser.add_argument('--n_mels', default=40, type=int)

#parser.add_argument('--feat_type', default='mel', type=str, help = 'mel | linear (detail, mel : logmel + (0,1) // linear : log(1+S) + CMVN)')
parser.add_argument('--preprocess', default='file', type=str, help = 'file | code(detail, file: logmel + (0,1) // code : log(1+S) + CMVN)')
parser.add_argument('--process_mel', default=False, type=str2bool)
parser.add_argument('--normalize', default=False, type=str2bool)

parser.add_argument('--arch_ver', default='ken', type=str, help = 'ken (1D CNN, lReLU)|orig(2D CNN, hardtanh)')

parser.add_argument('--nDownsample', type = int, default=1)
parser.add_argument('--print_every', type = int, default=100)



parser.add_argument('--epochs', default=300, type=int, help='Number of training epochs')
#parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--gpu', default=-1, type=int)

parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--optim', default='adam', help='adam|sgd')

parser.add_argument('--one_sample_DEBUG', default=False, type=str2bool)

parser.add_argument('--include_first_BN', default=True, type=str2bool)


#parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients') # do not use gradient clipping
#parser.add_argument('--learning_anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch') # do not use learnin rate annealing
#parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
#parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
#parser.add_argument('--checkpoint_per_batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
#parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
#parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
#parser.add_argument('--log_dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log_params', dest='log_params', action='store_true', help='Log parameter values and gradients')
#parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model_path', default='', help='Location to save best validation model')  # set up this in code
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
#parser.add_argument('--finetune', dest='finetune', action='store_true',
#                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', type=str2bool,  default=False, help='Use random tempo and gain perturbations.')
parser.add_argument('--transcript_prob', type=float, default=0.002)
#parser.add_argument('--noise_dir', default=None,
#                    help='Directory to inject noise into audio. If default, noise Inject not added')
#parser.add_argument('--noise_prob', default=0.4, help='Probability of noise being added per sample')
#parser.add_argument('--noise_min', default=0.0,
#                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
#parser.add_argument('--noise_max', default=0.5,
#                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
#parser.add_argument('--no_shuffle', dest='no_shuffle', action='store_true',
#                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--sortagrad', default=False, type=str2bool, help='load minibatch with order of increasing length from shorter to longer')

#parser.add_argument('--no_bidirectional', dest='bidirectional', action='store_false', default=True,
#                    help='Turn off bi-directional RNNs, introduces lookahead convolution')

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

if __name__ == '__main__':
    #args = parser.parse_args()

    print('pass')
    args, unparsed = parser.parse_known_args()
    #pdb.set_trace()
    if(len(unparsed) > 0):
        print(unparsed)
        assert(len(unparsed) == 0), 'length of unparsed option should be 0'

    target_source = ['train_simplified.py', 'model_ken.py']
    check_config_used(args, target_source)

    save_folder = args.save_folder

    if(args.process_mel):
        args.nFreq = args.n_mels

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    args.model_path = 'models/' + args.DB_name + '_' + str(args.expnum) + '_final.pth.tar'

    loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(args.epochs)
    param_norm_tensor, param_abs_max_tensor, grad_norm_tensor, grad_abs_max_tensor = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(args.epochs)
    best_wer = None

    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Model Save directory already exists.')
        else:
            raise
    criterion = CTCLoss()

    if(args.arch_ver == 'ken'):
        myModel = DeepSpeech_ken
    #elif(args.arch_ver == 'orig'):
#        myModel = DeepSpeech
    elif(args.arch_ver == 'ResidualDeepSpeech'):
        myModel = ResidualDeepSpeech
    elif(args.arch_ver == 'ResidualCNN4block'):
        myModel = ResidualCNN4block

    avg_loss, start_epoch, start_iter = 0, 0, 0
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from, map_location = lambda storage, loc: storage)
        model = myModel.load_model_package(package)
        labels = myModel.get_labels(model)
        #audio_conf = DeepSpeech.get_audio_conf(model)
        parameters = model.parameters()

        if(args.optim == 'adam'):
            optimizer = torch.optim.Adam(parameters, lr=args.lr)
        elif(args.optim == 'sgd'):
            optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, nesterov=True)

        optimizer.load_state_dict(package['optim_dict'])
        start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
        start_iter = package.get('iteration', None)
        if start_iter is None:
            start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
            start_iter = 0
        else:
            start_iter += 1
        avg_loss = int(package.get('avg_loss', 0))
        loss_results, cer_results, wer_results = package['loss_results'], package['cer_results'], package['wer_results']
    else:
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        if(args.preprocess == 'code'):
            audio_conf = dict(sample_rate=args.sample_rate,
                              window_size=args.window_size,
                              window_stride=args.window_stride,
                              window=args.window,
                              n_mels=args.n_mels,
                              process_mel = args.process_mel)

            if(not args.process_mel):
                args.nFreq = int(args.sample_rate * args.window_size/2 +1) # 16000*0.020/2+1 = 161
        else:
            audio_conf = None

        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
        if(args.arch_ver == 'ResidualDeepSpeech'):
            model = myModel(rnn_hidden_size=args.rnn_size,
                               rnn_layers=args.rnn_layers,
                               rnn_type=supported_rnns[rnn_type],
                               labels=labels,
                               audio_conf=audio_conf,
                               kernel_sz=args.conv_kernel,
                               stride=args.conv_stride,
                               map=args.conv_map,
                               cnn_residual_blocks=args.cnn_residual_blocks,
                               nFreq=args.nFreq,
                               nDownsample=args.nDownsample
                               )
        elif (args.arch_ver == 'ResidualCNN4block'):
                model = myModel(labels=labels,
                                audio_conf=audio_conf,
                                kernel_sz=args.conv_kernel,
                                stride=args.conv_stride,
                                map=args.conv_map,
                                nFreq=args.nFreq,
                                nDownsample=args.nDownsample
                                )
        else:
            model = myModel(rnn_hidden_size=args.rnn_size,
                               rnn_layers=args.rnn_layers,
                               rnn_type=supported_rnns[rnn_type],
                               labels=labels,
                               audio_conf=audio_conf,
                               kernel_sz=args.conv_kernel,
                               stride=args.conv_stride,
                               map=args.conv_map,
                               cnn_layers=args.conv_layers,
                               nFreq=args.nFreq,
                               nDownsample=args.nDownsample,
                               include_first_BN=args.include_first_BN
                               )

        model.apply(weights_init)
        parameters = model.parameters()
        #pdb.set_trace()

        if(args.optim == 'adam'):
            optimizer = torch.optim.Adam(parameters, lr=args.lr)
        elif(args.optim == 'sgd'):
            optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, nesterov=True)

    decoder = GreedyDecoder(labels)
    if(args.preprocess == 'file'):
        train_dataset = FeatDataset(manifest_filepath=args.train_manifest, labels=labels)
        test_dataset = FeatDataset(manifest_filepath=args.val_manifest, labels=labels)
    elif(args.preprocess == 'code'):
        train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                           normalize=args.normalize, augment=args.augment)
        test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=args.normalize, augment=False)

    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)

    if(args.preprocess == 'file'):
        train_loader = FeatLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler)
        test_loader = FeatLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    elif(args.preprocess == 'code'):
        train_loader = AudioDataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler)
        test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if not (args.sortagrad and start_epoch == 0):
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle()

    print(model)
    print("Number of parameters: %d" % myModel.get_param_size(model))

    #print('after init')
    get_weight_statistic(model)
    #pdb.set_trace()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    if args.gpu >=0:
        model = model.cuda()

    # Save model file for error check
    file_path = '%s/%s_%d.pth.tar' % (save_folder, args.DB_name, args.expnum)  # always overwrite recent epoch's model
    torch.save(myModel.serialize(model, optimizer=optimizer, epoch=0),
                    file_path)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        end = time.time()
        #pdb.set_trace()
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            input, target, input_percentages, target_size = data

            # measure data loading time
            data_time.update(time.time() - end)
            inputs = Variable(input, requires_grad=False)
            target_sizes = Variable(target_size, requires_grad=False)
            targets = Variable(target, requires_grad=False)

            if args.gpu >=0:
                inputs = inputs.cuda()

            #pdb.set_trace()
            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH

            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)

            #pdb.set_trace()
            loss = criterion(out, targets, sizes, target_sizes)
            loss = loss / inputs.size(0)  # average the loss by minibatch

            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.data[0]

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # compute gradient
            #pdb.set_trace()
            optimizer.zero_grad()

            # Ver1
            """
            if(args.one_sample_DEBUG):
                print('backward with retain graph')
                loss.backward(retain_graph=True)                else:
                loss.backward()
            """

            # Ver2
            loss.backward()

            #torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm) # remove gradient clipping
            optimizer.step()

            if args.gpu >= 0:
                torch.cuda.synchronize()  # don't need this because DataParallel is disabled

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if(i%args.print_every == 0):
                param_norm = 0
                param_abs_max = -100
                grad_norm = 0
                grad_abs_max = -100
                for param in model.parameters():
                    param_norm += param.data.norm()                    
                    param_max = param.data.abs().max()
                    if(param_max > param_abs_max):
                        param_abs_max = param_max
                        
                    if param.grad is not None:
                        grad_norm += param.grad.norm().data[0]
                        grad_max = param.grad.abs().max().data[0]
                        if(grad_max > grad_abs_max):
                            grad_abs_max = grad_max


                print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_sampler),
                    batch_time=batch_time,
                    data_time=data_time, loss=losses))

                str_param = "Param : norm = {:.4f}, max(abs) = {:.7f} " \
                    .format(param_norm, param_abs_max)
                print(str_param)


                str_grad = "Grad : norm = {:.4f}, nax(abs) = {:.7f}" \
                    .format(grad_norm, grad_abs_max)
                print(str_grad)

                param_norm_tensor[epoch] = param_norm
                param_abs_max_tensor[epoch] = param_abs_max
                grad_norm_tensor[epoch] = grad_norm
                grad_abs_max_tensor[epoch] = grad_abs_max

            del loss
            del out

        avg_loss /= len(train_sampler)

        print('Training Summary Epoch: [{0}]\t'
              'Average Loss {loss:.3f}\t'.format(
            epoch + 1, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        total_cer, total_wer = 0, 0
        model.eval()
        #rand_idx_display = random.randint(0, len(test_loader.dataset)-1)
        for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, targets, input_percentages, target_sizes = data

            inputs = Variable(inputs, volatile=True)

            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            if args.gpu >= 0:
                inputs = inputs.cuda()

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH
            seq_length = out.size(0)
            sizes = input_percentages.mul_(int(seq_length)).int()

            decoded_output, _ = decoder.decode(out.data, sizes)
            target_strings = decoder.convert_to_strings(split_targets)
            wer, cer = 0, 0
            for x in range(len(target_strings)):
                decoding, reference = decoded_output[x][0], target_strings[x][0]
                wer_i = decoder.wer(decoding, reference) / float(len(reference.split()))
                cer_i = decoder.cer(decoding, reference) / float(len(reference))
                wer += wer_i
                cer += cer_i
                if(random.uniform(0,1) < args.transcript_prob):
                    print('reference = ' + reference)
                    print('decoding = ' + decoding)
                    print('wer = ' + str(wer_i) + ', cer = ' + str(cer_i))
            total_cer += cer
            total_wer += wer

            if args.gpu >= 0:
                torch.cuda.synchronize()
            del out
        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)
        wer *= 100
        cer *= 100
        loss_results[epoch] = avg_loss
        wer_results[epoch] = wer
        cer_results[epoch] = cer



        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            epoch + 1, wer=wer, cer=cer))


        file_path = '%s/%s_%d.pth.tar' % (save_folder, args.DB_name, args.expnum)  # always overwrite recent epoch's model
        torch.save(myModel.serialize(model, optimizer=optimizer, epoch=epoch, loss_results = loss_results, wer_results=wer_results, cer_results=cer_results,
                                     param_norm=param_norm_tensor, param_max=param_abs_max_tensor, grad_norm = grad_norm_tensor, grad_max = grad_abs_max_tensor),
                       file_path)
        # anneal lr
        optim_state = optimizer.state_dict()
        #optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal # disable learning rate anneal
        optimizer.load_state_dict(optim_state)
        # print('Learning rate annealed to: {lr:.9f}'.format(lr=optim_state['param_groups'][0]['lr'])) # disable learning rate anneal

        if best_wer is None or best_wer > wer:
            print("Found better validated model, saving to %s" % args.model_path)
            torch.save(myModel.serialize(model, optimizer=optimizer, epoch=epoch, loss_results = loss_results,
                                            wer_results=wer_results, cer_results=cer_results)
                       , args.model_path)
            best_wer = wer

        avg_loss = 0
        if not args.sortagrad:
            print("Shuffling batches...")
            train_sampler.shuffle()
