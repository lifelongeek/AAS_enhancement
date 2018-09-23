import os
from glob import glob
from itertools import chain
from shutil import copyfile

import torch
from torch import nn
from tqdm import trange
import decimal
from decoder import GreedyDecoder
import random
import pdb

from utils import _get_variable, _get_variable_volatile, _get_variable_nograd, AverageMeter
from model import *

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Embedding') != -1: # added by ken
        m.weight.data.normal_(0.0, 0.01)

class Trainer(object): # the most basic model
    def __init__(self, config, data_loader=None):
        self.config = config
        self.data_loader = data_loader  # needed for VAE

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.diffLoss = L1Loss_mask() # custom module

        self.valmin_iter = 0
        self.model_dir = 'logs/' + str(config.expnum)
        self.savename_G = ''
        self.savename_D = ''
        self.savename_ASR = ''

        self.kt = 0  # used for Proportional Control Theory in BEGAN, initialized as 0
        self.lb = self.config.lambda_k
        self.gamma = self.config.gamma
        self.conv_measure = 0 # convergence measure

        self.dce_tr = AverageMeter()
        self.dce_tr_local = AverageMeter()
        self.dce_val = AverageMeter()
        self.adv_ny_tr = AverageMeter()
        self.adv_ny_val = AverageMeter()
        self.wer_tr = AverageMeter()
        self.wer_val = AverageMeter()
        self.cer_tr = AverageMeter()
        self.cer_val = AverageMeter()

        self.decoder = GreedyDecoder(data_loader.labels)

        self.build_model()
        self.G.loss_stop = 100000
        #self.get_weight_statistic()

        if self.config.gpu >= 0:
            self.G.cuda()
            self.D.cuda()
            self.diffLoss.cuda()
            self.ASR.cuda()

        if len(self.config.load_path) > 0:
            self.load_model()

        if config.mode == 'train':
            self.logFile = open(self.model_dir + '/log.txt', 'w')

    def zero_grad_all(self):
        self.G.zero_grad()
        self.D.zero_grad()
        self.ASR.zero_grad()


    def build_model(self):
        print('initialize enhancement & discriminator model')
        self.G = stackedBRNN(I=self.config.nFeat_in, O = self.config.nFeat_out, H = self.config.rnn_size, L = self.config.rnn_layers, rnn_type=supported_rnns[self.config.rnn_type])
        self.D = stackedBRNN(I=self.config.nFeat_D, O=self.config.nFeat_out, H = self.config.rnn_size, L = self.config.rnn_layers, rnn_type=supported_rnns[self.config.rnn_type])

        print('load pre-trained ASR model')
        package_ASR = torch.load(self.config.ASR_path, map_location = lambda storage, loc: storage)
        self.ASR = DeepSpeech.load_model_package(package_ASR)

        # Weight initialization is done inside the module

    def load_model(self):
        print("[*] Load models from {}...".format(self.config.load_path))
        postfix = '_valmin'
        paths = glob(os.path.join(self.config.load_path, 'G{}*.pth'.format(postfix)))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.config.load_path))
            assert(0), 'checkpoint not avilable'

        idxes = [int(os.path.basename(path.split('.')[0].split('_')[-1])) for path in paths]
        if self.config.start_iter <= 0 :
            self.config.start_iter = max(idxes)
            if(self.config.start_iter <0): # if still 0, then raise error
                raise Exception("start iter is still less than 0 --> probably try to load initial random model")

        if self.config.gpu < 0:  #CPU
            map_location = lambda storage, loc: storage
        else: # GPU
            map_location = None

        # Ver2
        print('Load models from ' + self.config.load_path + ', ITERATION = ' + str(self.config.start_iter))
        self.G.load_state_dict(torch.load('{}/G{}_{}.pth'.format(self.config.load_path, postfix, self.config.start_iter), map_location=map_location))

        print("[*] Model loaded")

    def train(self):
        # Setting
        optimizer_g = torch.optim.Adam(self.G.parameters(), lr=self.config.lr, betas=(self.beta1, self.beta2), amsgrad = True)
        optimizer_d = torch.optim.Adam(self.D.parameters(), lr=self.config.lr, betas=(self.beta1, self.beta2), amsgrad = True)

        for iter in trange(self.config.start_iter, self.config.max_iter):
            self.zero_grad_all()

            # Train
            # Noisy data
            data_list = self.data_loader.next(cl_ny = 'ny', type = 'train')
            #inputs, targets, input_percentages, target_sizes, mask = \
#                _get_variable_nograd(data_list[0]), _get_variable_nograd(data_list[1], cuda=False), data_list[2], _get_variable_nograd(data_list[3], cuda=False), _get_variable_nograd(data_list[4])
            mixture, cleans, mask = \
                            _get_variable_nograd(data_list[0]), _get_variable_nograd(data_list[1]), _get_variable_nograd(data_list[2])

            # forward generator
            enhanced = self.G(mixture)
            enhanced_D = enhanced.detach()

            # adversarial training: G-step
            ae_ny_G = self.D.forward_paired(enhanced, mixture)
            l_adv_ny_G, _ = self.diffLoss(ae_ny_G, enhanced, mask) # normalized inside function
            l_adv_ny_G = l_adv_ny_G*self.config.w_adversarial
            l_adv_ny_G_data = l_adv_ny_G.data[0]
            l_adv_ny_G.backward(retain_graph=True)
            g_adv = self.get_gradient_norm(self.G)
            self.D.zero_grad() # this makes no gradient for discriminator
            del l_adv_ny_G

            # adversarial training: D-step
            ae_ny_D = self.D.forward_paired(enhanced_D, mixture)
            l_adv_ny_D, _ = self.diffLoss(ae_ny_D, enhanced_D, mask) # normalized inside function
            l_adv_ny_D = l_adv_ny_D*(-self.kt)*self.config.w_adversarial
            l_adv_ny_D.backward()
            del l_adv_ny_D

            # DCE loss
            dce, nElement = self.diffLoss(enhanced, cleans, mask) # already normalized inside function
            dce_loss = dce.data[0]
            dce_tr_local.update(dce_loss, nElement)


            # Clean data
            ae_cl = self.D(cleans, mixture)
            l_adv_cl, _ = self.diffLoss(ae_cl, cleans, mask)  # normalized inside function
            l_adv_cl = self.config.w_adversarial*l_adv_cl
            l_adv_cl.backward()
            l_adv_cl_data = l_adv_cl.data[0]
            del l_adv_cl

            # update
            optimizer_g.step()
            optimizer_d.step()

            # Proportional Control Theory
            g_d_balance = self.gamma * l_adv_cl_data - l_adv_ny_G_data
            self.kt += self.lb * g_d_balance
            self.kt = max(min(1, self.kt), 0)
            conv_measure = l_adv_cl_data + abs(g_d_balance)

            # log
            #pdb.set_trace()
            if (iter+1) % self.config.log_iter == 0:
                str_loss= "[{}/{}] (train) DCE: {:.7f}, ADV_cl: {:.7f}, ADV_ny: {:.7f}".format(iter, self.config.max_iter, self.dce_tr_local.avg, l_adv_cl_data, l_adv_ny_G_data)
                print(str_loss)
                self.logFile.write(str_loss + '\n')

                str_loss= "[{}/{}] (train) conv_measure: {:.4f}, kt: {:.4f} ".format(iter, self.config.max_iter, conv_measure, self.kt)
                print(str_loss)
                self.logFile.write(str_loss + '\n')

                self.logFile.flush()
                self.dce_tr_local.reset()


            if (iter+1) % self.config.save_iter == 0:
                self.G.eval()

                # Measure performance on training subset
                self.dce_tr.reset()
                self.adv_ny_tr.reset()
                self.wer_tr.reset()
                self.cer_tr.reset()
                for _ in trange(0, len(self.data_loader.trsub_dl)):
                    data_list = self.data_loader.next(cl_ny='ny', type='trsub')
                    mixture, cleans, mask, targets, input_percentages, target_sizes = \
                        data_list[0], data_list[1], _get_variable_volatile(data_list[2]), data_list[3], data_list[4], data_list[5]
                    dce, adv_ny, nElement, wer, cer, nWord, nChar = self.greedy_decoding_and_FSEGAN(mixture, cleans, targets, input_percentages, target_sizes, mask)

                    self.dce_tr.update(dce.data[0], nElement)
                    self.adv_ny_tr.update(adv_ny.data[0], nElement)
                    self.wer_tr.update(wer, nWord)
                    self.cer_tr.update(cer, nChar)

                    del dce, adv_ny

                str_loss= "[{}/{}] (training subset) CTC: {:.7f}, WER: {:.7f}, CER: {:.7f}".format(iter, self.config.max_iter, self.dce_tr.avg, self.wer_tr.avg*100, self.cer_tr.avg*100)
                print(str_loss)
                self.logFile.write(str_loss + '\n')

                # Measure performance on validation data
                self.dce_val.reset()
                self.adv_ny_val.reset()
                self.wer_val.reset()
                self.cer_val.reset()
                for _ in trange(0, len(self.data_loader.val_dl)):
                    data_list = self.data_loader.next(cl_ny='ny', type='val')
                    mixture, cleans, mask, targets, input_percentages, target_sizes = \
                        data_list[0], data_list[1], _get_variable_volatile(data_list[2]), data_list[3], data_list[4], data_list[5]
                    dce, adv_ny, nElement, wer, cer, nWord, nChar = self.greedy_decoding_and_FSEGAN(mixture, cleans, targets, input_percentages, target_sizes, mask)

                    self.dce_val.update(dce.data[0], nElement)
                    self.adv_ny_val.update(adv_ny.data[0], nElement)
                    self.wer_val.update(wer, nWord)
                    self.cer_val.update(cer, nChar)

                    del ctc, adv_ny

                str_loss = "[{}/{}] (validation) CTC: {:.7f}, WER: {:.7f}, CER: {:.7f}".format(iter, self.config.max_iter, self.dce_val.avg, self.wer_val.avg*100, self.cer_val.avg*100)
                print(str_loss)
                self.logFile.write(str_loss + '\n')
                self.logFile.flush()

                self.G.train() # end of validation

                # Save model
                if (len(self.savename_G) > 0): # do not remove here
                    if os.path.exists(self.savename_G):
                        os.remove(self.savename_G) # remove previous model
                self.savename_G = '{}/G_{}.pth'.format(self.model_dir, iter)
                torch.save(self.G.state_dict(), self.savename_G)

                if(self.G.loss_stop > self.wer_val.avg):
                    self.G.loss_stop = self.wer_val.avg
                    savename_G_valmin_prev = '{}/G_valmin_{}.pth'.format(self.model_dir, self.valmin_iter)
                    if os.path.exists(savename_G_valmin_prev):
                        os.remove(savename_G_valmin_prev) # remove previous model

                    print('save model for this checkpoint')
                    savename_G_valmin = '{}/G_valmin_{}.pth'.format(self.model_dir, iter)
                    copyfile(self.savename_G, savename_G_valmin)

                    self.valmin_iter = iter



    def greedy_decoding_and_FSEGAN(self, mixture, cleans, targets, input_percentages, target_sizes, mask, transcript_prob=0.001):
        mixture = _get_variable_volatile(mixture)
        N = inputs.size(0)
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        # step 1) Decoding to get wer & cer
        enhanced = self.G(mixture)
        prob = self.ASR(enhanced)
        prob = prob.transpose(0,1)
        T = prob.size(0)
        sizes = input_percentages.mul_(int(T)).int()

        decoded_output, _ = self.decoder.decode(prob.data, sizes)
        target_strings = self.decoder.convert_to_strings(split_targets)
        we, ce, total_word, total_char = 0, 0, 0, 0

        for x in range(len(target_strings)):
            decoding, reference = decoded_output[x][0], target_strings[x][0]
            nChar = len(reference)
            nWord = len(reference.split())
            we_i = self.decoder.wer(decoding, reference)
            ce_i = self.decoder.cer(decoding, reference)
            we += we_i
            ce += ce_i
            total_word += nWord
            total_char += nChar
            if (random.uniform(0, 1) < transcript_prob):
                print('reference = ' + reference)
                print('decoding = ' + decoding)
                print('wer = ' + str(we_i/float(nWord)) + ', cer = ' + str(ce_i/float(nChar)))

        wer = we/total_word
        cer = ce/total_word

        # step 2) get adversarial loss (for noisy data only)
        ae_ny = self.D.forward_paired(enhanced, mixture)
        l_adv_ny, nElement = self.diffLoss(ae_ny, enhanced, mask) # normalized inside function
        l_adv_ny = l_adv_ny*self.config.w_adversarial

        # step 3) get DCE loss
        dce, nElement_ = self.diffLoss(enhanced, cleans, mask)  # already normalized inside function
        assert(nElement == nElement_)

        return dce, l_adv_ny, nElement, wer, cer, total_word, total_char