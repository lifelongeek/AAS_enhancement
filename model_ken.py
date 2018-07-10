import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import pdb

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            #batch_size = input_.size()[0]
            #return torch.stack([F.softmax(input_[i], dim=1) for i in range(batch_size)], 0)
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        # should we handle batch_first=True?
        super(Lookahead, self).__init__()
        self.n_features = n_features
        self.weight = Parameter(torch.Tensor(n_features, context + 1))
        assert context > 0
        self.context = context
        self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):  # what's a better way initialiase this layer?
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        seq_len = input.size(0)
        # pad the 0th dimension (T/sequence) with zeroes whose number = context
        # Once pytorch's padding functions have settled, should move to those.
        padding = torch.zeros(self.context, *(input.size()[1:])).type_as(input.data)
        x = torch.cat((input, Variable(padding)), 0)

        # add lookahead windows (with context+1 width) as a fourth dimension
        # for each seq-batch-feature combination
        x = [x[i:i + self.context + 1] for i in range(seq_len)]  # TxLxNxH - sequence, context, batch, feature
        x = torch.stack(x)
        x = x.permute(0, 2, 3, 1)  # TxNxHxL - sequence, batch, feature, context

        x = torch.mul(x, self.weight).sum(dim=3)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech_ken(nn.Module):
    def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=512, rnn_layers=2, bidirectional=True,
                 kernel_sz=11, stride=2, map=256, cnn_layers=2,
                 nFreq=40, nDownsample=1, audio_conf = None):
        super(DeepSpeech_ken, self).__init__()

        # model metadata needed for serialization/deserialization
        self.nFreq = nFreq

        self._version = '0.0.1'

        self._audio_conf = audio_conf # not used

        # RNN
        self.rnn_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        # CNN
        self.cnn_stride = stride   # use stride for subsampling
        self.cnn_map = map
        self.cnn_kernel = kernel_sz
        self.nDownsample = nDownsample

        self.cnn_layers = cnn_layers

        self._labels = labels


        num_classes = len(self._labels)

        conv_list = []
        conv_list.append(nn.Conv1d(nFreq, map, kernel_size=kernel_sz, stride=stride))
        conv_list.append(nn.BatchNorm1d(map))
        conv_list.append(nn.LeakyReLU(map, inplace=True))

        if(self.nDownsample == 1):
            stride=1

        for x in range(self.cnn_layers - 1):
            conv_list.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride))
            conv_list.append(nn.BatchNorm1d(map))
            conv_list.append(nn.LeakyReLU(map, inplace=True))

        self.conv = nn.Sequential(*conv_list)
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        # how to calculate like this?
        # Ver1
        """
        rnn_input_size = int(math.floor((sample_rate * window_size - kernel_sz) / stride) + 1)
        rnn_input_size = int(math.floor(rnn_input_size - kernel_sz) / stride + 1)
        rnn_input_size = int(math.floor(rnn_input_size - kernel_sz) / stride + 1)
        rnn_input_size *= map
        """

        #Ver2
        rnn_input_size = map

        print('rnn input size = ' + str(rnn_input_size))

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(self.rnn_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()


    def forward(self, x):
        if(x.dim() == 4):
            #pdb.set_trace()
            #if(x.size(1) > 1):
#                x = x.mean(1)
            x = x.squeeze()


        #pdb.set_trace()
        x = self.conv(x)

        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        x = self.rnns(x)

        x = self.fc(x)
        x = x.transpose(0, 1)

        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x

    @classmethod
    def load_model(cls, path, gpu=-1):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        #pdb.set_trace()
        model = cls(rnn_hidden_size=package['rnn_size'], rnn_layers=package['rnn_layers'], rnn_type=supported_rnns[package['rnn_type']],
                    map=package['cnn_map'], stride = package['cnn_stride'], kernel_sz=package['cnn_kernel'], cnn_layers=package['cnn_layers'],
                    labels=package['labels']
                    )
        # the blacklist parameters are params that were previous erroneously saved by the model
        # care should be taken in future versions that if batch_norm on the first rnn is required
        # that it be named something else
        blacklist = ['rnns.0.batch_norm.module.weight', 'rnns.0.batch_norm.module.bias',
                     'rnns.0.batch_norm.module.running_mean', 'rnns.0.batch_norm.module.running_var']
        for x in blacklist:
            if x in package['state_dict']:
                del package['state_dict'][x]
        model.load_state_dict(package['state_dict'])
        for x in model.rnns:
            x.flatten_parameters()

        if gpu>=0:
            model = model.cuda()
        #if cuda:
#            model = torch.nn.DataParallel(model).cuda()
        return model

    @classmethod
    def load_model_package(cls, package, gpu=-1):
        model = cls(rnn_hidden_size=package['rnn_size'], rnn_layers=package['rnn_layers'],rnn_type=supported_rnns[package['rnn_type']],
                    map=package['cnn_map'], stride = package['cnn_stride'], kernel_sz=package['cnn_kernel'], cnn_layers=package['cnn_layers'],
                    labels=package['labels'],
                    )
        model.load_state_dict(package['state_dict'])
        if(gpu>=0):
            model = model.cuda()
        #if cuda:
#            model = torch.nn.DataParallel(model).cuda()
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None, meta=None):
        #model_is_cuda = next(model.parameters()).is_cuda
        #pdb.set_trace()
        #model = model.module if model_is_cuda else model
        #model = model._modules if model_is_cuda else model

        package = {
            'version': model._version,
            'rnn_size': model.rnn_size,
            'rnn_layers': model.rnn_layers,
            'cnn_map': model.cnn_map,
            'cnn_kernel': model.cnn_kernel,
            'cnn_stride': model.cnn_stride,
            'cnn_layers': model.cnn_layers,
            'rnn_type': supported_rnns_inv.get(model.rnn_type, model.rnn_type.__name__.lower()),
            'labels': model._labels,
            'state_dict': model.state_dict()
       }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        """
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._labels if model_is_cuda else model._labels
        """
        return model._labels

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params


    @staticmethod
    def get_audio_conf(model):
        return model._audio_conf


    @staticmethod
    def get_meta(model):
        model_is_cuda = next(model.parameters()).is_cuda
        m = model.module if model_is_cuda else model
        meta = {
            "version": m._version,
            "rnn_size": m.rnn_size,
            "rnn_layers": m.rnn_layers,
            "cnn_map": m.cnn_map,
            "cnn_kernel": m.cnn_kernel,
            "cnn_stride": m.cnn_stride,
            "cnn_layers": m.cnn_layers,
            "rnn_type": supported_rnns_inv[m.rnn_type]
        }
        return meta



class DeepSpeech_ken(nn.Module):
    def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=512, rnn_layers=2, bidirectional=True,
                 kernel_sz=11, stride=2, map=256, cnn_layers=2,
                 nFreq=40, nDownsample=1, audio_conf = None,
                 include_first_BN=True):
        super(DeepSpeech_ken, self).__init__()

        # model metadata needed for serialization/deserialization
        self.nFreq = nFreq

        self._version = '0.0.1'

        self._audio_conf = audio_conf # not used

        # RNN
        self.rnn_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        # CNN
        self.cnn_stride = stride   # use stride for subsampling
        self.cnn_map = map
        self.cnn_kernel = kernel_sz
        self.nDownsample = nDownsample

        self.cnn_layers = cnn_layers

        self._labels = labels


        num_classes = len(self._labels)

        conv_list = []
        conv_list.append(nn.Conv1d(nFreq, map, kernel_size=kernel_sz, stride=stride))
        if(include_first_BN):
            conv_list.append(nn.BatchNorm1d(map))
        conv_list.append(nn.LeakyReLU(map, inplace=True))

        if(self.nDownsample == 1):
            stride=1

        for x in range(self.cnn_layers - 1):
            #conv_list.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride, bias=False))
            conv_list.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride))
            conv_list.append(nn.BatchNorm1d(map))
            conv_list.append(nn.LeakyReLU(map, inplace=True))

        self.conv = nn.Sequential(*conv_list)
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        # how to calculate like this?
        # Ver1
        """
        rnn_input_size = int(math.floor((sample_rate * window_size - kernel_sz) / stride) + 1)
        rnn_input_size = int(math.floor(rnn_input_size - kernel_sz) / stride + 1)
        rnn_input_size = int(math.floor(rnn_input_size - kernel_sz) / stride + 1)
        rnn_input_size *= map
        """

        #Ver2
        rnn_input_size = map

        print('rnn input size = ' + str(rnn_input_size))

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(self.rnn_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
            #nn.Linear(rnn_hidden_size, num_classes)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()


    def forward(self, x):
        if(x.dim() == 4):
            x = x.squeeze()

        #pdb.set_trace()
        x = self.conv(x)

        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        x = self.rnns(x)

        x = self.fc(x)
        x = x.transpose(0, 1)

        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x

    @classmethod
    def load_model(cls, path, gpu=-1):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        #pdb.set_trace()
        model = cls(rnn_hidden_size=package['rnn_size'], rnn_layers=package['rnn_layers'], rnn_type=supported_rnns[package['rnn_type']],
                    map=package['cnn_map'], stride = package['cnn_stride'], kernel_sz=package['cnn_kernel'], cnn_layers=package['cnn_layers'],
                    labels=package['labels']
                    )
        # the blacklist parameters are params that were previous erroneously saved by the model
        # care should be taken in future versions that if batch_norm on the first rnn is required
        # that it be named something else
        blacklist = ['rnns.0.batch_norm.module.weight', 'rnns.0.batch_norm.module.bias',
                     'rnns.0.batch_norm.module.running_mean', 'rnns.0.batch_norm.module.running_var']
        for x in blacklist:
            if x in package['state_dict']:
                del package['state_dict'][x]
        model.load_state_dict(package['state_dict'])
        for x in model.rnns:
            x.flatten_parameters()

        if gpu>=0:
            model = model.cuda()
        #if cuda:
#            model = torch.nn.DataParallel(model).cuda()
        return model

    @classmethod
    def load_model_package(cls, package, gpu=-1):
        model = cls(rnn_hidden_size=package['rnn_size'], rnn_layers=package['rnn_layers'],rnn_type=supported_rnns[package['rnn_type']],
                    map=package['cnn_map'], stride = package['cnn_stride'], kernel_sz=package['cnn_kernel'], cnn_layers=package['cnn_layers'],
                    labels=package['labels'],
                    )
        model.load_state_dict(package['state_dict'])
        if(gpu>=0):
            model = model.cuda()
        #if cuda:
#            model = torch.nn.DataParallel(model).cuda()
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None,
                  param_norm=None, param_max=None, grad_norm=None, grad_max=None,
                  meta=None):
        #model_is_cuda = next(model.parameters()).is_cuda
        #pdb.set_trace()
        #model = model.module if model_is_cuda else model
        #model = model._modules if model_is_cuda else model

        package = {
            'version': model._version,
            'rnn_size': model.rnn_size,
            'rnn_layers': model.rnn_layers,
            'cnn_map': model.cnn_map,
            'cnn_kernel': model.cnn_kernel,
            'cnn_stride': model.cnn_stride,
            'cnn_layers': model.cnn_layers,
            'rnn_type': supported_rnns_inv.get(model.rnn_type, model.rnn_type.__name__.lower()),
            'labels': model._labels,
            'state_dict': model.state_dict()
       }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results

            package['param_norm'] = param_norm
            package['param_max'] = param_max
            package['grad_norm'] = grad_norm
            package['grad_max'] = grad_max
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        """
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._labels if model_is_cuda else model._labels
        """
        return model._labels

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params


    @staticmethod
    def get_audio_conf(model):
        return model._audio_conf


    @staticmethod
    def get_meta(model):
        model_is_cuda = next(model.parameters()).is_cuda
        m = model.module if model_is_cuda else model
        meta = {
            "version": m._version,
            "rnn_size": m.rnn_size,
            "rnn_layers": m.rnn_layers,
            "cnn_map": m.cnn_map,
            "cnn_kernel": m.cnn_kernel,
            "cnn_stride": m.cnn_stride,
            "cnn_layers": m.cnn_layers,
            "rnn_type": supported_rnns_inv[m.rnn_type]
        }
        return meta

#class DeepSpeech(nn.Module): # is removed from this source

class ResidualDeepSpeech(nn.Module):
    def __init__(self, rnn_type=nn.GRU, labels="abc", rnn_hidden_size=512, rnn_layers=2, bidirectional=True,
                 kernel_sz=11, stride=2, map=256, cnn_residual_blocks=1,
                 nFreq=40, nDownsample=1, audio_conf = None):
        super(ResidualDeepSpeech, self).__init__()

        # model metadata needed for serialization/deserialization
        self.nFreq = nFreq

        self._version = '0.0.1'

        self._audio_conf = audio_conf # not used

        # RNN
        self.rnn_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        # CNN
        self.cnn_stride = stride   # use stride for subsampling
        self.cnn_map = map
        self.cnn_kernel = kernel_sz
        self.nDownsample = nDownsample

        self.cnn_residual_blocks = cnn_residual_blocks

        self._labels = labels


        num_classes = len(self._labels)

        conv1 = []
        conv1.append(nn.Conv1d(nFreq, map, kernel_size=kernel_sz, stride=stride))
        #conv1.append(nn.BatchNorm1d(map))
        conv1.append(nn.LeakyReLU(map, inplace=True))
        self.conv1 = nn.Sequential(*conv1)

        if(self.nDownsample == 1):
            stride = 1

        #for x in range(self.cnn_layers - 1):
        padding = int((kernel_sz-1)/2)
        assert(stride==1), 'padding is only valid when stride=1'

        residual2 = []
        residual2.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride, padding = padding))
        residual2.append(nn.BatchNorm1d(map))
        residual2.append(nn.LeakyReLU(map, inplace=True))
        residual2.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride, padding = padding))
        residual2.append(nn.BatchNorm1d(map))
        self.residual2 = nn.Sequential(*residual2)




        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        # how to calculate like this?
        # Ver1
        """
        rnn_input_size = int(math.floor((sample_rate * window_size - kernel_sz) / stride) + 1)
        rnn_input_size = int(math.floor(rnn_input_size - kernel_sz) / stride + 1)
        rnn_input_size = int(math.floor(rnn_input_size - kernel_sz) / stride + 1)
        rnn_input_size *= map
        """

        #Ver2
        #rnn_input_size = map

        #Ver3
        #layer for matching dimension between CNN & RNN (their #feature map is different)
        self.dim_match_layer = nn.Conv1d(map, rnn_hidden_size, kernel_size=1, stride=1, padding=0)
        rnn_input_size = rnn_hidden_size

        print('rnn input size = ' + str(rnn_input_size))

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                       bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(self.rnn_layers - 1):
            rnn = BatchRNN(input_size=rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type,
                           bidirectional=bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()


    def forward(self, x):
        if(x.dim() == 4):
            x = x.squeeze()

        x = self.conv1(x)

        x = self.residual2(x) + x

        # set cnn & rnn dimension same (their #feature map is different)
        x = self.dim_match_layer(x)

        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # (NxHxT) --> (TxNxH)


        for l in range(self.rnn_layers):
            #pdb.set_trace()
            x = self.rnns[l](x) + x  # is it valid?

        x = self.fc(x) # (TxNxH) --> ()
        x = x.transpose(0, 1)

        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x

    @classmethod
    def load_model(cls, path, gpu=-1):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        #pdb.set_trace()
        model = cls(rnn_hidden_size=package['rnn_size'], rnn_layers=package['rnn_layers'], rnn_type=supported_rnns[package['rnn_type']],
                    map=package['cnn_map'], stride = package['cnn_stride'], kernel_sz=package['cnn_kernel'], cnn_residual_blocks=package['cnn_residual_blocks'],
                    labels=package['labels']
                    )
        # the blacklist parameters are params that were previous erroneously saved by the model
        # care should be taken in future versions that if batch_norm on the first rnn is required
        # that it be named something else
        blacklist = ['rnns.0.batch_norm.module.weight', 'rnns.0.batch_norm.module.bias',
                     'rnns.0.batch_norm.module.running_mean', 'rnns.0.batch_norm.module.running_var']
        for x in blacklist:
            if x in package['state_dict']:
                del package['state_dict'][x]
        model.load_state_dict(package['state_dict'])
        for x in model.rnns:
            x.flatten_parameters()

        if gpu>=0:
            model = model.cuda()
        #if cuda:
#            model = torch.nn.DataParallel(model).cuda()
        return model

    @classmethod
    def load_model_package(cls, package, gpu=-1):
        model = cls(rnn_hidden_size=package['rnn_size'], rnn_layers=package['rnn_layers'],rnn_type=supported_rnns[package['rnn_type']],
                    map=package['cnn_map'], stride = package['cnn_stride'], kernel_sz=package['cnn_kernel'], cnn_residual_blocks=package['cnn_residual_blocks'],
                    labels=package['labels'],
                    )
        model.load_state_dict(package['state_dict'])
        if(gpu>=0):
            model = model.cuda()
        #if cuda:
#            model = torch.nn.DataParallel(model).cuda()
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None, meta=None):
        #model_is_cuda = next(model.parameters()).is_cuda
        #pdb.set_trace()
        #model = model.module if model_is_cuda else model
        #model = model._modules if model_is_cuda else model

        package = {
            'version': model._version,
            'rnn_size': model.rnn_size,
            'rnn_layers': model.rnn_layers,
            'cnn_map': model.cnn_map,
            'cnn_kernel': model.cnn_kernel,
            'cnn_stride': model.cnn_stride,
            'cnn_residual_blocks': model.cnn_residual_blocks,
            'rnn_type': supported_rnns_inv.get(model.rnn_type, model.rnn_type.__name__.lower()),
            'labels': model._labels,
            'state_dict': model.state_dict()
       }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        """
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._labels if model_is_cuda else model._labels
        """
        return model._labels

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params


    @staticmethod
    def get_audio_conf(model):
        return model._audio_conf


    @staticmethod
    def get_meta(model):
        model_is_cuda = next(model.parameters()).is_cuda
        m = model.module if model_is_cuda else model
        meta = {
            "version": m._version,
            "rnn_size": m.rnn_size,
            "rnn_layers": m.rnn_layers,
            "cnn_map": m.cnn_map,
            "cnn_kernel": m.cnn_kernel,
            "cnn_stride": m.cnn_stride,
            "cnn_layers": m.cnn_layers,
            "rnn_type": supported_rnns_inv[m.rnn_type]
        }
        return meta


class ResidualCNN4block(nn.Module):
    def __init__(self, labels="abc",
                 kernel_sz=11, stride=2, map=512,
                 nFreq=40, nDownsample=1, audio_conf = None):
        super(ResidualCNN4block, self).__init__()

        # model metadata needed for serialization/deserialization
        self.nFreq = nFreq

        self._version = '0.0.1'

        self._audio_conf = audio_conf # not used

        # CNN
        self.cnn_stride = stride   # use stride for subsampling
        self.cnn_map = map
        self.cnn_kernel = kernel_sz
        self.nDownsample = nDownsample

        self._labels = labels


        num_classes = len(self._labels)

        conv1 = []
        conv1.append(nn.Conv1d(nFreq, map, kernel_size=kernel_sz, stride=stride, bias=False))
        #conv1.append(nn.BatchNorm1d(map))
        conv1.append(nn.LeakyReLU(map, inplace=True))
        self.conv1 = nn.Sequential(*conv1)

        if(self.nDownsample == 1):
            stride = 1

        #for x in range(self.cnn_layers - 1):
        padding = int((kernel_sz-1)/2)
        assert(stride==1), 'padding is only valid when stride=1'

        residual2 = []
        residual2.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride, padding = padding, bias=False))
        residual2.append(nn.BatchNorm1d(map))
        residual2.append(nn.LeakyReLU(map, inplace=True))
        residual2.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride, padding = padding, bias=False))
        residual2.append(nn.BatchNorm1d(map))
        self.residual2 = nn.Sequential(*residual2)

        residual3 = []
        residual3.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride, padding = padding, bias=False))
        residual3.append(nn.BatchNorm1d(map))
        residual3.append(nn.LeakyReLU(map, inplace=True))
        residual3.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride, padding = padding, bias=False))
        residual3.append(nn.BatchNorm1d(map))
        self.residual3 = nn.Sequential(*residual3)

        residual4 = []
        residual4.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride, padding = padding, bias=False))
        residual4.append(nn.BatchNorm1d(map))
        residual4.append(nn.LeakyReLU(map, inplace=True))
        residual4.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride, padding = padding, bias=False))
        residual4.append(nn.BatchNorm1d(map))
        self.residual4 = nn.Sequential(*residual4)

        residual5 = []
        residual5.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride, padding = padding, bias=False))
        residual5.append(nn.BatchNorm1d(map))
        residual5.append(nn.LeakyReLU(map, inplace=True))
        residual5.append(nn.Conv1d(map, map, kernel_size=kernel_sz, stride=stride, padding = padding, bias=False))
        residual5.append(nn.BatchNorm1d(map))
        self.residual5 = nn.Sequential(*residual5)

        fully_connected = nn.Sequential(
            nn.Linear(map, num_classes)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()


    def forward(self, x):
        if(x.dim() == 4):
            x = x.squeeze()

        #pdb.set_trace()
        x = self.conv1(x)
        x = self.residual2(x) + x
        x = self.residual3(x) + x
        x = self.residual4(x) + x
        x = self.residual5(x) + x

        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # (NxHxT) --> (TxNxH)

        x = self.fc(x) # (TxNxH) --> ()
        x = x.transpose(0, 1)

        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x

    @classmethod
    def load_model(cls, path, gpu=-1):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        #pdb.set_trace()
        model = cls(map=package['cnn_map'], stride = package['cnn_stride'], kernel_sz=package['cnn_kernel'],
                    labels=package['labels']
                    )
        # the blacklist parameters are params that were previous erroneously saved by the model
        # care should be taken in future versions that if batch_norm on the first rnn is required
        # that it be named something else
        model.load_state_dict(package['state_dict'])
        for x in model.rnns:
            x.flatten_parameters()

        if gpu>=0:
            model = model.cuda()
        #if cuda:
#            model = torch.nn.DataParallel(model).cuda()
        return model

    @classmethod
    def load_model_package(cls, package, gpu=-1):
        model = cls(
                    map=package['cnn_map'], stride = package['cnn_stride'], kernel_sz=package['cnn_kernel'],
                    labels=package['labels'],
                    )
        model.load_state_dict(package['state_dict'])
        if(gpu>=0):
            model = model.cuda()
        #if cuda:
#            model = torch.nn.DataParallel(model).cuda()
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None, meta=None):
        #model_is_cuda = next(model.parameters()).is_cuda
        #pdb.set_trace()
        #model = model.module if model_is_cuda else model
        #model = model._modules if model_is_cuda else model

        package = {
            'version': model._version,
            'cnn_map': model.cnn_map,
            'cnn_kernel': model.cnn_kernel,
            'cnn_stride': model.cnn_stride,
            'labels': model._labels,
            'state_dict': model.state_dict()
       }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        """
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._labels if model_is_cuda else model._labels
        """
        return model._labels

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params


    @staticmethod
    def get_audio_conf(model):
        return model._audio_conf


    @staticmethod
    def get_meta(model):
        model_is_cuda = next(model.parameters()).is_cuda
        m = model.module if model_is_cuda else model
        meta = {
            "version": m._version,
            "cnn_map": m.cnn_map,
            "cnn_kernel": m.cnn_kernel,
            "cnn_stride": m.cnn_stride,
            "cnn_layers": m.cnn_layers,
            }
        return meta

if __name__ == '__main__':
    import os.path
    import argparse

    parser = argparse.ArgumentParser(description='DeepSpeech model information')
    parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                        help='Path to model file created by training')
    args = parser.parse_args()
    package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model = DeepSpeech_ken.load_model(args.model_path)

    print("Model name:         ", os.path.basename(args.model_path))
    print("DeepSpeech version: ", model._version)
    print("")
    print("Recurrent Neural Network Properties")
    print("  RNN Type:         ", model._rnn_type.__name__.lower())
    print("  RNN Layers:       ", model._hidden_layers)
    print("  RNN Size:         ", model._hidden_size)
    print("  Classes:          ", len(model._labels))
    print("")
    print("Model Features")
    print("  Labels:           ", model._labels)
    print("  Sample Rate:      ", model._audio_conf.get("sample_rate", "n/a"))
    print("  Window Type:      ", model._audio_conf.get("window", "n/a"))
    print("  Window Size:      ", model._audio_conf.get("window_size", "n/a"))
    print("  Window Stride:    ", model._audio_conf.get("window_stride", "n/a"))

    if package.get('loss_results', None) is not None:
        print("")
        print("Training Information")
        epochs = package['epoch']
        print("  Epochs:           ", epochs)
        print("  Current Loss:      {0:.3f}".format(package['loss_results'][epochs - 1]))
        print("  Current CER:       {0:.3f}".format(package['cer_results'][epochs - 1]))
        print("  Current WER:       {0:.3f}".format(package['wer_results'][epochs - 1]))

    if package.get('meta', None) is not None:
        print("")
        print("Additional Metadata")
        for k, v in model._meta:
            print("  ", k, ": ", v)
