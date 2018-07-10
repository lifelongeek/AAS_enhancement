import pdb
import decimal

def check_config_used(config, target_source):
    config_count = dict(vars(config))
    #Initialize each key have zero value
    for k in config_count.keys():
        config_count[k] = 0

    for source in target_source:
        fp = open(source, 'r')
        text = fp.read()
        for k in config_count.keys():
            if(text.find(k) >= 0):
                config_count[k] = 1
        fp.close()

    config_unused = []

    #Check whether unused config exists
    for k in config_count.keys():
        if(config_count[k] == 0):
            config_unused.append(k)

    print('unused config = ')
    print(config_unused)
    assert(len(config_unused) == 0), 'unused config exists, please properly use it or comment it'


def to_np(x):
    return x.data.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_weight_statistic(M):
    print('Model parameter statistic')
    #pdb.set_trace()
    modules = list(M.modules())[0]._modules
    for k,v in modules.items():
        #print(k)
        #print(type(v))
        #pdb.set_trace()
        if(len(v.state_dict())>2):
            for l in range(len(v)):
                layer = v[l]
                #print('layer = '); print(layer); print(' ')
                if(hasattr(layer, 'module')):
                    layer_m = layer.module
                    for j in range(len(layer_m)):
                        sublayer = layer_m[j]
                        if(hasattr(sublayer, 'bias')):
                            #pdb.set_trace()
                            if(sublayer.bias is not None):
                                print(str(sublayer) + ' : weight = ' + str(sublayer.weight.data.min())[:5] + ' ~ ' + str(sublayer.weight.data.max())[:5] + ' (' + str(decimal.Decimal(sublayer.weight.data.mean()))[:7] + ')'
                                ', bias = ' + str(sublayer.bias.data.min())[:5] + ' ~ ' + str(sublayer.bias.data.max())[:5] + ' (' + str(decimal.Decimal(sublayer.bias.data.mean()))[:7] + ')' )
                            else:
                                print(str(sublayer) + ' : weight = ' + str(sublayer.weight.data.min())[:5] + ' ~ ' + str(sublayer.weight.data.max())[:5] + ' (' + str(
                                    decimal.Decimal(sublayer.weight.data.mean()))[:7] + ')')
                        else:
                            print(str(sublayer) + ' : weight = ' + str(sublayer.weight.data.min())[:5] + ' ~ ' + str(sublayer.weight.data.max())[:5] + ' (' + str(decimal.Decimal(sublayer.weight.data.mean()))[:7] + ')' )
                else:
                    if(hasattr(layer, 'weight')):
                        if(hasattr(layer, 'bias')):
                            if(hasattr(layer.bias, 'data')):
                                #pdb.set_trace()
                                print(str(layer) + ' : weight = ' + str(layer.weight.data.min())[:5] + ' ~ ' + str(layer.weight.data.max())[:5] + ' (' + str(decimal.Decimal(layer.weight.data.mean()))[:7] + ')'
                                ', bias = ' + str(layer.bias.data.min())[:5] + ' ~ ' + str(layer.bias.data.max())[:5] + ' (' + str(decimal.Decimal(layer.bias.data.mean()))[:7] + ')' )
                        else:
                            print(str(layer) + ' : weight = ' + str(layer.weight.data.min())[:5] + ' ~ ' + str(layer.weight.data.max())[:5] + ' (' + str(decimal.Decimal(layer.weight.data.mean()))[:7] + ')' )
                    if(hasattr(layer, 'rnn')):
                        #pdb.set_trace()
                        rnn_layer = layer.rnn
                        print(str(rnn_layer))
                        print('\tweight_hh_l0 = ' + str(rnn_layer.weight_hh_l0.data.min())[:5] + ' ~ ' + str(
                            rnn_layer.weight_hh_l0.data.max())[:5] + ' (' + str(decimal.Decimal(rnn_layer.weight_hh_l0.data.mean()))[:7] + ')')
                        print('\tweight_hh_l0_reverse = ' + str(rnn_layer.weight_hh_l0_reverse.data.min())[:5] + ' ~ ' + str(
                            rnn_layer.weight_hh_l0_reverse.data.max())[:5] + ' (' + str(decimal.Decimal(rnn_layer.weight_hh_l0_reverse.data.mean()))[:7] + ')')
                        print('\tweight_ih_l0 = ' + str(rnn_layer.weight_ih_l0.data.min())[:5] + ' ~ ' + str(
                            rnn_layer.weight_ih_l0.data.max())[:5] + ' (' + str(decimal.Decimal(rnn_layer.weight_ih_l0.data.mean()))[:7] + ')')
                        print('\tweight_hh_l0_reverse = ' + str(rnn_layer.weight_hh_l0_reverse.data.min())[:5] + ' ~ ' + str(
                            rnn_layer.weight_hh_l0_reverse.data.max())[:5] + ' (' + str(decimal.Decimal(rnn_layer.weight_hh_l0_reverse.data.mean()))[:7] + ')')

                    if(hasattr(layer, 'batch_norm')):
                        if(layer.batch_norm):
                            bn_layer = layer.batch_norm.module
                            print(str(bn_layer) + ' : weight = ' + str(bn_layer.weight.data.min())[:5] + ' ~ ' + str(bn_layer.weight.data.max())[:5] + ' (' + str(decimal.Decimal(bn_layer.weight.data.mean()))[:7] +
                                  '), bias = ' + str(bn_layer.bias.data.min())[:5] + ' ~ ' + str(bn_layer.bias.data.max())[:5] + ' (' + str(decimal.Decimal(bn_layer.bias.data.mean()))[:7] + ')')



        else:
            if(hasattr(v, 'weight')):
                if(hasattr(v, 'bias')):
                    print(k + ' : weight = ' + str(v.weight.data.min())[:5] + ' ~ ' + str(v.weight.data.max())[:5] + ' (' + str(decimal.Decimal(v.weight.data.mean()))[:7] + ')'
                  ', bias = ' + str(v.bias.data.min())[:5] + ' ~ ' + str(v.bias.data.max())[:5] + ' (' + str(decimal.Decimal(v.bias.data.mean()))[:7] + ')' )
                else:
                    print(k + ' : weight = ' + str(v.weight.data.min())[:5] + ' ~ ' + str(v.weight.data.max())[:5] + ' (' + str(decimal.Decimal(v.weight.data.mean()))[:7] + ')' )
        print(' ')
        print(' ')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('ConvResidualBlock') == -1:
        #m.weight.data.normal_(0.0, 0.01)
        #m.weight.data.normal_(0.0, 0.0001)
        m.weight.data.normal_(0.0, 0.1)
        if hasattr(m, 'bias'):
            if(hasattr(m.bias, 'data')):
                m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Embedding') != -1: # added by ken
        m.weight.data.normal_(0.0, 0.01)
    #elif classname.find('rnn')  # no RNN for now


def _get_variable(inputs, cuda=True):
    if(cuda):
        out = Variable(inputs.cuda())
    else:
        out = Variable(inputs)
    return out


def _get_variable_volatile(inputs, cuda=True):
    if(cuda):
        out = Variable(inputs.cuda(), volatile=True)
    else:
        out = Variable(inputs, volatile=True)
    return out


def _get_variable_nograd(inputs, cuda=True):
    if(cuda):
        out = Variable(inputs.cuda(), requires_grad=False)
    else:
        out = Variable(inputs, requires_grad=False)
    return out