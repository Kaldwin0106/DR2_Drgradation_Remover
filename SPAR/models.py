'''
Using SPARNet from https://github.com/chaofengc/Face-SPARNet as enhancement module.
This can be swaped to any other blind face restoration method.
'''


import torch.nn as nn
from torch.nn import init
import torch.nn.utils as tutils
from SPAR.sparnet import BaseModel


def apply_norm(net, weight_norm_type):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if weight_norm_type.lower() == 'spectral_norm':
                tutils.spectral_norm(m)
            elif weight_norm_type.lower() == 'weight_norm':
                tutils.weight_norm(m)
            else:
                pass


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d'
        ) != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    if isinstance(net, nn.DataParallel):
        network_name = net.module.__class__.__name__
    else:
        network_name = net.__class__.__name__

    # print('initialize network %s with %s' % (network_name, init_type))
    net.apply(init_func)  # apply the initialization function <init_func>


def init_network(net, use_norm='none'):
    apply_norm(net, use_norm)
    init_weights(net, init_type='normal', init_gain=0.02)
    return net


def createModel(model_name):
    if model_name == "SPARNet":
        '''
        Upsample a 16x16 bicubic downsampled face image to 128x128,
        and there is no need to align the LR face.
        '''
        model = BaseModel(ch_in=3, size=128)
    elif model_name == "SPARNetLight":
        '''
        Using 3D attention to replace 2D attention in SPARNet
        and reducing the number of residual layers to 1.
        '''
        model = BaseModel(res_depth=1, att_name="spar3d")
        model.name = "SPARNetLight"
    elif model_name == "SPARNetHD2D":
        '''
        Enhancing a pre-aligned lo quality face image with 2D attention.
        '''
        model = BaseModel(min_ch=32,
                          max_ch=512,
                          size=512,
                          min_feat_size=32,
                          att_name='spar',
                          norm_type='in')
        model = init_network(model, use_norm='spectral_norm')
        model.name = "SPARNetHD2D"
    elif model_name == "SPARNetHD3D":
        '''
        Enhancing a pre-aligned lo quality face image with 3D attention.
        '''
        model = BaseModel(min_ch=32,
                          max_ch=512,
                          size=512,
                          min_feat_size=32,
                          att_name='spar3d',
                          norm_type='in')
        model = init_network(model, use_norm='spectral_norm')
        model.name = "SPARNetHD3D"
    else:
        model = None
        raise NotImplementedError("Unknown model name: %s" % (model_name))
    return model

