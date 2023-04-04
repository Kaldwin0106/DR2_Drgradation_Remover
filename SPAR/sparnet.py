import numpy as np
from torch import nn, cat
from SPAR.blocks import ConvLayer, ResidualBlock
from SPAR.unet import UNet


class BaseModel(nn.Module):
    """Deep residual network with spatial attention for face SR.
    # Arguments:
        - n_ch: base convolution channels
        - down_steps: how many times to downsample in the encoder
        - res_depth: depth of residual layers in the main body 
        - up_res_depth: depth of residual layers in each upsample block
    # SPARNet:
        - res_depth=1
        - att_name='spar3d'
    """
    def __init__(
        self,
        size=128,
        ch_in=3,
        min_ch=32,
        max_ch=128,
        min_feat_size=16,
        res_depth=10,
        relu_type='leakyrelu',
        norm_type='bn',
        att_name='spar',
        bottleneck_size=4,
    ):
        super(BaseModel, self).__init__()
        self.name = "SPARNet"

        nrargs = {'norm_type': norm_type, 'relu_type': relu_type}

        ch_clip = lambda x: max(min_ch, min(x, max_ch))
        # x = x if x in [32, 128] else 32(<32) or 128(>128)

        down_steps = int(np.log2(size // min_feat_size))
        up_steps = int(np.log2(size // min_feat_size))
        n_ch = ch_clip(max_ch // int(np.log2(size // min_feat_size) + 1))

        # ------------ define encoder --------------------
        self.encoder = []
        self.encoder.append(ConvLayer(ch_in, n_ch, 3, 1))
        hg_depth = int(np.log2(64 / bottleneck_size))
        for i in range(down_steps):
            cin, cout = ch_clip(n_ch), ch_clip(n_ch * 2)
            self.encoder.append(
                ResidualBlock(cin,
                              cout,
                              scale='down',
                              hg_depth=hg_depth,
                              att_name=att_name,
                              **nrargs))

            n_ch = n_ch * 2
            hg_depth = hg_depth - 1
        hg_depth = hg_depth + 1
        self.encoder = nn.Sequential(*self.encoder)

        # ------------ define residual layers --------------------
        self.res_layers = []
        for i in range(res_depth + 3 - down_steps):
            channels = ch_clip(n_ch)
            self.res_layers.append(
                ResidualBlock(channels,
                              channels,
                              hg_depth=hg_depth,
                              att_name=att_name,
                              **nrargs))
        self.res_layers = nn.Sequential(*self.res_layers)

        # ------------ define decoder --------------------
        self.decoder = []
        for i in range(up_steps):
            hg_depth = hg_depth + 1
            cin, cout = ch_clip(n_ch), ch_clip(n_ch // 2)
            self.decoder.append(
                ResidualBlock(cin,
                              cout,
                              scale='up',
                              hg_depth=hg_depth,
                              att_name=att_name,
                              **nrargs))
            n_ch = n_ch // 2

        self.decoder = nn.Sequential(*self.decoder)
        self.out_conv = ConvLayer(ch_clip(n_ch), 3, 3, 1)

    def forward(self, input_img):
        out = self.encoder(input_img)
        out = self.res_layers(out)
        out = self.decoder(out)
        out_img = self.out_conv(out)
        return out_img


class SegModel(BaseModel):
    def __init__(self,
                 ch_in,
                 size=128,
                 min_ch=32,
                 max_ch=128,
                 min_feat_size=16,
                 res_depth=10,
                 relu_type='leakyrelu',
                 norm_type='bn',
                 att_name='spar',
                 bottleneck_size=4):
        super().__init__(size, ch_in, min_ch, max_ch, min_feat_size, res_depth,
                         relu_type, norm_type, att_name, bottleneck_size)

        self.first_block = UNet(16, 10)
    
    def forward(self, input_img, is_first):
        if is_first:
            t = self.first_block(input_img)
            input_img = cat((input_img, t), 1)
        else:
            t = None
        
        return super().forward(input_img)
