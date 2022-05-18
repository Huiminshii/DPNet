import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from mmcv.cnn import ConvModule, constant_init, normal_init,kaiming_init,build_norm_layer,build_conv_layer, DepthwiseSeparableConvModule
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import load_checkpoint,BaseModule
import math
from ..builder import BACKBONES
#from .base_backbone import BaseBackbone
from mmdet.utils import get_root_logger
import math

        
class LSAM_Pyramid2(BaseModule):
    def __init__(self, input_size,channel=128,channel_down=2,pool_size=7):
        super().__init__()
        self.pool_size=pool_size
        self.input_size=input_size
        self.channel=channel
        self.channel_down=channel_down if channel_down>0 else channel
        self.ch_wv=nn.Conv2d(channel,channel//self.channel_down+1+channel//self.channel_down,kernel_size=(1,1))

        self.ch_wz=nn.Conv2d(channel//self.channel_down,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//self.channel_down,kernel_size=(1,1))

        stride=math.floor(input_size / pool_size ) 
        kenerl_size=input_size-(pool_size-1)*stride
        self.agp=nn.AvgPool2d(kernel_size=kenerl_size,stride=stride)
        self.sp_wz=nn.Conv2d(pool_size*pool_size,1,kernel_size=(1,1))
        self.out=nn.Identity()

    def forward(self, x):
        b, c, h, w = x.size()
        
        pool_x=self.agp(x)
        pool_x=self.ch_wv(pool_x) 
        #Channel Self-Attention
        channel_wv=pool_x[:,0:self.channel//self.channel_down,:]#bs,c//2,h,w
        channel_wq=pool_x[:,self.channel//self.channel_down:self.channel//self.channel_down+1,:] #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//self.channel_down,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1

        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wv=spatial_wv.reshape(b,c//self.channel_down,-1)
        spatial_wq=pool_x[:,self.channel//self.channel_down+1:,:] 
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,self.pool_size*self.pool_size,c//self.channel_down)
        #spatial_wq=self.softmax(spatial_wq)

        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(self.sp_wz(spatial_wz.reshape(b,self.pool_size*self.pool_size,h,w))) #bs,1,h,w
        spatial_out=spatial_weight*x
        
        out= self.out(spatial_out+channel_out)

        return out
 
        
   
class LSAM_avgpool(BaseModule):
    def __init__(self,input_size, channel=128,channel_down=2,pool_size=5):
        super().__init__()
        self.pool_size=pool_size
        self.input_size=input_size
        self.channel=channel
        self.channel_down=channel_down if channel_down>0 else channel
        self.liner=nn.Conv2d(channel,channel//self.channel_down+1+channel//self.channel_down+channel//self.channel_down,kernel_size=(1,1))
        #self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax=nn.Softmax(1)
        self.ch_wz=nn.Conv2d(channel//self.channel_down,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        #self.sp_wv=nn.Conv2d(channel,channel//self.channel_down,kernel_size=(1,1))
        #self.sp_wq=nn.Conv2d(channel,channel//self.channel_down,kernel_size=(1,1))
        #self.agp=nn.AdaptiveAvgPool2d((pool_size,pool_size))
        stride=math.floor(input_size / pool_size ) 
        kenerl_size=input_size-(pool_size-1)*stride
        self.agp=nn.AvgPool2d(kernel_size=kenerl_size,stride=stride)
        self.sp_wz=nn.Conv2d(pool_size*pool_size,1,kernel_size=(1,1))
        self.out=nn.Identity()

    def forward(self, x):
        b, c, h, w = x.size()
        #print(h,w,self.input_size)
        
        pool_x=self.liner(x)
        #Channel-only Self-Attention
        channel_wv=pool_x[:,0:self.channel//self.channel_down,:]#bs,c//2,h,w
        channel_wq=pool_x[:,self.channel//self.channel_down:self.channel//self.channel_down+1,:] #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//self.channel_down,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        #channel_wq=self.softmax(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=pool_x[:,self.channel//self.channel_down+1:self.channel//self.channel_down+1+self.channel//self.channel_down,:]#bs,c//2,h,w
        spatial_wv=spatial_wv.reshape(b,c//self.channel_down,-1)
        spatial_wq=pool_x[:,self.channel//self.channel_down+1+self.channel//self.channel_down:,:] 
        spatial_wq=self.agp(spatial_wq)
        #print(spatial_wq.shape)
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,self.pool_size*self.pool_size,c//self.channel_down)
        #spatial_wq=self.softmax(spatial_wq)

        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(self.sp_wz(spatial_wz.reshape(b,self.pool_size*self.pool_size,h,w))) #bs,1,h,w
        spatial_out=spatial_weight*x
        
        out= self.out(spatial_out+channel_out)

        return out 


      

class SpatialWeighting(BaseModule):

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out
        
class MultiConv(BaseModule):

    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False,act_cfg=None,conv_cfg=None,norm_cfg= dict(type='BN', requires_grad=True)):
        super(MultiConv, self).__init__()
        self.conv1 =  ConvModule(
                    in_chan,
                    out_chan,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    groups=groups,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None)
        self.conv2 =   ConvModule(
                    in_chan,
                    out_chan,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size//2,
                    groups=groups,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None)
    def forward(self, feat):
        feat=self.conv1(feat)
        feat=self.conv2(feat)
        
        return feat
def channel_shuffle(x, groups):
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """

    batch_size, num_channels, height, width = x.size()
    assert (num_channels % groups == 0), ('num_channels should be '
                                          'divisible by groups')
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x

class InvertedResidual(BaseModule):
    """InvertedResidual block for ShuffleNetV2 backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 convolution layer. Default: 1
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 residual=False,
                 multi3x3=False,
                 high_resolution=False,
                 use_se='polar',
                 channel_down=2,
                 pool_size=7,
                 groups=8,
                 input_size=80):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.with_cp = with_cp
        self.residual=residual
        self.multi3x3=multi3x3
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        
        branch_features = out_channels // 2
                
        if multi3x3==False:
            ConvModule3x3=ConvModule
        else:
            ConvModule3x3=MultiConv
            
        self.downsample=True if self.stride > 1 or high_resolution== True else False
        
        if self.downsample==True:
            self.branch1 = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    padding=kernel_size//2,
                    groups=in_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                ConvModule(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
            )
            if residual==True:
                self.branch3 = nn.Sequential(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
            )
        if use_se=='conv':
            self.branch2 = nn.Sequential(
                ConvModule(
                    in_channels if (self.downsample==True) else branch_features,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                ConvModule3x3(
                    branch_features,
                    branch_features,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    padding=kernel_size//2,
                    groups=branch_features,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                ConvModule(
                    branch_features,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)) 
        elif use_se=='LSAM':
            self.branch2 = nn.Sequential(
                ConvModule(
                    in_channels if (self.downsample==True) else branch_features,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                ConvModule3x3(
                    branch_features,
                    branch_features,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    padding=kernel_size//2,
                    groups=branch_features,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                LSAM_Pyramid2(input_size,branch_features,channel_down, self.pool_size),
                ConvModule(
                    branch_features,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))         
        else:  
            self.branch2 = nn.Sequential(
                ConvModule(
                    in_channels if (self.downsample==True) else branch_features,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                ConvModule3x3(
                    branch_features,
                    branch_features,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    padding=kernel_size//2,
                    groups=branch_features,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                SpatialWeighting(branch_features))    
        self.ShuffleOut = nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            if self.downsample==True:
                out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            else:
                x1, x2 = x.chunk(2, dim=1)
                out = torch.cat((x1, self.branch2(x2)), dim=1)
            out = channel_shuffle(out, 2)
            out = self.ShuffleOut(out)

            return out

        out = _inner_forward(x)

        return out


class fuse_layer(BaseModule):
    def __init__(self, low_chan, high_chan , stride=2 , conv_cfg=None,norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super(fuse_layer, self).__init__()
        self.conv1 = ConvModule(
                    low_chan,
                    high_chan,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
        self.conv2 = DepthwiseSeparableConvModule(
                    high_chan,
                    low_chan,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
        self.High = nn.Identity()
        self.Low = nn.Identity()
    
    
    def forward(self, x):
    
        feat_high=self.High(F.interpolate(self.conv1(x[0]), size=x[1].shape[2:], mode='bilinear') + x[1])
        feat_low =self.Low(F.interpolate(self.conv2(x[1]), size=x[0].shape[2:], mode='bilinear') + x[0])
        
        
        return [feat_low, feat_high]


@BACKBONES.register_module()   
class ShuffleNetV2_dual_se(BaseModule):
    """ShuffleNetV2 backbone.

    Args:
        widen_factor (float): Width multiplier - adjusts the number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 widen_factor=1.0,
                 low_channels=[116, 232, 464],
                 high_channels= [116,116],
                 out_indices=(2, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN'),
                 act_cfg=dict(type='ReLU',inplace=True),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu') ,
                 residual=False,
                 multi3x3=False,
                 kernel_size=3,
                 use_se='polar',
                 channel_down=2,
                 pool_size=5,
                 groups=8):
        super(ShuffleNetV2_dual_se, self).__init__(init_cfg)
        self.stage_blocks = [4, 8, 4]
        for index in out_indices:
            if index not in range(0, 4):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 4). But received {index}')

        if frozen_stages not in range(-1, 4):
            raise ValueError('frozen_stages must be in range(-1, 4). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.residual=residual
        self.multi3x3=multi3x3
        self.use_se=use_se
        self.channel_down=channel_down
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        if widen_factor == 0.5:
            channels = [48, 96, 192, 1024]
        elif widen_factor == 1.0:
            channels = [116, 232, 464, 1024]
        elif widen_factor == 1.5:
            channels = [176, 352, 704, 1024]
        elif widen_factor == 2.0:
            channels = [244, 488, 976, 2048]
        else:
            raise ValueError('widen_factor must be in [0.5, 1.0, 1.5, 2.0]. '
                             f'But received {widen_factor}')

        self.in_channels = 24
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers_low = nn.ModuleList()
        input_size=160
        for i, num_blocks in enumerate(self.stage_blocks):
            input_size=input_size//2
            layer = self._make_layer(low_channels[i], num_blocks, stride=2,use_se=use_se,channel_down=channel_down,groups=groups,input_size=input_size)
            self.layers_low.append(layer)

        self.layers_high= nn.ModuleList()
        self.in_channels = low_channels[0]
        for i, num_blocks in enumerate(self.stage_blocks[1:]):
            layer = self._make_layer(high_channels[i], num_blocks, stride=1,use_se=use_se,channel_down=channel_down,groups=groups,input_size=40)
            self.layers_high.append(layer)

        self.fuse_layer= nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks[1:]):
          	layer=fuse_layer(low_channels[i+1],high_channels[i],stride=2)
          	self.fuse_layer.append(layer)

        
        


    def _make_layer(self, out_channels, num_blocks, stride,use_se,channel_down,groups,input_size):
        """Stack blocks to make a layer.

        Args:
            out_channels (int): out_channels of the block.
            num_blocks (int): number of blocks.
        """
        layers = []
        for i in range(num_blocks):
            if stride == 2:
                stride = 2 if i == 0 else 1 
            high_resolution=True if i==0 and stride == 1 else False
            input_size=input_size//stride
            layers.append(
                InvertedResidual(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    kernel_size=self.kernel_size,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp,
                    residual=self.residual,
                    multi3x3=self.multi3x3,
                    high_resolution=high_resolution,
                    use_se=use_se,
                    channel_down=channel_down,
                    pool_size=self.pool_size,
                    groups=groups,
                    input_size=input_size))
                    
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x=self.layers_low[0](x)

        outs = []
        x_low=x
        x_high=x
        for i in range(len(self.layers_high)):
            x_low = self.layers_low[i+1](x_low)
            x_high = self.layers_high[i](x_high)
            x_low,x_high=self.fuse_layer[i]([x_low,x_high])
            if i==0:
              x_16=x_low
        #outs.append(x_high+F.interpolate(x_low, size=x_high.shape[2:], mode='bilinear'))    
        #outs.append(x_high) 
        return tuple([x_high,x_16,x_low])


    def train(self, mode=True):
        super(ShuffleNetV2_dual_se, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

