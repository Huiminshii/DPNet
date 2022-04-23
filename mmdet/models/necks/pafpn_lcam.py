# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
import math
from ..builder import NECKS


from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
class LCAM_avgpool_TD(BaseModule):
    def __init__(self,input_size, channel=128,channel_down=2,pool_size=7):
        super().__init__()
        self.pool_size=pool_size
        self.input_size=input_size
        self.channel=channel
        self.channel_down=channel_down if channel_down>0 else channel
        self.liner_low=nn.Conv2d(channel,channel//self.channel_down+channel//self.channel_down,kernel_size=(1,1))
        
        self.liner_high=nn.Conv2d(channel,1,kernel_size=(1,1))
        #self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        #self.softmax=nn.Softmax(1)
        self.ch_wz=nn.Conv2d(channel//self.channel_down,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//self.channel_down,kernel_size=(1,1))
        #self.sp_wq=nn.Conv2d(channel,channel//self.channel_down,kernel_size=(1,1))
        #self.agp=nn.AdaptiveAvgPool2d((pool_size,pool_size))
        
        stride=math.floor(input_size / pool_size ) 
        kenerl_size=input_size-(pool_size-1)*stride
        self.agp_low=nn.AvgPool2d(kernel_size=kenerl_size,stride=stride)
        
        stride=math.floor(input_size*2 / pool_size ) 
        kenerl_size=input_size*2-(pool_size-1)*stride
        self.agp_high=nn.AvgPool2d(kernel_size=kenerl_size,stride=stride)        
        self.sp_wz=nn.Conv2d(pool_size*pool_size,1,kernel_size=(1,1))
        self.out=nn.Identity()

    def forward(self, x_high,x_low):
        b, c, h, w = x_high.size()
        #print(h,w,self.input_size)

        x_low=self.agp_low(x_low)        
        x_low=self.liner_low(x_low)
        

        x_high_pool=self.agp_high(x_high)

        x_high_pool=self.liner_high(x_high_pool)
        
        #print(self.input_size,x_low.shape,x_high_pool.shape)
        #Channel-only Self-Attention
        channel_wv=x_low[:,0:self.channel//self.channel_down,:]#bs,c//2,h,w
        channel_wq=x_high_pool #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//self.channel_down,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        #channel_wq=self.softmax(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x_high

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x_high)
        spatial_wv=spatial_wv.reshape(b,c//self.channel_down,-1)
        #spatial_wq=pool_x[:,self.channel//self.channel_down+1+self.channel//self.channel_down:,:] 
        spatial_wq=x_low[:,self.channel//self.channel_down:self.channel//self.channel_down+1+self.channel//self.channel_down,:]
        #print(spatial_wq.shape)

        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,self.pool_size*self.pool_size,c//self.channel_down)
        #spatial_wq=self.softmax(spatial_wq)
        
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w 
        spatial_weight=self.sigmoid(self.sp_wz(spatial_wz.reshape(b,self.pool_size*self.pool_size,h,w))) #bs,1,h,w
        spatial_out=spatial_weight*x_high
        
        out= self.out(spatial_out+channel_out+x_high)

        return out  
class LCAM_avgpool_BU(BaseModule):
    def __init__(self,input_size, channel=128,channel_down=2,pool_size=7):
        super().__init__()
        self.pool_size=pool_size
        self.input_size=input_size
        self.channel=channel
        self.channel_down=channel_down if channel_down>0 else channel
        self.liner_low=nn.Conv2d(channel,channel//self.channel_down+channel//self.channel_down,kernel_size=(1,1))
        
        self.liner_high=nn.Conv2d(channel,1,kernel_size=(1,1))
        #self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        #self.softmax=nn.Softmax(1)
        self.ch_wz=nn.Conv2d(channel//self.channel_down,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//self.channel_down,kernel_size=(1,1))
        #self.sp_wq=nn.Conv2d(channel,channel//self.channel_down,kernel_size=(1,1))
        #self.agp=nn.AdaptiveAvgPool2d((pool_size,pool_size))
        
        stride=math.floor(input_size / pool_size ) 
        kenerl_size=input_size-(pool_size-1)*stride
        self.agp_low=nn.AvgPool2d(kernel_size=kenerl_size,stride=stride)
        
        stride=math.floor(input_size*2 / pool_size ) 
        kenerl_size=input_size*2-(pool_size-1)*stride
        self.agp_high=nn.AvgPool2d(kernel_size=kenerl_size,stride=stride)        
        #self.sp_wz=nn.Conv2d(pool_size*pool_size,1,kernel_size=(1,1))
        self.out=nn.Identity()
        
        self.sp_wz=ConvModule(
            pool_size*pool_size,
            1,
            kernel_size=5,
            stride=2,
            padding=5 // 2,
            groups=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='Swish'))

    def forward(self, x_high,x_low):
        b, c, h, w = x_high.size()
        #print(h,w,self.input_size)
        
        x_low_pool=self.agp_low(x_low)        
        x_low_pool=self.liner_low(x_low_pool)
        
        x_high_pool=self.agp_high(x_high)
        x_high_pool=self.liner_high(x_high_pool)
        #Channel-only Self-Attention
        channel_wv=x_low_pool[:,0:self.channel//self.channel_down,:]#bs,c//2,h,w
        channel_wq=x_high_pool #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//self.channel_down,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        #channel_wq=self.softmax(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x_low

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x_high)
        spatial_wv=spatial_wv.reshape(b,c//self.channel_down,-1)
        spatial_wq=x_low_pool[:,self.channel//self.channel_down:self.channel//self.channel_down+1+self.channel//self.channel_down,:]
        #print(spatial_wq.shape)
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,self.pool_size*self.pool_size,c//self.channel_down)
        #spatial_wq=self.softmax(spatial_wq)

        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(self.sp_wz(spatial_wz.reshape(b,self.pool_size*self.pool_size,h,w))) #bs,1,h,w
        spatial_out=spatial_weight*x_low
        
        out= self.out(spatial_out+channel_out+x_low)

        return out  
class MultiConv(BaseModule):

    def __init__(self, in_chan, out_chan, kernel_size=5, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, act_cfg=dict(type='Swish'), conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True), inplace=False,channel_down=2):
        super(MultiConv, self).__init__()
        self.conv1_1 = ConvModule(
            in_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            out_chan,
            out_chan,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.conv1_2 = ConvModule(
            out_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        identity=x
        x = self.conv1_1(x)
        x = self.conv2(x)
        x = self.conv1_2(x)        
        return x+identity
        
@NECKS.register_module()
class PAFPN_LCAM(BaseModule):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 channel_down=8,
                 pool_size=5,
                 use_depthwise=False,
                 kernel_size=5,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 lateral=True,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(PAFPN_LCAM, self).__init__()
        # add extra bottom up pathway
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.lateral=lateral
        self.kernel_size=kernel_size

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'
        # add extra bottom up pathway
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            fpn_conv = MultiConv(
                in_channels[i] if not self.lateral else out_channels,
                out_channels,
                self.kernel_size,
                padding=1,
                groups=out_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = MultiConv(
                out_channels,
                out_channels,
                self.kernel_size,
                groups=out_channels,
                padding=2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            if self.lateral:
                l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)
            else:
                l_conv= nn.Identity()
            self.lateral_convs.append(l_conv)
            self.pafpn_convs.append(pafpn_conv)
            self.fpn_convs.append(fpn_conv)
        self.downsample_attn = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        input_size1=[0,10,20]
        input_size2=[0,20,10]
        
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_attn = LCAM_avgpool_TD(input_size2[i],out_channels,channel_down,pool_size)
            up_attn = LCAM_avgpool_BU(input_size2[i],out_channels,channel_down,pool_size)
            self.downsample_attn.append(d_attn)
            self.up_attn.append(up_attn)
            
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = self.downsample_attn[i-1](laterals[i-1],laterals[i])

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] = self.up_attn[i](inter_outs[i],inter_outs[i+1])#inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i]++inputs[i])
  
        outs = [self.pafpn_convs[i](inter_outs[i]) for i in range(0, used_backbone_levels)]
        


        # part 3: add extra levels

        return tuple(outs)
