import torch
import torch.nn as nn
from model.layers import regional_routing_attention_torch
import math
import torch.nn.functional as F
from torch import LongTensor, Tensor

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=64):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat_out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat_out)
        return self.sigmoid(out) * x
    
class CBAM(nn.Module):
    def __init__(self, out_channels):
        super(CBAM, self).__init__()
        self.ca = ChannelAttentionModule(out_channels)
        self.sa = SpatialAttentionModule()
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
    
class AAFE(nn.Module):
    def __init__(self, lamda=1e-5):
        super(AAFE, self).__init__()
        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        mean = torch.mean(x, dim=[-2,-1], keepdim=True)
        var = torch.sum(torch.pow((x - mean), 2), dim=[-2, -1], keepdim=True) / n
        e_t = torch.pow((x - mean), 2) / (4 * (var + self.lamda)) + 0.5
        out = self.sigmoid(e_t) * x
        return out
    
class SE(nn.Module):
    def __init__(self, c1, ratio=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        #c*1*W
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class Connection(nn.Module):
    def __init__(self, channel):
        super(Connection, self).__init__()
    def forward(self, x):
        return x
    
class newMotionAwareBlock(nn.Module):
    def __init__(self, seq_len, in_channels):
        super().__init__()
        self.seqlen = seq_len
        self.dim = in_channels
        self.ca = ChannelAttentionModule(in_channels=in_channels)
        self.dconv3_3 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels)
        self.dconv3_3_d2 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=2,dilation=2,groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,7),padding=(0,3),groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(7,1),padding=(3,0),groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,11),padding=(0,5),groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(11,1),padding=(5,0),groups=in_channels)

        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=(1,1),padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        seqlen = self.seqlen
        batchseq, channel, height, width = x.size()
        x = x.view(-1, seqlen, channel, height, width)
        weights = torch.zeros([1,seqlen,1,1,1], device=x.device)  
        for t in range(seqlen):
            weights[:,t,...] = math.e ** (-seqlen+t+1)
        inputs = torch.sum(x * weights, dim=1)
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        
        inputs = self.ca(inputs)

        x_0 = self.dconv3_3(inputs)
        x_1 = self.dconv3_3_d2(inputs)
        x_2 = self.dconv1_7(x_0)
        x_2 = self.dconv7_1(x_2)
        x_3 = self.dconv1_11(x_0)
        x_3 = self.dconv11_1(x_3)
        x = x_0 + x_1 + x_2 + x_3
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out

class MotionAwareBlock(nn.Module):
    def __init__(self, seqlen, dim, num_heads=8, n_win=7, qk_scale=None, topk=4,  side_dwconv=3, auto_pad=False):
        super(MotionAwareBlock, self).__init__()
        self.seqlen = seqlen
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5 # NOTE: to be consistent with old models.

        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        
        self.topk = topk
        self.n_win = n_win  # number of windows per row/col

        self.q_linear = nn.Conv2d(self.dim, self.dim, kernel_size=1)
        self.kv_linear = nn.Conv2d(self.dim, 2*self.dim, kernel_size=1)
        self.output_linear = nn.Conv2d(self.dim, self.dim, kernel_size=1)
        self.attn_fn = regional_routing_attention_torch
       
    def forward(self, x):
        seqlen = self.seqlen
        batchseq, channel, height, width = x.size()
        x = x.view(-1, seqlen, channel, height, width)
        frame_now = x[:, -1, ...]
        frame_his = x[:, :-1, ...]
        weights = torch.zeros([1,seqlen-1,1,1,1], device=x.device)  
        for t in range(seqlen-1):
            weights[:,t,...] = math.e ** (-seqlen-t+2)
        frame_his = torch.sum(frame_his * weights, dim=1)

        
        region_size = (height//self.n_win, width//self.n_win)
        q = self.q_linear.forward(frame_now)
        kv = self.kv_linear.forward(frame_his)
        k, v = kv.chunk(2, dim=1)
        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False) # nchw
        q_r:Tensor = q_r.permute(0, 2, 3, 1).flatten(1, 2) # n(hw)c
        k_r:Tensor = k_r.flatten(2, 3) # nc(hw)
        a_r = q_r @ k_r # n(hw)(hw), adj matrix of regional graph
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1) # n(hw)k long tensor
        idx_r:LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1) 

        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size
                                       )
        
        output = output + self.lepe(v) # ncHW
        output = self.output_linear(output) # ncHW

        return output

channels = [64, 128, 256, 512]
# channels = [32, 64, 128, 256]
# 定义主网络结构
class MSAMNet(nn.Module):
    def __init__(self, frame_length = 1, fusionBlock=Connection, input_channels = 1):
        super(MSAMNet, self).__init__()
        self.seq_len = frame_length
        if frame_length > 1:
            self.multiframe = True
            self.temporal4 = newMotionAwareBlock(frame_length, channels[3])  
            #self.temporal4 = MotionAwareBlock(frame_length, channels[3])   
        else:
            self.multiframe = False
        
        self.enc1 = self._conv_block(input_channels, channels[0])
        self.enc2 = self._conv_block(channels[0], channels[1])
        self.enc3 = self._conv_block(channels[1], channels[2])
        self.enc4 = self._conv_block(channels[2], channels[3])

        self.dec3 = nn.Sequential(
            nn.Conv2d(channels[3], channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(
            nn.Conv2d(channels[1], channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True))

        self.up3 = self._up_block(channels[3], channels[2])
        self.up2 = self._up_block(channels[2], channels[1])
        self.up1 = self._up_block(channels[1], channels[0])

        self.fusion1 = fusionBlock(channels[0])
        self.fusion2 = fusionBlock(channels[1])
        self.fusion3 = fusionBlock(channels[2])
        self.fusion4 = Connection(channels[3])
        # self.fusion4 = fusionBlock(channels[3])
        
        self.final_conv = nn.Conv2d(channels[0], 2, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        if self.multiframe:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2,   mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        if self.multiframe:
            batch, seqchannel, height, width = x.size()
            x = x.view(batch*self.seq_len, -1, height, width)

        x1 = self.enc1(x)
        x2 = self.enc2(nn.MaxPool2d(2)(x1))
        x3 = self.enc3(nn.MaxPool2d(2)(x2))
        x4 = self.enc4(nn.MaxPool2d(2)(x3))

        if self.multiframe:
            t4 = self.temporal4(x4)
            x4 = x4.view(batch, self.seq_len, -1, height // 8, width // 8)[:,-1,...]
            
            x3 = self.fusion3(x3.view(batch, self.seq_len, -1, height // 4, width // 4)[:,-1,...])
            x2 = self.fusion2(x2.view(batch, self.seq_len, -1, height // 2, width // 2)[:,-1,...])
            x1 = self.fusion1(x1.view(batch, self.seq_len, -1, height, width)[:,-1,...])
            x4 = x4 + t4
            
        else:
            x4 = self.fusion4(x4)
            x3 = self.fusion3(x3)
            x2 = self.fusion2(x2)
            x1 = self.fusion1(x1)

        d3 = self.up3(x4)
        d3 = torch.cat([x3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([x2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([x1, d1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return torch.unsqueeze(out[:,-1,:,:],dim=1)
    
# Initialize and test the network
if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    model = MSAMNet(input_channels=1,frame_length=3)
    x = torch.randn(4, 3, 512, 512)  # Example input
    mode = model.cuda()
    x = x.cuda()
    y = model(x)
    print(y.shape)  # Output shape
