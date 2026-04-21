import torch
import torch.nn as nn

# SPD-Conv
class SPDConv(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=5, s=1, p=2, g=1, d=1, act=True):
        super(SPDConv, self).__init__()
        c1 = c1 * 4
        self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p, d), groups=g, dilation=d, bias=False)
        # self.bn = nn.BatchNorm2d(c2)
        self.gn = nn.GroupNorm(num_groups=16, num_channels=c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def autopad(self, k, p=None, d=1):  # kernel, padding, dilation
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        return p
 
    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        # return self.act(self.bn(self.conv(x)))
        return self.act(self.gn(self.conv(x)))


class DSPConv(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=5, s=1, p=2, g=1, d=1, act=True):
        super(DSPConv, self).__init__()
        self.dimension = c1
        c2 = c2 * 4
        self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p, d), groups=g, dilation=d, bias=False)
        # self.bn = nn.BatchNorm2d(c2)
        self.gn = nn.GroupNorm(num_groups=16, num_channels=c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def autopad(self, k, p=None, d=1):  # kernel, padding, dilation
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        return p
 
    def forward(self, x):
        c = self.dimension
        # x = self.act(self.bn(self.conv(x)))
        x = self.act(self.gn(self.conv(x)))
        batch, channel, height, width = x.size()
        out = torch.zeros(batch, c, height*2, width*2, device=x.device)
        out[..., ::2, ::2] = x[..., :c, :, :]
        out[..., 1::2, ::2] = x[..., c:c*2, :, :]
        out[..., ::2, 1::2] = x[..., c*2:c*3, :, :]
        out[..., 1::2, 1::2] = x[..., c*3:c*4, :, :]
        return out


# Depthwise Separable Convolution (DW-Conv)
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(DWConv, self).__init__()
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.gn = nn.GroupNorm(num_groups=16, num_channels=out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        # x = self.bn(x)
        x = self.gn(x)
        return self.activation(x)

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        self.dw_conv_3 = DWConv(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.dw_conv_5 = DWConv(in_channels, in_channels, kernel_size=5, stride=1, padding=2)
        self.dw_conv_7 = DWConv(in_channels, in_channels, kernel_size=7, stride=1, padding=3)
        self.conv_1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):
        x3 = self.dw_conv_3(x)
        x5 = self.dw_conv_5(x)
        x7 = self.dw_conv_7(x)
        x_concat = torch.cat([x3, x5, x7], dim=1)  # Concatenate along the channel dimension
        return self.conv_1(x_concat)


# MFF-LSTM
class MFFLSTMCell(nn.Module):
    def __init__(self, in_channels):
        super(MFFLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.mff = MultiScaleFeatureFusion(in_channels)

        # LSTM gates
        self.f_gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.i_gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.g_gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.o_gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x_t, h_t_minus_1, c_t_minus_1):
        # Multi-scale feature fusion
        fused_features = self.mff(x_t + h_t_minus_1)

        # Compute LSTM gates
        f_t = torch.sigmoid(self.f_gate(fused_features))  # Forget gate
        i_t = torch.sigmoid(self.i_gate(fused_features))  # Input gate
        g_t = torch.tanh(self.g_gate(fused_features))     # Cell candidate
        o_t = torch.sigmoid(self.o_gate(fused_features))  # Output gate

        # Update cell state and hidden state
        c_t = f_t * c_t_minus_1 + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


# Example: Using MFF-LSTM in a network
class MFFLSTM(nn.Module):
    def __init__(self, in_channels, seq_length=5):
        super(MFFLSTM, self).__init__()
        self.cell = MFFLSTMCell(in_channels)
        self.seq_len = seq_length

    def forward(self, x_seq):
        _, channels, height, width = x_seq.size()
        x_seq = x_seq.view(-1, self.seq_len, channels, height, width)
        batch_size, seq_length, channels, height, width = x_seq.size()
        h_t = torch.zeros(batch_size, channels, height, width, device=x_seq.device)
        c_t = torch.zeros(batch_size, channels, height, width, device=x_seq.device)

        outputs = []
        for t in range(seq_length):
            x_t = x_seq[:, t, :, :, :]
            h_t, c_t = self.cell(x_t, h_t, c_t)
            outputs.append(h_t.unsqueeze(1))
        output = torch.cat(outputs, dim=1)
        output = output.view(-1, channels, height, width)

        return output  # Concatenate along the sequence dimension
    
# Full DnT-Net
class DnTNet(nn.Module):
    '''
    input: (batch, seq, channels, height, width)
    '''
    def __init__(self, seq_length):
        super(DnTNet, self).__init__()
        # Head
        self.head_conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3)
        self.head_conv2 = nn.Conv2d(1, 8, kernel_size=13, stride=1, padding=6)
        self.head_conv3 = nn.Conv2d(8+8, 16, kernel_size=1)

        # Down-sampling stages
        self.enc1 = nn.Sequential(SPDConv(16, 16), MFFLSTM(16, seq_length), DWConv(16, 32))
        self.enc2 = nn.Sequential(SPDConv(32, 32), MFFLSTM(32, seq_length), DWConv(32, 64))
        self.enc3 = nn.Sequential(SPDConv(64, 64), MFFLSTM(64, seq_length), DWConv(64, 128))
        
        # Up-sampling stages
        self.up1 = DSPConv(32, 32)
        self.up2 = DSPConv(64, 64)
        self.up3 = DSPConv(128, 128)
        self.dec1 = nn.Sequential(DWConv(32 + 16, 16), MFFLSTM(16, seq_length))
        self.dec2 = nn.Sequential(DWConv(64 + 32, 32), MFFLSTM(32, seq_length))
        self.dec3 = nn.Sequential(DWConv(128 + 64, 64), MFFLSTM(64, seq_length))
        
        # Tail
        self.tail = nn.Sequential(
            DWConv(16, 16, kernel_size=13, stride=1, padding=6), nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x):
        batch, seq, height, width = x.size()
        x = x.view(batch*seq, -1, height, width)
        # Head
        x1 = self.head_conv1(x)
        x2 = self.head_conv2(x)
        x = self.head_conv3(torch.cat([x1, x2], dim=1))
        
        # Down-sampling
        d1 = self.enc1(x)
        d2 = self.enc2(d1)
        d3 = self.enc3(d2)
        
        # Up-sampling with skip connections
        u3 = self.dec3(torch.cat([self.up3(d3), d2], dim=1))
        u2 = self.dec2(torch.cat([self.up2(u3), d1], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), x], dim=1))
        
        # Tail
        out = self.tail(u1)
        out = out.view(batch, seq, height, width)
        out = out[:, -1, :, :]
        
        return out.unsqueeze(1)


# Initialize and test the network
if __name__ == "__main__":
    model = DnTNet(seq_length=3)
    x = torch.randn(4, 3, 512, 512)  # Example input
    y = model(x)
    print(y.shape)  # Output shape
