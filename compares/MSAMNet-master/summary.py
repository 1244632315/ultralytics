import torch
from thop import clever_format, profile
from torchsummary import summary
from model.DNANet import  DNANet, Res_CBAM_block
from model.CSAUNet import CSAUNet
from model.DnTNet import DnTNet
from model.MSAMNet import MSAMNet, AAFE, CBAM, SE, CoordAtt, Connection
import os

if __name__ == "__main__":
    #phi         = 'l'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MSAMNet(frame_length=3, fusionBlock=AAFE)
    
    
    
    summary(model.to(device), (3, 512, 512))
    
    dummy_input     = torch.randn(1, 3, 512, 512).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
