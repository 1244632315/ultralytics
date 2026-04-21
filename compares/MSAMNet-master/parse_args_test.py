from model.utils import *

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # choose model
    parser.add_argument('--model', type=str, default='MSAMNet',
                        help='model name: DNANet CSAUNet MSAMNet DnTNet')
    # parameter for MSAMNet
    parser.add_argument('--fusionblock', type=str, default='AAFE',
                        help='CBAM, AAFE, None')
    parser.add_argument('--loss_gamma', type=int, default=2)
    parser.add_argument('--loss_alpha', type=float, default=0.75)
    
    # parameter for DNANet
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')
    parser.add_argument('--deep_supervision', type=str, default='False', help='True or False (model==DNANet)')


    # data and pre-process
    parser.add_argument('--dataset', type=str, default='MSOD')
    parser.add_argument('--root', type=str, default='./dataset',
                        help='dataset root path')
    parser.add_argument('--suffix', type=str, default='.tif')

    parser.add_argument('--st_model', type=str, default='MSOD_MSAMNet_17_12_2024_22_54_47_wDS',
                        help='pretrained result dir')
    parser.add_argument('--model_dir', type=str,
                        default = 'MSOD_MSAMNet_17_12_2024_22_54_47_wDS/f1_best_MSAMNet_MSOD_epoch.pth.tar',
                        help    = 'path for model pth')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--split_method', type=str, default='50_50',
                        help='50_50, 10000_100(for NUST-SIRST)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--T_frame', type=int, default=3,
                        help='T frames for input')
    parser.add_argument('--input_size', type=int, default=512,
                        help='input image size')

    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 110)')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='3',
                        help='Training with GPUs, you can specify 1,3 for example.')

    # ROC threshold
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='crop image size')
    parser.add_argument('--cof_thr', type=float, default=0.3,
                        help='confidence threshold')


    args = parser.parse_args()

    # the parser
    return args
