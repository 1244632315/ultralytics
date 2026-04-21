from model.utils import *

def parse_args():
    """Training Options for Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # choose model
    parser.add_argument('--model', type=str, default='MSAMNet',
                        help='model name: DNANet, CSAUNet, DnTNet, MSAMNet')
    # parameter for MSAMNet
    parser.add_argument('--fusionblock', type=str, default='AAFE',
                        help='CBAM, AAFE, None')
    parser.add_argument('--loss_gamma', type=float, default=2)
    parser.add_argument('--loss_alpha', type=float, default=0.75)
    
    # parameter for DNANet
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')
    parser.add_argument('--deep_supervision', type=str, default='False', help='True or False (model==DNANet)')


    # data and pre-process
    parser.add_argument('--root', type=str, default='/dataset',
                        help='dataset root path')
    parser.add_argument('--dataset', type=str, default='MSOD')
    parser.add_argument('--suffix', type=str, default='.tif')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when mode==Ratio')
    parser.add_argument('--split_method', type=str, default='5400_1000',
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
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--min_lr', default=1e-5,
                        type=float, help='minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='Adagrad',
                        help=' Adam, Adagrad')
    parser.add_argument('--scheduler', default='StepLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR'])
    parser.add_argument('--step_size', type=int, default=50,
                        help='learning rate drop step (default: 50 for seg, 100 for det)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--resume', type=str,
                        default = None,
                        help    = 'MSOD_MSAMNet_17_12_2024_22_54_47_wDS/f1_best_MSAMNet_MSOD_epoch.pth.tar')
    # cuda and logging
    parser.add_argument('--gpus', type=str, default='4',
                        help='Training with GPUs, you can specify 1,3 for example.')


    args = parser.parse_args()
    # make dir for save result
    args.save_dir = make_dir(args.deep_supervision, args.dataset, args.model)
    # save training log
    save_train_log(args, args.save_dir)
    # the parser
    return args
