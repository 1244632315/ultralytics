from PIL import Image, ImageOps, ImageFilter
import platform, os
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch
from torch.nn import init
from datetime import datetime
import argparse
import shutil
from  matplotlib import pyplot as plt

def load_dataset (dataset, split_method):
    train_txt = dataset + '/' + split_method + '/' + 'train.txt'
    test_txt  = dataset + '/' + split_method + '/' + 'test.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids,val_img_ids


def load_param(channel_size, backbone):
    if channel_size == 'one':
        nb_filter = [4, 8, 16, 32, 64]
    elif channel_size == 'two':
        nb_filter = [8, 16, 32, 64, 128]
    elif channel_size == 'three':
        nb_filter = [16, 32, 64, 128, 256]
    elif channel_size == 'four':
        nb_filter = [32, 64, 128, 256, 512]

    if   backbone == 'resnet_10':
        num_blocks = [1, 1, 1, 1]
    elif backbone == 'resnet_18':
        num_blocks = [2, 2, 2, 2]
    elif backbone == 'resnet_34':
        num_blocks = [3, 4, 6, 3]
    elif backbone == 'vgg_10':
        num_blocks = [1, 1, 1, 1]
    return nb_filter, num_blocks

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)


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

def save_ckpt(state, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))

def save_train_log(args, save_dir):
    dict_args=vars(args)
    args_key=list(dict_args.keys())
    args_value = list(dict_args.values())
    with open('result/%s/train_log.txt'%save_dir ,'w') as  f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')
    return

def save_model_for_det(mAP, best_mAP, r, p, best_recall, f1, best_f1, save_dir, save_prefix, lr, train_loss, test_loss, epoch, net):
    save_mAP_dir = 'result/' + save_dir + '/' + save_prefix + '_metric.log'
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(save_mAP_dir, 'a') as f:
        f.write('{} - {:04d}:\t - learning_rate: {:04f}:\t - train_loss: {:04f}:\t - test_loss: {:04f}:\t recall {:.4f}  precision {:.4f}  F1 {:.4f}  mAP {:.4f}\n' .format(dt_string, epoch,lr,train_loss, test_loss, r, p, f1, mAP))
    if mAP > best_mAP :
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'mAP': mAP,
        }, save_path='result/' + save_dir,
            filename='mAP_best' + '_' + save_prefix + '_epoch' + '.pth.tar')
    if r > best_recall and p > 0.8:
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'recall': r,
        }, save_path='result/' + save_dir,
            filename='recall_best' + '_' + save_prefix + '_epoch' + '.pth.tar')
    if f1 > best_f1 :
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'f1': f1,
        }, save_path='result/' + save_dir,
            filename='f1_best' + '_' + save_prefix + '_epoch' + '.pth.tar')
        
        
def save_model(mean_IOU, best_iou, recall, best_r, precision, f1, best_f1, save_dir, save_prefix, lr, train_loss, test_loss, epoch, net):
    save_metric_dir = 'result/' + save_dir + '/' + save_prefix + '_metric.log'
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(save_metric_dir, 'a') as f:
        f.write('{} - {:04d}:\t - learning_rate: {:04f}:\t - train_loss: {:04f}:\t - test_loss: {:04f}:\t mIoU {:.4f} recall {:.4f}  precision {:.4f}  F1 {:.4f}\n' .format(dt_string, epoch, lr, train_loss, test_loss, mean_IOU, recall, precision, f1))
    if mean_IOU > best_iou :
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'mean_IOU': mean_IOU,
        }, save_path='result/' + save_dir,
            filename='mIoU_best' + '_' + save_prefix + '_epoch' + '.pth.tar')
    if recall > best_r and precision > 0.8:
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'recall': recall,
        }, save_path='result/' + save_dir,
            filename='recall_best' + '_' + save_prefix + '_epoch' + '.pth.tar')
    if f1 > best_f1 :
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'f1': f1,
        }, save_path='result/' + save_dir,
            filename='f1_best' + '_' + save_prefix + '_epoch' + '.pth.tar')

def save_result_for_test(dataset_dir, st_model, epochs, best_iou, recall, precision ):
    with open(dataset_dir + '/' + 'value_result'+'/' + st_model +'_best_IoU.log', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epochs, best_iou))
    with open(dataset_dir + '/' +'value_result'+'/'+ st_model + '_best_other_metric.log', 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epochs))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
    return

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def make_dir(deep_supervision, dataset, model):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    if deep_supervision:
        save_dir = "%s_%s_%s_wDS" % (dataset, model, dt_string)
    else:
        save_dir = "%s_%s_%s_woDS" % (dataset, model, dt_string)
    os.makedirs('result/%s' % save_dir, exist_ok=True)
    return save_dir


def make_visulization_dir(target_image_path, target_dir):
    if os.path.exists(target_image_path):
        shutil.rmtree(target_image_path)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_image_path)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_dir)


def save_Pred(pred, wh, target_image_path, val_img_ids, num, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    img = Image.fromarray(predsss.reshape(512, 512))
    img = img.resize((wh[0], wh[1]),resample=Image.BILINEAR)  
    img.save(target_image_path + '/' + '%s' % (val_img_ids[num]) +suffix)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
