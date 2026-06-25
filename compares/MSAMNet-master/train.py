import time
# torch and visulization
from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from parse_args_train import  parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.dataloader import *

# model
from model.DNANet import  Res_CBAM_block, DNANet
from model.CSAUNet import CSAUNet
from model.DnTNet import DnTNet
from model.MSAMNet import MSAMNet, AAFE, CBAM, Connection, SE, CoordAtt
import cv2

fusionblock = {'CBAM':CBAM,
             'AAFE':AAFE,
             'CA':CoordAtt,
             'SE':SE,
             'None':Connection
}

def cal_centroid_coord(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    return num_labels - 1, centroids[1:] - 0.5


def compute_loss_and_pred(model_output, labels, focal_loss):
    if isinstance(model_output, (list, tuple)):
        preds = list(model_output)
        loss = 0
        for pred_item in preds:
            loss += SoftIoULoss(pred_item, labels)
            loss += focal_loss(pred_item, labels)
        loss /= len(preds)
        pred = preds[-1]
        return loss, pred
    pred = model_output
    loss = SoftIoULoss(pred, labels) + focal_loss(pred, labels)
    return loss, pred


def load_split_ids(split_path):
    img_ids = []
    with open(split_path, "r") as f:
        line = f.readline()
        while line:
            sample_id = line.split("\n")[0]
            if sample_id:
                img_ids.append(sample_id)
            line = f.readline()
    return img_ids


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.mIoU = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT for any MSOD-style dataset directory.
        dataset_dir = args.root + '/' + args.dataset
        train_txt = dataset_dir + '/' + 'train.txt'
        test_txt = dataset_dir + '/' + 'val.txt'
        if not os.path.exists(train_txt):
            raise FileNotFoundError('Missing train split file: %s' % train_txt)
        if not os.path.exists(test_txt):
            raise FileNotFoundError('Missing val split file: %s' % test_txt)
        train_img_ids = load_split_ids(train_txt)
        val_img_ids = load_split_ids(test_txt)
        trainset = MSODataset(dataset_dir, mode='train', img_ids=train_img_ids, input_size=args.input_size, num_frame=args.T_frame, suffix=args.suffix)
        testset = MSODataset(dataset_dir, mode='val', img_ids=val_img_ids, input_size=args.input_size, num_frame=args.T_frame, suffix=args.suffix)
        
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1, input_channels=args.T_frame, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model   == 'CSAUNet':
            model       = CSAUNet(input_channels=args.T_frame)
        elif args.model   == 'DnTNet':
            model       = DnTNet(seq_length=args.T_frame)
        elif args.model   == 'MSAMNet':
            model       = MSAMNet(frame_length=args.T_frame, fusionBlock=fusionblock[args.fusionblock])
        
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model
        self.start_epoch=args.start_epoch
    
        # Evaluation metrics
        self.best_iou       = 0
        self.best_f1       = 0
        self.best_recall    = 0
        self.best_precision = 0

        self.Floss = FocalLoss(gamma=args.loss_gamma, alpha=args.loss_alpha)

        # Optimizer and lr scheduling
        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-2)
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        elif args.scheduler == 'StepLR':
            self.scheduler = lr_scheduler.StepLR( self.optimizer, step_size=args.step_size, gamma=0.1)
        elif args.scheduler   == 'ReduceLROnPlateau':
            self.scheduler  = lr_scheduler.ReduceLROnPlateau( self.optimizer, mode='min', factor=0.5, patience=10, min_lr=args.min_lr)
        
        if args.resume is not None:
            # Load trained model
            print("Loading Resumed Model")
            checkpoint        = torch.load('result/' + args.resume, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.start_epoch = checkpoint.get('epoch', self.start_epoch)
                self.train_loss = checkpoint.get('loss', self.train_loss)
                if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler and checkpoint.get('scheduler_state_dict') is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'random_state' in checkpoint:
                    random.setstate(checkpoint['random_state'])
                if 'torch_state' in checkpoint:
                    torch.random.set_rng_state(checkpoint['torch_state'])
                if 'np_state' in checkpoint:
                    np.random.set_state(checkpoint['np_state'])
            else:
                state_dict = checkpoint.get('state_dict')
                if state_dict is None:
                    raise KeyError('Checkpoint must contain model_state_dict or state_dict.')
                self.model.load_state_dict(state_dict)
                self.start_epoch = args.start_epoch
                self.train_loss = checkpoint.get('loss', 0)

    # Training
    def training(self,epoch):

        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        start = time.time()
        for i, ( data, labels) in enumerate(tbar):
            load = time.time()
            data   = data.cuda()
            labels = labels.cuda()
            loss, pred = compute_loss_and_pred(self.model(data), labels, self.Floss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            compute = time.time()
            tbar.set_description('Epoch %d, training loss %.4f, dataload time %f, compute time %f' % (epoch, losses.avg, load-start, compute-load))
            start = time.time()
        self.train_loss = losses.avg
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'loss': self.train_loss,
        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        'random_state': random.getstate(),
        'torch_state': torch.random.get_rng_state(),
        'np_state': np.random.get_state(),
        }
        torch.save(checkpoint, os.path.join('result/' + self.save_dir, 'last_epoch.pth.tar'))

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        losses = AverageMeter()
        tp_num = 0
        fp_num = 0
        pred_num = 0
        gt_num = 0

        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                batchsize, _, _, _ = labels.size()
                loss, pred = compute_loss_and_pred(self.model(data), labels, self.Floss)
                losses.update(loss.item(), pred.size(0))
                for batch_i in range(batchsize): 
                    predsss = np.array((pred[batch_i,...] > 0).cpu()).astype('int64') * 255
                    predsss = np.uint8(predsss)
                    img = predsss.reshape(512, 512)
                    num_preds, pred_coords = cal_centroid_coord(img)
                    targets = np.array((labels[batch_i,...] > 0).cpu()).astype('int64') * 255
                    targets = np.uint8(targets)
                    img = targets.reshape(512, 512)
                    num_targets, target_coords = cal_centroid_coord(img)
                    pred_num += num_preds
                    gt_num += num_targets
                    gt_matched = np.zeros(num_targets)
                    if num_targets > 0:
                        for pred_i in range(num_preds):
                            distances = [np.linalg.norm(pred_coords[pred_i]- target_coords[t_i]) for t_i in range(num_targets)]
                            min_dis = min(distances)
                            if min_dis <= 3:
                                min_idx = distances.index(min_dis)
                                if not gt_matched[min_idx]:
                                    tp_num += 1
                                    gt_matched[min_idx] = 1
                                else:
                                    fp_num += 1
                            else:
                                fp_num += 1
                    else:
                        fp_num += num_preds
                self.mIoU.update(pred, labels)
                pix_recall, pix_precision, mean_IOU = self.mIoU.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f, pix_recall: %.4f, pix_precision: %.4f' % (epoch, losses.avg, mean_IOU, pix_recall, pix_precision))
            test_loss=losses.avg
            recall      = tp_num / (gt_num  + 0.001)
            precision   = tp_num / (pred_num + 0.001)
            f1 = 2 * precision * recall / (precision + recall + 0.001)
        # save high-performance model
        save_model(mean_IOU, self.best_iou, recall, self.best_recall, precision, f1, self.best_f1, self.save_dir, self.save_prefix,
                   self.optimizer.state_dict()['param_groups'][0]['lr'], self.train_loss, test_loss, epoch, self.model.state_dict())
        if mean_IOU > self.best_iou:
            self.best_iou = mean_IOU
        if recall > self.best_recall and precision > 0.8:
            self.best_recall = recall
        if f1 > self.best_f1:
            self.best_f1 = f1

def main(args):
    trainer = Trainer(args)
    for epoch in range(trainer.start_epoch, args.epochs):
        st_time = time.time()
        trainer.training(epoch)
        train_time = time.time()
        trainer.testing(epoch)
        test_time = time.time()
        print("train time: %f test time: %f"%(train_time-st_time, test_time-train_time))
        if args.scheduler   == 'CosineAnnealingLR':
            trainer.scheduler.step()
        elif args.scheduler   == 'ReduceLROnPlateau':
            trainer.scheduler.step(trainer.test_loss)
        elif args.scheduler   == 'StepLR':
            trainer.scheduler.step()


if __name__ == "__main__":
    args = parse_args()    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
